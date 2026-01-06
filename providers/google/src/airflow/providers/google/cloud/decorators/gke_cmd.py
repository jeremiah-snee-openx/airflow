from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.providers.google.cloud.operators.kubernetes_engine import GKEStartPodOperator

# "compat.sdk" keeps this usable across Airflow 2.x/3.x provider stacks.
from airflow.providers.common.compat.sdk import (
    DecoratedOperator,
    TaskDecorator,
    context_merge,
    task_decorator_factory,
)
from airflow.utils.operator_helpers import determine_kwargs

if TYPE_CHECKING:
    from airflow.sdk import Context


# ---- Defaults resolution ----

def _get_var(key: str) -> str | None:
    try:
        v = Variable.get(key)
        return v.strip() if isinstance(v, str) and v.strip() else None
    except Exception:
        return None


def _get_conn_extra(conn_id: str, key: str) -> str | None:
    # Put your defaults in Connection "Extra" as JSON, e.g.:
    # {"project_id":"...", "location":"us-central1", "cluster_name":"...", "namespace":"default"}
    conn = BaseHook.get_connection(conn_id)
    val = conn.extra_dejson.get(key)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def resolve_gke_defaults(
    *,
    project_id: str | None,
    location: str | None,
    cluster_name: str | None,
    namespace: str | None,
    gcp_conn_id: str | None,
    defaults_conn_id: str | None,
) -> dict[str, str]:
    """
    Precedence:
      explicit arg > conn extra > variable > hard default
    """
    # Connection id that stores cluster defaults (NOT credentials).
    # You can set this as a Variable too.
    defaults_conn_id = defaults_conn_id or _get_var("MYCO_GKE_DEFAULTS_CONN_ID")

    def pick(explicit: str | None, var_key: str, extra_key: str, fallback: str | None = None) -> str | None:
        if explicit:
            return explicit
        if defaults_conn_id:
            v = _get_conn_extra(defaults_conn_id, extra_key)
            if v:
                return v
        v = _get_var(var_key)
        if v:
            return v
        return fallback

    resolved = {
        "project_id": pick(project_id, "MYCO_GKE_PROJECT_ID", "project_id"),
        "location": pick(location, "MYCO_GKE_LOCATION", "location"),
        "cluster_name": pick(cluster_name, "MYCO_GKE_CLUSTER_NAME", "cluster_name"),
        "namespace": pick(namespace, "MYCO_GKE_NAMESPACE", "namespace", fallback="default") or "default",
        "gcp_conn_id": pick(gcp_conn_id, "MYCO_GCP_CONN_ID", "gcp_conn_id", fallback="google_cloud_default")
        or "google_cloud_default",
    }

    missing = [k for k in ("project_id", "location", "cluster_name") if not resolved[k]]
    if missing:
        raise AirflowException(
            "Missing required GKE defaults: "
            + ", ".join(missing)
            + ". Provide them to @task.gke_cmd(...) or set Variables "
            + "(MYCO_GKE_PROJECT_ID / MYCO_GKE_LOCATION / MYCO_GKE_CLUSTER_NAME) "
            + "or put them in Connection Extra JSON via MYCO_GKE_DEFAULTS_CONN_ID."
        )

    return resolved


# ---- Decorated operator ----

class _GKECmdDecoratedOperator(DecoratedOperator, GKEStartPodOperator):
    """
    @task.kubernetes_cmd-style:
      - python_callable runs in Airflow to generate list[str]
      - the pod runs that command
    """
    custom_operator_name = "@task.gke_cmd"
    template_fields: Sequence[str] = tuple({"op_args", "op_kwargs", *GKEStartPodOperator.template_fields})
    overwrite_rtif_after_execution: bool = True

    def __init__(
        self,
        *,
        python_callable: Callable[..., list[str]],
        args_only: bool = False,
        multiple_outputs: bool | None = None,
        defaults_conn_id: str | None = None,
        # GKE fields (may be None; we’ll resolve)
        project_id: str | None = None,
        location: str | None = None,
        cluster_name: str | None = None,
        namespace: str | None = None,
        gcp_conn_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.args_only = args_only
        self._multiple_outputs = bool(multiple_outputs)

        # Match upstream kubernetes_cmd behavior: cmds/arguments are “owned” by the decorator.
        cmds = kwargs.pop("cmds", None)
        arguments = kwargs.pop("arguments", None)
        if cmds is not None or arguments is not None:
            warnings.warn(
                f"`cmds`/`arguments` are ignored by {self.custom_operator_name}. "
                "Return list[str] from the decorated function instead.",
                UserWarning,
                stacklevel=3,
            )

        resolved = resolve_gke_defaults(
            project_id=project_id,
            location=location,
            cluster_name=cluster_name,
            namespace=namespace,
            gcp_conn_id=gcp_conn_id,
            defaults_conn_id=defaults_conn_id,
        )

        # Let users omit `name`; default something deterministic-ish.
        name = kwargs.pop("name", f"gke-cmd-{python_callable.__name__}")
        random_name_suffix = kwargs.pop("random_name_suffix", True)

        super().__init__(
            python_callable=python_callable,
            project_id=resolved["project_id"],
            location=resolved["location"],
            cluster_name=resolved["cluster_name"],
            namespace=resolved["namespace"],
            gcp_conn_id=resolved["gcp_conn_id"],
            name=name,
            random_name_suffix=random_name_suffix,
            cmds=None,
            arguments=None,
            **kwargs,
        )

    def execute(self, context: Context):
        # Render templates before generating commands (so args/macros can be used).
        self.render_template_fields(context)

        generated = self._generate_cmds(context)

        if self.args_only:
            self.cmds = []
            self.arguments = generated
        else:
            self.cmds = generated
            self.arguments = []

        # Render again so returned strings can be templated too.
        self.render_template_fields(context)

        result = super().execute(context)

        # Optional: push dict keys as separate XComs (TaskFlow-ish ergonomics)
        if self._multiple_outputs and isinstance(result, Mapping):
            ti = context["ti"]
            for k, v in result.items():
                ti.xcom_push(key=str(k), value=v)

        return result

    def _generate_cmds(self, context: Context) -> list[str]:
        context_merge(context, self.op_kwargs)
        kwargs = determine_kwargs(self.python_callable, self.op_args, context)
        out = self.python_callable(*self.op_args, **kwargs)

        if not isinstance(out, list) or not out or not all(isinstance(x, str) for x in out):
            raise TypeError("Decorated function must return a non-empty list[str]")
        return out


def gke_cmd_task(
    python_callable: Callable[..., list[str]] | None = None,
    *,
    args_only: bool = False,
    multiple_outputs: bool | None = None,
    defaults_conn_id: str | None = None,
    **kwargs: Any,
) -> TaskDecorator:
    """
    Becomes @task.gke_cmd once registered in get_provider_info().
    """
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GKECmdDecoratedOperator,
        args_only=args_only,
        multiple_outputs=multiple_outputs,
        defaults_conn_id=defaults_conn_id,
        **kwargs,
    )

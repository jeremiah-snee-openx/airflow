Decorators to eval


```python
from __future__ import annotations

"""
GKE TaskFlow decorators mirroring apache-airflow-providers-cncf-kubernetes TaskFlow patterns,
implemented using apache-airflow-providers-google GKE operators.

Order provided (as requested):
  - gke_pod (with opinionated defaults)
  - gke_pod_cmd_alpha (without opinionated defaults)
  - gke_pod_cmd (with opinionated defaults)

  - gke_job (with opinionated defaults)
  - gke_job_cmd_alpha (without opinionated defaults)
  - gke_job_cmd (with opinionated defaults)
"""

import base64
import os
import pickle
import warnings
from collections.abc import Callable, Mapping, Sequence
from shlex import quote
from tempfile import TemporaryDirectory
from typing import Any, TYPE_CHECKING

import dill
from kubernetes.client import models as k8s

from airflow.configuration import conf
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.providers.google.cloud.operators.kubernetes_engine import (
    GKEStartJobOperator,
    GKEStartPodOperator,
)
from airflow.providers.google.common.hooks.base_google import PROVIDE_PROJECT_ID

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


# ---------------------------------------------------------------------------
# Shared implementation bits (mirrors upstream cncf.kubernetes @task.kubernetes)
# ---------------------------------------------------------------------------

# These env var names match upstream cncf.kubernetes @task.kubernetes decorator.
_PYTHON_SCRIPT_ENV = "__PYTHON_SCRIPT"
_PYTHON_INPUT_ENV = "__PYTHON_INPUT"


def _generate_decoded_command(env_var: str, file: str) -> str:
    """
    Build a shell-safe python one-liner that:
      - base64-decodes bytes from $env_var
      - writes them to file

    This matches the upstream decorator implementation.
    """
    return (
        f'python -c "import base64, os;'
        rf"x = base64.b64decode(os.environ[\"{env_var}\"]);"
        rf'f = open(\"{file}\", \"wb\"); f.write(x); f.close()"'
    )


def _read_file_contents(filename: str) -> str:
    """Read a local file and return base64(text) of its bytes."""
    with open(filename, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Opinionated defaults resolution (ONLY used by the "*with opinionated defaults" variants)
# ---------------------------------------------------------------------------

def _get_var(key: str) -> str | None:
    """
    Safe Variable.get wrapper:
      - returns stripped string
      - returns None if unset/unreadable/empty
    """
    try:
        v = Variable.get(key)
        return v.strip() if isinstance(v, str) and v.strip() else None
    except Exception:
        return None


def _get_conn_extra(conn_id: str, key: str) -> str | None:
    """
    Read defaults from Connection.extra JSON (NOT credentials), e.g.:
      {"project_id":"...", "location":"us-central1", "cluster_name":"...", "namespace":"default"}
    """
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
    # Where to read cluster defaults (NOT credentials). Can be set via Variable too.
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

    # These are required by the GKE operators (location/cluster_name), and project_id is required
    # unless you’re relying on provider-specific project resolution.
    missing = [k for k in ("project_id", "location", "cluster_name") if not resolved[k]]
    if missing:
        raise AirflowException(
            "Missing required GKE defaults: "
            + ", ".join(missing)
            + ". Provide them to the decorator call, or set Variables "
            + "(MYCO_GKE_PROJECT_ID / MYCO_GKE_LOCATION / MYCO_GKE_CLUSTER_NAME) "
            + "or put them in Connection Extra JSON via MYCO_GKE_DEFAULTS_CONN_ID."
        )

    return resolved


# ===========================================================================
# 1) gke_pod (with opinionated defaults)
# ===========================================================================

class _GKEPodDecoratedOperator(DecoratedOperator, GKEStartPodOperator):
    """
    Like upstream @task.kubernetes, but runs in a GKE cluster via GKEStartPodOperator.

    The python_callable's source + serialized inputs are embedded into env vars and
    executed inside the pod.
    """

    # Shows up in UI / logs similar to upstream custom_operator_name.
    custom_operator_name = "@task.gke_pod"

    # cmds/arguments are "owned" by the decorator, so we remove them from templating,
    # mirroring upstream @task.kubernetes. (cmds/arguments are present in pod template_fields) :contentReference[oaicite:3]{index=3}
    template_fields: Sequence[str] = tuple(
        {"op_args", "op_kwargs", *GKEStartPodOperator.template_fields} - {"cmds", "arguments"}
    )

    # Since we won't mutate the callable, shallow copy avoids deepcopy issues (e.g. protobuf)
    shallow_copy_attrs: Sequence[str] = ("python_callable",)

    def __init__(
        self,
        *,
        python_callable: Callable[..., Any],
        # Upstream @task.kubernetes option
        use_dill: bool = False,
        # Opinionated defaults config
        defaults_conn_id: str | None = None,
        # Optional GKE fields (resolved via resolve_gke_defaults)
        project_id: str | None = None,
        location: str | None = None,
        cluster_name: str | None = None,
        namespace: str | None = None,
        gcp_conn_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Label: serialization strategy used for op_args/op_kwargs payload
        self.use_dill = use_dill

        # Label: resolve cluster addressing + namespace
        resolved = resolve_gke_defaults(
            project_id=project_id,
            location=location,
            cluster_name=cluster_name,
            namespace=namespace,
            gcp_conn_id=gcp_conn_id,
            defaults_conn_id=defaults_conn_id,
        )

        # Label: name/random suffix behavior mirrors upstream @task.kubernetes :contentReference[oaicite:4]{index=4}
        op_name = kwargs.pop("name", f"gke-airflow-pod-{python_callable.__name__}")
        random_name_suffix = kwargs.pop("random_name_suffix", True)

        # Label: call the real operator __init__ with resolved GKE fields
        super().__init__(
            python_callable=python_callable,
            # Required GKE fields
            project_id=resolved["project_id"],
            location=resolved["location"],
            cluster_name=resolved["cluster_name"],
            gcp_conn_id=resolved["gcp_conn_id"],
            # Standard KPO field (passed via **kwargs because GKEStartPodOperator uses *args/**kwargs)
            namespace=resolved["namespace"],
            # Standard KPO identity fields
            name=op_name,
            random_name_suffix=random_name_suffix,
            # Decorator owns cmds/arguments
            cmds=["placeholder-command"],
            # Everything else passes through to GKEStartPodOperator/KubernetesPodOperator.
            **kwargs,
        )

    def _generate_cmds(self) -> list[str]:
        """
        Build the container command that:
          1) writes script.py from $__PYTHON_SCRIPT
          2) writes script.in from $__PYTHON_INPUT
          3) (optional) creates /airflow/xcom for XCom sidecar collection
          4) executes python script.py script.in {output_file}

        Path and control flow match upstream @task.kubernetes. :contentReference[oaicite:5]{index=5}
        """
        script_filename = "/tmp/script.py"
        input_filename = "/tmp/script.in"

        if getattr(self, "do_xcom_push", False):
            output_filename = "/airflow/xcom/return.json"
            make_xcom_dir_cmd = "mkdir -p /airflow/xcom"
        else:
            output_filename = "/dev/null"
            make_xcom_dir_cmd = ":"  # shell no-op

        write_local_script_cmd = _generate_decoded_command(quote(_PYTHON_SCRIPT_ENV), quote(script_filename))
        write_local_input_cmd = _generate_decoded_command(quote(_PYTHON_INPUT_ENV), quote(input_filename))
        exec_python_cmd = f"python {script_filename} {input_filename} {output_filename}"

        return [
            "bash",
            "-cx",
            f"{write_local_script_cmd} && {write_local_input_cmd} && {make_xcom_dir_cmd} && {exec_python_cmd}",
        ]

    def execute(self, context: Context):
        # Label: matches upstream tempdir workflow :contentReference[oaicite:6]{index=6}
        with TemporaryDirectory(prefix="venv") as tmp_dir:
            pickling_lib = dill if self.use_dill else pickle

            local_script = os.path.join(tmp_dir, "script.py")
            local_input = os.path.join(tmp_dir, "script.in")

            # Label: serialize op_args/op_kwargs payload for in-container execution
            with open(local_input, "wb") as f:
                pickling_lib.dump({"args": self.op_args, "kwargs": self.op_kwargs}, f)

            # Label: inject python source into the in-container script template
            py_source = self.get_python_source()
            jinja_context = {
                "op_args": self.op_args,
                "op_kwargs": self.op_kwargs,
                "pickling_library": pickling_lib.__name__,
                "python_callable": self.python_callable.__name__,
                "python_callable_source": py_source,
                "string_args_global": False,
            }

            # Label: reuse cncf.kubernetes script writer (the upstream TaskFlow mechanism)
            from airflow.providers.cncf.kubernetes.python_kubernetes_script import write_python_script
            write_python_script(jinja_context=jinja_context, filename=local_script)

            # Label: append env vars that carry script + payload (base64-encoded)
            self.env_vars: list[k8s.V1EnvVar] = [
                *self.env_vars,
                k8s.V1EnvVar(name=_PYTHON_SCRIPT_ENV, value=_read_file_contents(local_script)),
                k8s.V1EnvVar(name=_PYTHON_INPUT_ENV, value=_read_file_contents(local_input)),
            ]

            # Label: set the actual container command and run
            self.cmds = self._generate_cmds()
            return super().execute(context)


def gke_pod_task(
    python_callable: Callable[..., Any] | None = None,
    *,
    multiple_outputs: bool | None = None,
    **kwargs: Any,
) -> TaskDecorator:
    """
    Register as "gke_pod" in get_provider_info().

    multiple_outputs is handled by task_decorator_factory (same pattern as upstream @task.kubernetes). :contentReference[oaicite:7]{index=7}
    """
    return task_decorator_factory(
        python_callable=python_callable,
        multiple_outputs=multiple_outputs,
        decorated_operator_class=_GKEPodDecoratedOperator,
        **kwargs,
    )


# ===========================================================================
# 2) gke_pod_cmd_alpha (without opinionated defaults)
# ===========================================================================

class _GKEPodCmdAlphaDecoratedOperator(DecoratedOperator, GKEStartPodOperator):
    """
    Like upstream @task.kubernetes_cmd, but on GKEStartPodOperator.

    ALPHA: does NOT do any default resolution; you must pass location/cluster_name/etc.
    """

    custom_operator_name = "@task.gke_pod_cmd_alpha"

    # Label: keep cmds/arguments templatable because this decorator sets them dynamically
    template_fields: Sequence[str] = tuple({"op_args", "op_kwargs", *GKEStartPodOperator.template_fields})

    # Label: matches upstream kubernetes_cmd behavior
    overwrite_rtif_after_execution: bool = True

    def __init__(
        self,
        *,
        python_callable: Callable[..., list[str]],
        args_only: bool = False,
        # Explicit GKE fields (no default-resolution)
        location: str,
        cluster_name: str,
        project_id: str = PROVIDE_PROJECT_ID,
        gcp_conn_id: str = "google_cloud_default",
        namespace: str | None = None,
        use_internal_ip: bool = False,
        use_dns_endpoint: bool = False,
        impersonation_chain: str | Sequence[str] | None = None,
        on_finish_action: str | None = None,
        # Default matches operators’ default_deferrable pattern in the Google operator module.
        deferrable: bool = conf.getboolean("operators", "default_deferrable", fallback=False),
        # Custom: TaskFlow-ish ergonomics for dict return (not part of upstream kubernetes_cmd)
        multiple_outputs: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.args_only = args_only
        self._multiple_outputs = bool(multiple_outputs)

        # Label: cmds/arguments are "owned" by the decorator (mirrors upstream kubernetes_cmd) 
        cmds = kwargs.pop("cmds", None)
        arguments = kwargs.pop("arguments", None)
        if cmds is not None or arguments is not None:
            warnings.warn(
                f"`cmds`/`arguments` are ignored by {self.custom_operator_name}. "
                "Return list[str] from the decorated function instead.",
                UserWarning,
                stacklevel=3,
            )

        name = kwargs.pop("name", f"gke-cmd-alpha-{python_callable.__name__}")
        random_name_suffix = kwargs.pop("random_name_suffix", True)

        super().__init__(
            python_callable=python_callable,
            # Required GKE fields
            location=location,
            cluster_name=cluster_name,
            project_id=project_id,
            gcp_conn_id=gcp_conn_id,
            # Optional GKE fields
            use_internal_ip=use_internal_ip,
            use_dns_endpoint=use_dns_endpoint,
            impersonation_chain=impersonation_chain,
            on_finish_action=on_finish_action,
            deferrable=deferrable,
            # Standard KPO fields
            namespace=namespace,
            name=name,
            random_name_suffix=random_name_suffix,
            # Decorator owns cmds/arguments (set at execute time)
            cmds=None,
            arguments=None,
            **kwargs,
        )

    def execute(self, context: Context):
        # Label: render first so args/macros can influence command generation
        self.render_template_fields(context)

        generated = self._generate_cmds(context)
        if self.args_only:
            self.cmds = []
            self.arguments = generated
        else:
            self.cmds = generated
            self.arguments = []

        # Label: render again so generated command strings can be templated too
        self.render_template_fields(context)

        result = super().execute(context)

        # Label: optional dict “unroll” (custom ergonomics)
        if self._multiple_outputs and isinstance(result, Mapping):
            ti = context["ti"]
            for k, v in result.items():
                ti.xcom_push(key=str(k), value=v)

        return result

    def _generate_cmds(self, context: Context) -> list[str]:
        # Label: upstream kubernetes_cmd pattern for context + kwargs binding 
        context_merge(context, self.op_kwargs)
        fn_kwargs = determine_kwargs(self.python_callable, self.op_args, context)
        out = self.python_callable(*self.op_args, **fn_kwargs)

        # Label: strict return contract (list[str])
        if not isinstance(out, list):
            raise TypeError(f"Expected {self.python_callable.__name__} to return list[str], got {type(out)}")
        if not out:
            raise ValueError(f"{self.python_callable.__name__} returned an empty command list")
        if not all(isinstance(x, str) for x in out):
            raise TypeError(f"Expected {self.python_callable.__name__} to return list[str], got {out}")

        return out


def gke_pod_cmd_alpha_task(
    python_callable: Callable[..., list[str]] | None = None,
    **kwargs: Any,
) -> TaskDecorator:
    """Register as "gke_pod_cmd_alpha" in get_provider_info()."""
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GKEPodCmdAlphaDecoratedOperator,
        **kwargs,
    )


# ===========================================================================
# 3) gke_pod_cmd (with opinionated defaults)
# ===========================================================================

class _GKEPodCmdDecoratedOperator(DecoratedOperator, GKEStartPodOperator):
    """
    @task.kubernetes_cmd-style for GKEStartPodOperator, WITH opinionated defaults.

    python_callable runs in Airflow to generate list[str], then the pod runs those.
    """

    custom_operator_name = "@task.gke_pod_cmd"
    template_fields: Sequence[str] = tuple({"op_args", "op_kwargs", *GKEStartPodOperator.template_fields})
    overwrite_rtif_after_execution: bool = True

    def __init__(
        self,
        *,
        python_callable: Callable[..., list[str]],
        args_only: bool = False,
        multiple_outputs: bool | None = None,
        defaults_conn_id: str | None = None,
        # GKE fields (may be None; resolved)
        project_id: str | None = None,
        location: str | None = None,
        cluster_name: str | None = None,
        namespace: str | None = None,
        gcp_conn_id: str | None = None,
        # Optional GKE fields (passed through)
        use_internal_ip: bool = False,
        use_dns_endpoint: bool = False,
        impersonation_chain: str | Sequence[str] | None = None,
        on_finish_action: str | None = None,
        deferrable: bool = conf.getboolean("operators", "default_deferrable", fallback=False),
        **kwargs: Any,
    ) -> None:
        self.args_only = args_only
        self._multiple_outputs = bool(multiple_outputs)

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

        name = kwargs.pop("name", f"gke-cmd-{python_callable.__name__}")
        random_name_suffix = kwargs.pop("random_name_suffix", True)

        super().__init__(
            python_callable=python_callable,
            location=resolved["location"],
            cluster_name=resolved["cluster_name"],
            project_id=resolved["project_id"],
            gcp_conn_id=resolved["gcp_conn_id"],
            namespace=resolved["namespace"],
            use_internal_ip=use_internal_ip,
            use_dns_endpoint=use_dns_endpoint,
            impersonation_chain=impersonation_chain,
            on_finish_action=on_finish_action,
            deferrable=deferrable,
            name=name,
            random_name_suffix=random_name_suffix,
            cmds=None,
            arguments=None,
            **kwargs,
        )

    def execute(self, context: Context):
        self.render_template_fields(context)

        generated = self._generate_cmds(context)
        if self.args_only:
            self.cmds = []
            self.arguments = generated
        else:
            self.cmds = generated
            self.arguments = []

        self.render_template_fields(context)

        result = super().execute(context)

        if self._multiple_outputs and isinstance(result, Mapping):
            ti = context["ti"]
            for k, v in result.items():
                ti.xcom_push(key=str(k), value=v)

        return result

    def _generate_cmds(self, context: Context) -> list[str]:
        context_merge(context, self.op_kwargs)
        fn_kwargs = determine_kwargs(self.python_callable, self.op_args, context)
        out = self.python_callable(*self.op_args, **fn_kwargs)

        if not isinstance(out, list):
            raise TypeError(f"Expected {self.python_callable.__name__} to return list[str], got {type(out)}")
        if not out:
            raise ValueError(f"{self.python_callable.__name__} returned an empty command list")
        if not all(isinstance(x, str) for x in out):
            raise TypeError(f"Expected {self.python_callable.__name__} to return list[str], got {out}")

        return out


def gke_pod_cmd_task(
    python_callable: Callable[..., list[str]] | None = None,
    **kwargs: Any,
) -> TaskDecorator:
    """Register as "gke_pod_cmd" in get_provider_info()."""
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GKEPodCmdDecoratedOperator,
        **kwargs,
    )


# ===========================================================================
# 4) gke_job (with opinionated defaults)
# ===========================================================================

class _GKEJobDecoratedOperator(DecoratedOperator, GKEStartJobOperator):
    """
    Like @task.kubernetes, but uses GKEStartJobOperator.

    The underlying Job operator still uses KubernetesPodOperator semantics for
    container execution, including cmds/arguments and XCom sidecar collection.
    """

    custom_operator_name = "@task.gke_job"
    template_fields: Sequence[str] = tuple(
        {"op_args", "op_kwargs", *GKEStartJobOperator.template_fields} - {"cmds", "arguments"}
    )
    shallow_copy_attrs: Sequence[str] = ("python_callable",)

    def __init__(
        self,
        *,
        python_callable: Callable[..., Any],
        use_dill: bool = False,
        defaults_conn_id: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        cluster_name: str | None = None,
        namespace: str | None = None,
        gcp_conn_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.use_dill = use_dill

        resolved = resolve_gke_defaults(
            project_id=project_id,
            location=location,
            cluster_name=cluster_name,
            namespace=namespace,
            gcp_conn_id=gcp_conn_id,
            defaults_conn_id=defaults_conn_id,
        )

        name = kwargs.pop("name", f"gke-airflow-job-{python_callable.__name__}")
        random_name_suffix = kwargs.pop("random_name_suffix", True)

        super().__init__(
            python_callable=python_callable,
            project_id=resolved["project_id"],
            location=resolved["location"],
            cluster_name=resolved["cluster_name"],
            gcp_conn_id=resolved["gcp_conn_id"],
            namespace=resolved["namespace"],
            name=name,
            random_name_suffix=random_name_suffix,
            cmds=["placeholder-command"],
            **kwargs,
        )

    def _generate_cmds(self) -> list[str]:
        script_filename = "/tmp/script.py"
        input_filename = "/tmp/script.in"

        if getattr(self, "do_xcom_push", False):
            output_filename = "/airflow/xcom/return.json"
            make_xcom_dir_cmd = "mkdir -p /airflow/xcom"
        else:
            output_filename = "/dev/null"
            make_xcom_dir_cmd = ":"

        write_local_script_cmd = _generate_decoded_command(quote(_PYTHON_SCRIPT_ENV), quote(script_filename))
        write_local_input_cmd = _generate_decoded_command(quote(_PYTHON_INPUT_ENV), quote(input_filename))
        exec_python_cmd = f"python {script_filename} {input_filename} {output_filename}"

        return [
            "bash",
            "-cx",
            f"{write_local_script_cmd} && {write_local_input_cmd} && {make_xcom_dir_cmd} && {exec_python_cmd}",
        ]

    def execute(self, context: Context):
        with TemporaryDirectory(prefix="venv") as tmp_dir:
            pickling_lib = dill if self.use_dill else pickle
            local_script = os.path.join(tmp_dir, "script.py")
            local_input = os.path.join(tmp_dir, "script.in")

            with open(local_input, "wb") as f:
                pickling_lib.dump({"args": self.op_args, "kwargs": self.op_kwargs}, f)

            py_source = self.get_python_source()
            jinja_context = {
                "op_args": self.op_args,
                "op_kwargs": self.op_kwargs,
                "pickling_library": pickling_lib.__name__,
                "python_callable": self.python_callable.__name__,
                "python_callable_source": py_source,
                "string_args_global": False,
            }

            from airflow.providers.cncf.kubernetes.python_kubernetes_script import write_python_script
            write_python_script(jinja_context=jinja_context, filename=local_script)

            self.env_vars: list[k8s.V1EnvVar] = [
                *self.env_vars,
                k8s.V1EnvVar(name=_PYTHON_SCRIPT_ENV, value=_read_file_contents(local_script)),
                k8s.V1EnvVar(name=_PYTHON_INPUT_ENV, value=_read_file_contents(local_input)),
            ]

            self.cmds = self._generate_cmds()
            return super().execute(context)


def gke_job_task(
    python_callable: Callable[..., Any] | None = None,
    *,
    multiple_outputs: bool | None = None,
    **kwargs: Any,
) -> TaskDecorator:
    """
    Register as "gke_job" in get_provider_info().

    multiple_outputs is handled by task_decorator_factory (same upstream pattern as @task.kubernetes). :contentReference[oaicite:10]{index=10}
    """
    return task_decorator_factory(
        python_callable=python_callable,
        multiple_outputs=multiple_outputs,
        decorated_operator_class=_GKEJobDecoratedOperator,
        **kwargs,
    )


# ===========================================================================
# 5) gke_job_cmd_alpha (without opinionated defaults)
# ===========================================================================

class _GKEJobCmdAlphaDecoratedOperator(DecoratedOperator, GKEStartJobOperator):
    """@task.kubernetes_cmd-style for GKEStartJobOperator, without opinionated defaults."""

    custom_operator_name = "@task.gke_job_cmd_alpha"
    template_fields: Sequence[str] = tuple({"op_args", "op_kwargs", *GKEStartJobOperator.template_fields})
    overwrite_rtif_after_execution: bool = True

    def __init__(
        self,
        *,
        python_callable: Callable[..., list[str]],
        args_only: bool = False,
        location: str,
        cluster_name: str,
        project_id: str = PROVIDE_PROJECT_ID,
        gcp_conn_id: str = "google_cloud_default",
        namespace: str | None = None,
        use_internal_ip: bool = False,
        use_dns_endpoint: bool = False,
        impersonation_chain: str | Sequence[str] | None = None,
        deferrable: bool = conf.getboolean("operators", "default_deferrable", fallback=False),
        job_poll_interval: float = 10.0,
        multiple_outputs: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.args_only = args_only
        self._multiple_outputs = bool(multiple_outputs)

        cmds = kwargs.pop("cmds", None)
        arguments = kwargs.pop("arguments", None)
        if cmds is not None or arguments is not None:
            warnings.warn(
                f"`cmds`/`arguments` are ignored by {self.custom_operator_name}. "
                "Return list[str] from the decorated function instead.",
                UserWarning,
                stacklevel=3,
            )

        name = kwargs.pop("name", f"gke-job-cmd-alpha-{python_callable.__name__}")
        random_name_suffix = kwargs.pop("random_name_suffix", True)

        super().__init__(
            python_callable=python_callable,
            location=location,
            cluster_name=cluster_name,
            project_id=project_id,
            gcp_conn_id=gcp_conn_id,
            namespace=namespace,
            use_internal_ip=use_internal_ip,
            use_dns_endpoint=use_dns_endpoint,
            impersonation_chain=impersonation_chain,
            deferrable=deferrable,
            job_poll_interval=job_poll_interval,
            name=name,
            random_name_suffix=random_name_suffix,
            cmds=None,
            arguments=None,
            **kwargs,
        )

    def execute(self, context: Context):
        self.render_template_fields(context)

        generated = self._generate_cmds(context)
        if self.args_only:
            self.cmds = []
            self.arguments = generated
        else:
            self.cmds = generated
            self.arguments = []

        self.render_template_fields(context)

        result = super().execute(context)

        if self._multiple_outputs and isinstance(result, Mapping):
            ti = context["ti"]
            for k, v in result.items():
                ti.xcom_push(key=str(k), value=v)

        return result

    def _generate_cmds(self, context: Context) -> list[str]:
        context_merge(context, self.op_kwargs)
        fn_kwargs = determine_kwargs(self.python_callable, self.op_args, context)
        out = self.python_callable(*self.op_args, **fn_kwargs)

        if not isinstance(out, list):
            raise TypeError(f"Expected {self.python_callable.__name__} to return list[str], got {type(out)}")
        if not out:
            raise ValueError(f"{self.python_callable.__name__} returned an empty command list")
        if not all(isinstance(x, str) for x in out):
            raise TypeError(f"Expected {self.python_callable.__name__} to return list[str], got {out}")

        return out


def gke_job_cmd_alpha_task(
    python_callable: Callable[..., list[str]] | None = None,
    **kwargs: Any,
) -> TaskDecorator:
    """Register as "gke_job_cmd_alpha" in get_provider_info()."""
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GKEJobCmdAlphaDecoratedOperator,
        **kwargs,
    )


# ===========================================================================
# 6) gke_job_cmd (with opinionated defaults)
# ===========================================================================

class _GKEJobCmdDecoratedOperator(DecoratedOperator, GKEStartJobOperator):
    """@task.kubernetes_cmd-style for GKEStartJobOperator, WITH opinionated defaults."""

    custom_operator_name = "@task.gke_job_cmd"
    template_fields: Sequence[str] = tuple({"op_args", "op_kwargs", *GKEStartJobOperator.template_fields})
    overwrite_rtif_after_execution: bool = True

    def __init__(
        self,
        *,
        python_callable: Callable[..., list[str]],
        args_only: bool = False,
        multiple_outputs: bool | None = None,
        defaults_conn_id: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        cluster_name: str | None = None,
        namespace: str | None = None,
        gcp_conn_id: str | None = None,
        use_internal_ip: bool = False,
        use_dns_endpoint: bool = False,
        impersonation_chain: str | Sequence[str] | None = None,
        deferrable: bool = conf.getboolean("operators", "default_deferrable", fallback=False),
        job_poll_interval: float = 10.0,
        **kwargs: Any,
    ) -> None:
        self.args_only = args_only
        self._multiple_outputs = bool(multiple_outputs)

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

        name = kwargs.pop("name", f"gke-job-cmd-{python_callable.__name__}")
        random_name_suffix = kwargs.pop("random_name_suffix", True)

        super().__init__(
            python_callable=python_callable,
            location=resolved["location"],
            cluster_name=resolved["cluster_name"],
            project_id=resolved["project_id"],
            gcp_conn_id=resolved["gcp_conn_id"],
            namespace=resolved["namespace"],
            use_internal_ip=use_internal_ip,
            use_dns_endpoint=use_dns_endpoint,
            impersonation_chain=impersonation_chain,
            deferrable=deferrable,
            job_poll_interval=job_poll_interval,
            name=name,
            random_name_suffix=random_name_suffix,
            cmds=None,
            arguments=None,
            **kwargs,
        )

    def execute(self, context: Context):
        self.render_template_fields(context)

        generated = self._generate_cmds(context)
        if self.args_only:
            self.cmds = []
            self.arguments = generated
        else:
            self.cmds = generated
            self.arguments = []

        self.render_template_fields(context)

        result = super().execute(context)

        if self._multiple_outputs and isinstance(result, Mapping):
            ti = context["ti"]
            for k, v in result.items():
                ti.xcom_push(key=str(k), value=v)

        return result

    def _generate_cmds(self, context: Context) -> list[str]:
        context_merge(context, self.op_kwargs)
        fn_kwargs = determine_kwargs(self.python_callable, self.op_args, context)
        out = self.python_callable(*self.op_args, **fn_kwargs)

        if not isinstance(out, list):
            raise TypeError(f"Expected {self.python_callable.__name__} to return list[str], got {type(out)}")
        if not out:
            raise ValueError(f"{self.python_callable.__name__} returned an empty command list")
        if not all(isinstance(x, str) for x in out):
            raise TypeError(f"Expected {self.python_callable.__name__} to return list[str], got {out}")

        return out


def gke_job_cmd_task(
    python_callable: Callable[..., list[str]] | None = None,
    **kwargs: Any,
) -> TaskDecorator:
    """Register as "gke_job_cmd" in get_provider_info()."""
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GKEJobCmdDecoratedOperator,
        **kwargs,
    )
```

`do_xcom_push=True` for `KubernetesPodOperator`-derived operators (including `GKEStartPodOperator`) **does not capture stdout**. It reads the **contents of `/airflow/xcom/return.json`** from inside the container and pushes that as the XCom return value. ([Apache Airflow][1])

So to make your example work *without* manually adding `> /airflow/xcom/return.json` in every returned command, `gke_pod_cmd` needs an explicit “**redirect stdout → return.json**” mode.

Below is a **source-compatible** update to your `gke_pod_cmd` operator class that adds a parameter to do exactly that, without guessing.

---

## Patch: update `gke_pod_cmd` to support `stdout -> /airflow/xcom/return.json`

### What this adds

* **New parameter:** `xcom_push_mode`

  * `"file"` (default): your command is responsible for writing `/airflow/xcom/return.json`
  * `"stdout"`: the decorator wraps your generated command in `sh -ec 'mkdir -p /airflow/xcom; <cmd> > /airflow/xcom/return.json'`

### Why this is correct

* It matches the operator contract: XCom comes from `/airflow/xcom/return.json`. ([Apache Airflow][1])
* It preserves the upstream `kubernetes_cmd` generation pattern (generate list[str] then assign into `cmds/arguments`).
* It avoids heuristics like “detect whether the user already writes return.json” (you told me not to guess).

### Updated code (only the parts that change)

```python
from shlex import quote
from airflow.exceptions import AirflowException

class _GKEPodCmdDecoratedOperator(DecoratedOperator, GKEStartPodOperator):
    custom_operator_name = "@task.gke_pod_cmd"
    template_fields: Sequence[str] = tuple({"op_args", "op_kwargs", *GKEStartPodOperator.template_fields})
    overwrite_rtif_after_execution: bool = True

    def __init__(
        self,
        *,
        python_callable: Callable[..., list[str]],
        args_only: bool = False,
        multiple_outputs: bool | None = None,
        defaults_conn_id: str | None = None,
        # NEW: how XCom file is produced when do_xcom_push=True
        xcom_push_mode: str = "file",  # allowed: "file" | "stdout"
        # ... your existing params ...
        **kwargs: Any,
    ) -> None:
        self.args_only = args_only
        self._multiple_outputs = bool(multiple_outputs)

        # NEW: validate mode early
        if xcom_push_mode not in ("file", "stdout"):
            raise AirflowException("xcom_push_mode must be either 'file' or 'stdout'")
        self.xcom_push_mode = xcom_push_mode

        # ... your existing __init__ logic (defaults resolution, warnings, super().__init__) ...

    def execute(self, context: Context):
        self.render_template_fields(context)

        generated = self._generate_cmds(context)

        # NEW: stdout redirection mode
        if getattr(self, "do_xcom_push", False) and self.xcom_push_mode == "stdout":
            # args_only means “keep image entrypoint”; stdout redirection requires a shell wrapper,
            # so we refuse the combination explicitly (no silent behavior change).
            if self.args_only:
                raise AirflowException(
                    "args_only=True is incompatible with xcom_push_mode='stdout' because "
                    "stdout capture requires overriding the container command with a shell wrapper."
                )

            # Render list[str] into a safe shell command string
            cmd_str = " ".join(quote(s) for s in generated)

            # Ensure the directory exists, then redirect stdout to return.json
            script = f"mkdir -p /airflow/xcom; {cmd_str} > /airflow/xcom/return.json"

            # Use POSIX sh (more widely available than bash)
            self.cmds = ["sh", "-ec"]
            self.arguments = [script]

        else:
            # Original kubernetes_cmd-style behavior
            if self.args_only:
                self.cmds = []
                self.arguments = generated
            else:
                self.cmds = generated
                self.arguments = []

        self.render_template_fields(context)

        result = super().execute(context)

        if self._multiple_outputs and isinstance(result, Mapping):
            ti = context["ti"]
            for k, v in result.items():
                ti.xcom_push(key=str(k), value=v)

        return result
```

---

## How to call it (your exact example, now correct)

Because the operator only pushes what’s in `/airflow/xcom/return.json` ([Apache Airflow][1]), your command must emit **valid JSON to stdout** when `xcom_push_mode="stdout"`.

```python
@task.gke_pod_cmd(
    image="python:3.11-slim",
    do_xcom_push=True,
    xcom_push_mode="stdout",
)
def run_cmd():
    return [
        "python",
        "-m",
        "some_script",
    ]

out = run_cmd()  # XComArg containing parsed JSON from some_script's stdout
```

### Contract you must satisfy (no guessing)

* `some_script` must write **only** JSON (or at least ensure stdout ends up as valid JSON) to stdout, because that stdout is redirected verbatim into `return.json`.
* If it prints logs to stdout, your XCom parse will fail (because `return.json` won’t be valid JSON). The operator contract is file content → XCom. ([Apache Airflow][1])

If you want “logs + JSON,” print logs to **stderr** and keep stdout JSON-only (or have the script write the file itself and keep `xcom_push_mode="file"`).

---


Here’s the **provider registration snippet** you drop into `get_provider_info()` under `"task-decorators"` (with **placeholders** for the import paths you actually use).

This is the documented provider mechanism Airflow uses to expose decorators as `@task.<name>`.

```python
def get_provider_info():
    return {
        "package-name": "<YOUR_PROVIDER_DIST_NAME>",
        "name": "<YOUR_PROVIDER_DISPLAY_NAME>",
        "description": "<YOUR_PROVIDER_DESCRIPTION>",
        "task-decorators": [
            {
                "name": "gke_pod",
                "class-name": "<YOUR_PYTHON_PACKAGE>.decorators.gke.gke_pod_task",
            },
            {
                "name": "gke_pod_cmd_alpha",
                "class-name": "<YOUR_PYTHON_PACKAGE>.decorators.gke.gke_pod_cmd_alpha_task",
            },
            {
                "name": "gke_pod_cmd",
                "class-name": "<YOUR_PYTHON_PACKAGE>.decorators.gke.gke_pod_cmd_task",
            },
            {
                "name": "gke_job",
                "class-name": "<YOUR_PYTHON_PACKAGE>.decorators.gke.gke_job_task",
            },
            {
                "name": "gke_job_cmd_alpha",
                "class-name": "<YOUR_PYTHON_PACKAGE>.decorators.gke.gke_job_cmd_alpha_task",
            },
            {
                "name": "gke_job_cmd",
                "class-name": "<YOUR_PYTHON_PACKAGE>.decorators.gke.gke_job_cmd_task",
            },
        ],
    }
```

And the corresponding **entry point** (so Airflow can discover `get_provider_info()`):

```toml
[project.entry-points."apache_airflow_provider"]
your_provider_key = "<YOUR_PYTHON_PACKAGE>.provider_info:get_provider_info"
```

Replace:

* `<YOUR_PYTHON_PACKAGE>` with your actual import root (e.g. `openx_airflow_provider`)
* the `.decorators.gke...` module path with wherever you put those functions.





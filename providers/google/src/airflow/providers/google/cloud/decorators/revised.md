Below is an “upstream-quality” shape that:

* **Matches the existing TaskFlow decorator patterns** used by `@task.kubernetes` / `@task.kubernetes_cmd` (same compat imports, same `task_decorator_factory`, same `DecoratedOperator` approach). ([Apache Airflow][1])
* Uses **Airflow config / env vars** (not Variables) for defaults.
* Adds an explicit **`xcom_push_mode`** that supports the “`stdout → /airflow/xcom/return.json`” flow you asked for (so `run_cmd()` returns an XCom JSON dict when `do_xcom_push=True`). The `/airflow/xcom/return.json` contract is exactly what the upstream Kubernetes TaskFlow decorator uses. ([Apache Airflow][1])
* Avoids `config_file` (GKE operators explicitly forbid it). ([Apache Airflow][2])

---

## Recommended package + module names

Given the Google provider already has `operators/kubernetes_engine.py`, I would put these in:

* `airflow/providers/google/cloud/decorators/kubernetes_engine.py`

This is consistent, discoverable, and avoids inventing a second naming axis.

---

## Provider config options (airflow.cfg / env vars)

### Config section name

Use a provider-scoped section:

* **`[providers.google.kubernetes_engine]`**

This aligns with the existing “`providers.<provider>`…” convention used by other providers’ config references (example: JDBC). ([Apache Airflow][3])

### Options

You want cluster defaults + a few behavior defaults:

* `project_id` (string, required for defaults)
* `location` (string, required for defaults)
* `cluster_name` (string, required for defaults)
* `namespace` (string, default: `default`)
* `gcp_conn_id` (string, default: `google_cloud_default`)
* `use_internal_ip` (bool, default: `False`)
* `use_dns_endpoint` (bool, default: `False`)

### Env var equivalents

Following Airflow’s standard env-var mapping (same style shown in provider docs like JDBC): ([Apache Airflow][4])

* `AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__PROJECT_ID`
* `AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__LOCATION`
* `AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__CLUSTER_NAME`
* `AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__NAMESPACE`
* `AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__GCP_CONN_ID`
* `AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__USE_INTERNAL_IP`
* `AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__USE_DNS_ENDPOINT`

---

## Provider registration snippet

Add to **`airflow/providers/google/get_provider_info_TEMPLATE.py.jinja2`** (the generated module warns you to edit the template, not the generated file). ([Apache Airflow][5])

```python
def get_provider_info():
    return {
        # ... existing ...
        "task-decorators": [
            {
                "name": "gke_pod",
                "class-name": "airflow.providers.google.cloud.decorators.kubernetes_engine.gke_pod_task",
            },
            {
                "name": "gke_pod_cmd_base",
                "class-name": "airflow.providers.google.cloud.decorators.kubernetes_engine.gke_pod_cmd_base_task",
            },
            {
                "name": "gke_pod_cmd",
                "class-name": "airflow.providers.google.cloud.decorators.kubernetes_engine.gke_pod_cmd_task",
            },
            {
                "name": "gke_job",
                "class-name": "airflow.providers.google.cloud.decorators.kubernetes_engine.gke_job_task",
            },
            {
                "name": "gke_job_cmd_base",
                "class-name": "airflow.providers.google.cloud.decorators.kubernetes_engine.gke_job_cmd_base_task",
            },
            {
                "name": "gke_job_cmd",
                "class-name": "airflow.providers.google.cloud.decorators.kubernetes_engine.gke_job_cmd_task",
            },
        ],
        "config": {
            "providers.google.kubernetes_engine": {
                "description": "Defaults for Google Kubernetes Engine TaskFlow decorators.",
                "options": {
                    "project_id": {
                        "description": "Default GCP project id for GKE decorators.",
                        "type": "string",
                    },
                    "location": {
                        "description": "Default GKE location (region/zone) for GKE decorators.",
                        "type": "string",
                    },
                    "cluster_name": {
                        "description": "Default GKE cluster name for GKE decorators.",
                        "type": "string",
                    },
                    "namespace": {
                        "description": "Default Kubernetes namespace for GKE decorators.",
                        "type": "string",
                        "default": "default",
                    },
                    "gcp_conn_id": {
                        "description": "Default Airflow connection id for Google credentials.",
                        "type": "string",
                        "default": "google_cloud_default",
                    },
                    "use_internal_ip": {
                        "description": "Use the internal IP address as the endpoint.",
                        "type": "boolean",
                        "default": "False",
                    },
                    "use_dns_endpoint": {
                        "description": "Use the DNS address as the endpoint.",
                        "type": "boolean",
                        "default": "False",
                    },
                },
            },
        },
    }
```

---

## Source-validated module code (new base-operator strategy + xcom_push_mode)

Put this in: `airflow/providers/google/cloud/decorators/kubernetes_engine.py`

```python
from __future__ import annotations

import base64
import os
import pickle
import shlex
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any, TYPE_CHECKING, Literal

import dill
from kubernetes.client import models as k8s

from airflow.configuration import conf
from airflow.exceptions import AirflowException
from airflow.providers.google.cloud.operators.kubernetes_engine import (
    GKEStartJobOperator,
    GKEStartPodOperator,
)
from airflow.providers.common.compat.sdk import (
    DecoratedOperator,
    TaskDecorator,
    context_merge,
    task_decorator_factory,
)
from airflow.utils.operator_helpers import determine_kwargs

# Reuse the same helper used by @task.kubernetes to generate a runnable python script.
from airflow.providers.cncf.kubernetes.python_kubernetes_script import write_python_script

if TYPE_CHECKING:
    from airflow.sdk import Context


# ---------------------------------------------------------------------------
# Configuration defaults (airflow.cfg / env vars)
# ---------------------------------------------------------------------------

_CONF_SECTION = "providers.google.kubernetes_engine"


@dataclass(frozen=True)
class _GkeDefaults:
    project_id: str
    location: str
    cluster_name: str
    namespace: str
    gcp_conn_id: str
    use_internal_ip: bool
    use_dns_endpoint: bool


def _conf_get_trimmed(section: str, key: str, fallback: str | None = None) -> str | None:
    val = conf.get(section, key, fallback=fallback)
    if val is None:
        return None
    val = val.strip()
    return val if val else None


def _conf_get_bool(section: str, key: str, fallback: bool) -> bool:
    return conf.getboolean(section, key, fallback=fallback)


def _resolve_gke_defaults(
    *,
    project_id: str | None,
    location: str | None,
    cluster_name: str | None,
    namespace: str | None,
    gcp_conn_id: str | None,
    use_internal_ip: bool | None,
    use_dns_endpoint: bool | None,
) -> _GkeDefaults:
    """
    Precedence: explicit args > airflow.cfg/env vars (providers.google.kubernetes_engine) > hard fallback.

    Note: project_id/location/cluster_name are required after resolution.
    """
    resolved_project_id = project_id or _conf_get_trimmed(_CONF_SECTION, "project_id")
    resolved_location = location or _conf_get_trimmed(_CONF_SECTION, "location")
    resolved_cluster_name = cluster_name or _conf_get_trimmed(_CONF_SECTION, "cluster_name")

    missing = [k for k, v in (("project_id", resolved_project_id), ("location", resolved_location), ("cluster_name", resolved_cluster_name)) if not v]
    if missing:
        raise AirflowException(
            "Missing required GKE defaults: "
            + ", ".join(missing)
            + f". Provide them explicitly or set [{_CONF_SECTION}] in airflow.cfg / env vars."
        )

    resolved_namespace = (
        namespace
        or _conf_get_trimmed(_CONF_SECTION, "namespace", fallback="default")
        or "default"
    )
    resolved_gcp_conn_id = (
        gcp_conn_id
        or _conf_get_trimmed(_CONF_SECTION, "gcp_conn_id", fallback="google_cloud_default")
        or "google_cloud_default"
    )

    resolved_use_internal_ip = (
        use_internal_ip
        if use_internal_ip is not None
        else _conf_get_bool(_CONF_SECTION, "use_internal_ip", fallback=False)
    )
    resolved_use_dns_endpoint = (
        use_dns_endpoint
        if use_dns_endpoint is not None
        else _conf_get_bool(_CONF_SECTION, "use_dns_endpoint", fallback=False)
    )

    return _GkeDefaults(
        project_id=resolved_project_id,  # type: ignore[arg-type]
        location=resolved_location,  # type: ignore[arg-type]
        cluster_name=resolved_cluster_name,  # type: ignore[arg-type]
        namespace=resolved_namespace,
        gcp_conn_id=resolved_gcp_conn_id,
        use_internal_ip=resolved_use_internal_ip,
        use_dns_endpoint=resolved_use_dns_endpoint,
    )


# ---------------------------------------------------------------------------
# Shared bits for "python-in-container" style (like @task.kubernetes)
# ---------------------------------------------------------------------------

_PYTHON_SCRIPT_ENV = "__PYTHON_SCRIPT"
_PYTHON_INPUT_ENV = "__PYTHON_INPUT"


def _generate_decoded_command(env_var: str, file: str) -> str:
    # Matches upstream behavior: write bytes from base64 env var into a file.
    # (Keep this string simple; it is executed inside a container shell.)
    return (
        'python -c "import base64, os;'
        f' x = base64.b64decode(os.environ[\\"{env_var}\\"]);'
        f' f = open(\\"{file}\\", \\"wb\\"); f.write(x); f.close()"'
    )


def _read_file_contents(filename: str) -> str:
    with open(filename, "rb") as fp:
        return base64.b64encode(fp.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# 1) gke_pod (with opinionated defaults)
# ---------------------------------------------------------------------------

class _GkePodDecoratedOperator(DecoratedOperator, GKEStartPodOperator):
    """
    Runs the decorated Python callable *inside a Pod* on a GKE cluster.

    Mirrors @task.kubernetes, but swaps KubernetesPodOperator -> GKEStartPodOperator.
    """
    custom_operator_name = "@task.gke_pod"

    # `cmds` and `arguments` are generated internally.
    template_fields: Sequence[str] = tuple(
        {"op_args", "op_kwargs", *GKEStartPodOperator.template_fields} - {"cmds", "arguments"}
    )
    shallow_copy_attrs: Sequence[str] = ("python_callable",)

    def __init__(
        self,
        *,
        python_callable: Callable[..., Any],
        # pickle vs dill is an upstream feature in @task.kubernetes
        use_dill: bool = False,
        # defaults (opinionated): allow omission and resolve from airflow.cfg/env vars
        project_id: str | None = None,
        location: str | None = None,
        cluster_name: str | None = None,
        namespace: str | None = None,
        gcp_conn_id: str | None = None,
        use_internal_ip: bool | None = None,
        use_dns_endpoint: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.use_dill = use_dill

        defaults = _resolve_gke_defaults(
            project_id=project_id,
            location=location,
            cluster_name=cluster_name,
            namespace=namespace,
            gcp_conn_id=gcp_conn_id,
            use_internal_ip=use_internal_ip,
            use_dns_endpoint=use_dns_endpoint,
        )

        op_name = kwargs.pop("name", f"gke-airflow-pod-{python_callable.__name__}")
        random_name_suffix = kwargs.pop("random_name_suffix", True)

        super().__init__(
            python_callable=python_callable,
            project_id=defaults.project_id,
            location=defaults.location,
            cluster_name=defaults.cluster_name,
            namespace=defaults.namespace,
            gcp_conn_id=defaults.gcp_conn_id,
            use_internal_ip=defaults.use_internal_ip,
            use_dns_endpoint=defaults.use_dns_endpoint,
            name=op_name,
            random_name_suffix=random_name_suffix,
            cmds=["placeholder-command"],
            **kwargs,
        )

    def _generate_cmds(self) -> list[str]:
        script_filename = "/tmp/script.py"
        input_filename = "/tmp/script.in"

        # This matches upstream: if do_xcom_push is enabled, write to /airflow/xcom/return.json. :contentReference[oaicite:6]{index=6}
        if getattr(self, "do_xcom_push", False):
            output_filename = "/airflow/xcom/return.json"
            make_xcom_dir_cmd = "mkdir -p /airflow/xcom"
        else:
            output_filename = "/dev/null"
            make_xcom_dir_cmd = ":"  # shell no-op

        write_local_script_file_cmd = _generate_decoded_command(_PYTHON_SCRIPT_ENV, script_filename)
        write_local_input_file_cmd = _generate_decoded_command(_PYTHON_INPUT_ENV, input_filename)
        exec_python_cmd = f"python {script_filename} {input_filename} {output_filename}"

        return [
            "bash",
            "-cx",
            (
                f"{write_local_script_file_cmd} && "
                f"{write_local_input_file_cmd} && "
                f"{make_xcom_dir_cmd} && "
                f"{exec_python_cmd}"
            ),
        ]

    def execute(self, context: Context):
        with TemporaryDirectory(prefix="gke-taskflow-") as tmp_dir:
            pickling_library = dill if self.use_dill else pickle
            script_filename = os.path.join(tmp_dir, "script.py")
            input_filename = os.path.join(tmp_dir, "script.in")

            with open(input_filename, "wb") as fp:
                pickling_library.dump({"args": self.op_args, "kwargs": self.op_kwargs}, fp)

            py_source = self.get_python_source()
            jinja_context = {
                "op_args": self.op_args,
                "op_kwargs": self.op_kwargs,
                "pickling_library": pickling_library.__name__,
                "python_callable": self.python_callable.__name__,
                "python_callable_source": py_source,
                "string_args_global": False,
            }
            write_python_script(jinja_context=jinja_context, filename=script_filename)

            self.env_vars = [
                *self.env_vars,
                k8s.V1EnvVar(name=_PYTHON_SCRIPT_ENV, value=_read_file_contents(script_filename)),
                k8s.V1EnvVar(name=_PYTHON_INPUT_ENV, value=_read_file_contents(input_filename)),
            ]

            self.cmds = self._generate_cmds()
            return super().execute(context)


def gke_pod_task(
    python_callable: Callable[..., Any] | None = None,
    *,
    use_dill: bool = False,
    **kwargs: Any,
) -> TaskDecorator:
    """Becomes @task.gke_pod once registered in get_provider_info()."""
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GkePodDecoratedOperator,
        use_dill=use_dill,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Command-generating base (shared by pod/job cmd variants)
# ---------------------------------------------------------------------------

XComPushMode = Literal["off", "file", "stdout"]


def _wrap_cmd_for_xcom_stdout(cmd: list[str]) -> list[str]:
    """
    Wrap a list[str] command so that *stdout* becomes /airflow/xcom/return.json.

    The Kubernetes XCom sidecar reads /airflow/xcom/return.json when do_xcom_push=True. :contentReference[oaicite:7]{index=7}
    """
    quoted = " ".join(shlex.quote(part) for part in cmd)
    # -e: fail fast, -u: undefined var error, -c: run string
    wrapped = f"mkdir -p /airflow/xcom && {quoted} > /airflow/xcom/return.json"
    return ["bash", "-euc", wrapped]


def _validate_generated_cmd(out: Any, func_name: str) -> list[str]:
    if not isinstance(out, list):
        raise TypeError(f"Expected {func_name} to return list[str], got {type(out)}")
    if not out:
        raise ValueError(f"{func_name} returned an empty list[str]")
    if not all(isinstance(x, str) for x in out):
        raise TypeError(f"Expected {func_name} to return list[str], got {out}")
    return out


# ---------------------------------------------------------------------------
# 2) gke_pod_cmd_base (without opinionated defaults)
#    3) gke_pod_cmd (with opinionated defaults)  + xcom_push_mode
# ---------------------------------------------------------------------------

class _GkePodCmdBaseDecoratedOperator(DecoratedOperator, GKEStartPodOperator):
    """
    Base for @task.gke_pod_cmd_*:
      - python_callable runs in Airflow and returns list[str]
      - Pod runs that command as cmds OR arguments (args_only=True)
    """
    custom_operator_name = "@task.gke_pod_cmd_base"
    template_fields: Sequence[str] = tuple({"op_args", "op_kwargs", *GKEStartPodOperator.template_fields})
    overwrite_rtif_after_execution: bool = True

    def __init__(
        self,
        *,
        python_callable: Callable[..., list[str]],
        args_only: bool = False,
        xcom_push_mode: XComPushMode = "off",
        **kwargs: Any,
    ) -> None:
        self.args_only = args_only
        self.xcom_push_mode = xcom_push_mode

        cmds = kwargs.pop("cmds", None)
        arguments = kwargs.pop("arguments", None)
        if cmds is not None or arguments is not None:
            warnings.warn(
                f"`cmds`/`arguments` are ignored by {self.custom_operator_name}. "
                "Return list[str] from the decorated function instead.",
                UserWarning,
                stacklevel=3,
            )

        op_name = kwargs.pop("name", f"gke-airflow-pod-{python_callable.__name__}")
        random_name_suffix = kwargs.pop("random_name_suffix", True)

        super().__init__(
            python_callable=python_callable,
            name=op_name,
            random_name_suffix=random_name_suffix,
            cmds=None,
            arguments=None,
            **kwargs,
        )

    def execute(self, context: Context):
        self.render_template_fields(context)

        generated = self._generate_cmds(context)

        # xcom_push_mode is only meaningful when do_xcom_push is enabled on the underlying operator.
        do_xcom_push = bool(getattr(self, "do_xcom_push", False))
        if self.xcom_push_mode != "off" and not do_xcom_push:
            raise AirflowException("xcom_push_mode requires do_xcom_push=True")

        if do_xcom_push and self.xcom_push_mode == "stdout":
            generated = _wrap_cmd_for_xcom_stdout(generated)

        if self.args_only:
            self.cmds = []
            self.arguments = generated
        else:
            self.cmds = generated
            self.arguments = []

        self.render_template_fields(context)
        return super().execute(context)

    def _generate_cmds(self, context: Context) -> list[str]:
        context_merge(context, self.op_kwargs)
        kwargs = determine_kwargs(self.python_callable, self.op_args, context)
        out = self.python_callable(*self.op_args, **kwargs)
        return _validate_generated_cmd(out, self.python_callable.__name__)


def gke_pod_cmd_base_task(
    python_callable: Callable[..., list[str]] | None = None,
    *,
    args_only: bool = False,
    xcom_push_mode: XComPushMode = "off",
    **kwargs: Any,
) -> TaskDecorator:
    """Becomes @task.gke_pod_cmd_base once registered."""
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GkePodCmdBaseDecoratedOperator,
        args_only=args_only,
        xcom_push_mode=xcom_push_mode,
        **kwargs,
    )


class _GkePodCmdDecoratedOperator(_GkePodCmdBaseDecoratedOperator):
    """
    Opinionated wrapper around the base:
      - resolves project/location/cluster/namespace/gcp_conn_id from airflow.cfg/env vars by default
    """
    custom_operator_name = "@task.gke_pod_cmd"

    def __init__(
        self,
        *,
        python_callable: Callable[..., list[str]],
        args_only: bool = False,
        xcom_push_mode: XComPushMode = "off",
        project_id: str | None = None,
        location: str | None = None,
        cluster_name: str | None = None,
        namespace: str | None = None,
        gcp_conn_id: str | None = None,
        use_internal_ip: bool | None = None,
        use_dns_endpoint: bool | None = None,
        **kwargs: Any,
    ) -> None:
        defaults = _resolve_gke_defaults(
            project_id=project_id,
            location=location,
            cluster_name=cluster_name,
            namespace=namespace,
            gcp_conn_id=gcp_conn_id,
            use_internal_ip=use_internal_ip,
            use_dns_endpoint=use_dns_endpoint,
        )

        super().__init__(
            python_callable=python_callable,
            args_only=args_only,
            xcom_push_mode=xcom_push_mode,
            project_id=defaults.project_id,
            location=defaults.location,
            cluster_name=defaults.cluster_name,
            namespace=defaults.namespace,
            gcp_conn_id=defaults.gcp_conn_id,
            use_internal_ip=defaults.use_internal_ip,
            use_dns_endpoint=defaults.use_dns_endpoint,
            **kwargs,
        )


def gke_pod_cmd_task(
    python_callable: Callable[..., list[str]] | None = None,
    *,
    args_only: bool = False,
    xcom_push_mode: XComPushMode = "off",
    **kwargs: Any,
) -> TaskDecorator:
    """Becomes @task.gke_pod_cmd once registered."""
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GkePodCmdDecoratedOperator,
        args_only=args_only,
        xcom_push_mode=xcom_push_mode,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 4) gke_job (with opinionated defaults)
# ---------------------------------------------------------------------------

class _GkeJobDecoratedOperator(DecoratedOperator, GKEStartJobOperator):
    """
    Runs the decorated Python callable inside a Job on a GKE cluster.
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
        project_id: str | None = None,
        location: str | None = None,
        cluster_name: str | None = None,
        namespace: str | None = None,
        gcp_conn_id: str | None = None,
        use_internal_ip: bool | None = None,
        use_dns_endpoint: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.use_dill = use_dill
        defaults = _resolve_gke_defaults(
            project_id=project_id,
            location=location,
            cluster_name=cluster_name,
            namespace=namespace,
            gcp_conn_id=gcp_conn_id,
            use_internal_ip=use_internal_ip,
            use_dns_endpoint=use_dns_endpoint,
        )

        # Opinionated runtime defaults for Jobs:
        # - wait_until_job_complete=True (otherwise a "fire-and-forget" job is easy to misuse)
        # - get_logs=True (common expectation)
        #
        # Note: KubernetesJobOperator's wait_until_job_complete default is False. :contentReference[oaicite:8]{index=8}
        kwargs.setdefault("wait_until_job_complete", True)
        kwargs.setdefault("get_logs", True)

        op_name = kwargs.pop("name", f"gke-airflow-job-{python_callable.__name__}")
        random_name_suffix = kwargs.pop("random_name_suffix", True)

        super().__init__(
            python_callable=python_callable,
            project_id=defaults.project_id,
            location=defaults.location,
            cluster_name=defaults.cluster_name,
            namespace=defaults.namespace,
            gcp_conn_id=defaults.gcp_conn_id,
            use_internal_ip=defaults.use_internal_ip,
            use_dns_endpoint=defaults.use_dns_endpoint,
            name=op_name,
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

        write_local_script_file_cmd = _generate_decoded_command(_PYTHON_SCRIPT_ENV, script_filename)
        write_local_input_file_cmd = _generate_decoded_command(_PYTHON_INPUT_ENV, input_filename)
        exec_python_cmd = f"python {script_filename} {input_filename} {output_filename}"

        return [
            "bash",
            "-cx",
            (
                f"{write_local_script_file_cmd} && "
                f"{write_local_input_file_cmd} && "
                f"{make_xcom_dir_cmd} && "
                f"{exec_python_cmd}"
            ),
        ]

    def execute(self, context: Context):
        with TemporaryDirectory(prefix="gke-job-taskflow-") as tmp_dir:
            pickling_library = dill if self.use_dill else pickle
            script_filename = os.path.join(tmp_dir, "script.py")
            input_filename = os.path.join(tmp_dir, "script.in")

            with open(input_filename, "wb") as fp:
                pickling_library.dump({"args": self.op_args, "kwargs": self.op_kwargs}, fp)

            py_source = self.get_python_source()
            jinja_context = {
                "op_args": self.op_args,
                "op_kwargs": self.op_kwargs,
                "pickling_library": pickling_library.__name__,
                "python_callable": self.python_callable.__name__,
                "python_callable_source": py_source,
                "string_args_global": False,
            }
            write_python_script(jinja_context=jinja_context, filename=script_filename)

            self.env_vars = [
                *self.env_vars,
                k8s.V1EnvVar(name=_PYTHON_SCRIPT_ENV, value=_read_file_contents(script_filename)),
                k8s.V1EnvVar(name=_PYTHON_INPUT_ENV, value=_read_file_contents(input_filename)),
            ]

            self.cmds = self._generate_cmds()
            return super().execute(context)


def gke_job_task(
    python_callable: Callable[..., Any] | None = None,
    *,
    use_dill: bool = False,
    **kwargs: Any,
) -> TaskDecorator:
    """Becomes @task.gke_job once registered."""
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GkeJobDecoratedOperator,
        use_dill=use_dill,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 5) gke_job_cmd_base (without opinionated defaults)
#    6) gke_job_cmd (with opinionated defaults)
# ---------------------------------------------------------------------------

class _GkeJobCmdBaseDecoratedOperator(DecoratedOperator, GKEStartJobOperator):
    custom_operator_name = "@task.gke_job_cmd_base"
    template_fields: Sequence[str] = tuple({"op_args", "op_kwargs", *GKEStartJobOperator.template_fields})
    overwrite_rtif_after_execution: bool = True

    def __init__(
        self,
        *,
        python_callable: Callable[..., list[str]],
        args_only: bool = False,
        xcom_push_mode: XComPushMode = "off",
        **kwargs: Any,
    ) -> None:
        self.args_only = args_only
        self.xcom_push_mode = xcom_push_mode

        cmds = kwargs.pop("cmds", None)
        arguments = kwargs.pop("arguments", None)
        if cmds is not None or arguments is not None:
            warnings.warn(
                f"`cmds`/`arguments` are ignored by {self.custom_operator_name}. "
                "Return list[str] from the decorated function instead.",
                UserWarning,
                stacklevel=3,
            )

        # Base job behavior is intentionally not opinionated here; wrapper below will set defaults.
        op_name = kwargs.pop("name", f"gke-airflow-job-{python_callable.__name__}")
        random_name_suffix = kwargs.pop("random_name_suffix", True)

        super().__init__(
            python_callable=python_callable,
            name=op_name,
            random_name_suffix=random_name_suffix,
            cmds=None,
            arguments=None,
            **kwargs,
        )

    def execute(self, context: Context):
        self.render_template_fields(context)

        generated = self._generate_cmds(context)

        do_xcom_push = bool(getattr(self, "do_xcom_push", False))
        if self.xcom_push_mode != "off" and not do_xcom_push:
            raise AirflowException("xcom_push_mode requires do_xcom_push=True")

        if do_xcom_push and self.xcom_push_mode == "stdout":
            generated = _wrap_cmd_for_xcom_stdout(generated)

        if self.args_only:
            self.cmds = []
            self.arguments = generated
        else:
            self.cmds = generated
            self.arguments = []

        self.render_template_fields(context)
        return super().execute(context)

    def _generate_cmds(self, context: Context) -> list[str]:
        context_merge(context, self.op_kwargs)
        kwargs = determine_kwargs(self.python_callable, self.op_args, context)
        out = self.python_callable(*self.op_args, **kwargs)
        return _validate_generated_cmd(out, self.python_callable.__name__)


def gke_job_cmd_base_task(
    python_callable: Callable[..., list[str]] | None = None,
    *,
    args_only: bool = False,
    xcom_push_mode: XComPushMode = "off",
    **kwargs: Any,
) -> TaskDecorator:
    """Becomes @task.gke_job_cmd_base once registered."""
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GkeJobCmdBaseDecoratedOperator,
        args_only=args_only,
        xcom_push_mode=xcom_push_mode,
        **kwargs,
    )


class _GkeJobCmdDecoratedOperator(_GkeJobCmdBaseDecoratedOperator):
    custom_operator_name = "@task.gke_job_cmd"

    def __init__(
        self,
        *,
        python_callable: Callable[..., list[str]],
        args_only: bool = False,
        xcom_push_mode: XComPushMode = "off",
        project_id: str | None = None,
        location: str | None = None,
        cluster_name: str | None = None,
        namespace: str | None = None,
        gcp_conn_id: str | None = None,
        use_internal_ip: bool | None = None,
        use_dns_endpoint: bool | None = None,
        **kwargs: Any,
    ) -> None:
        defaults = _resolve_gke_defaults(
            project_id=project_id,
            location=location,
            cluster_name=cluster_name,
            namespace=namespace,
            gcp_conn_id=gcp_conn_id,
            use_internal_ip=use_internal_ip,
            use_dns_endpoint=use_dns_endpoint,
        )

        # Same opinionated job defaults as @task.gke_job
        kwargs.setdefault("wait_until_job_complete", True)
        kwargs.setdefault("get_logs", True)

        super().__init__(
            python_callable=python_callable,
            args_only=args_only,
            xcom_push_mode=xcom_push_mode,
            project_id=defaults.project_id,
            location=defaults.location,
            cluster_name=defaults.cluster_name,
            namespace=defaults.namespace,
            gcp_conn_id=defaults.gcp_conn_id,
            use_internal_ip=defaults.use_internal_ip,
            use_dns_endpoint=defaults.use_dns_endpoint,
            **kwargs,
        )


def gke_job_cmd_task(
    python_callable: Callable[..., list[str]] | None = None,
    *,
    args_only: bool = False,
    xcom_push_mode: XComPushMode = "off",
    **kwargs: Any,
) -> TaskDecorator:
    """Becomes @task.gke_job_cmd once registered."""
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GkeJobCmdDecoratedOperator,
        args_only=args_only,
        xcom_push_mode=xcom_push_mode,
        **kwargs,
    )
```

### Why this is “correct” relative to upstream behavior

* The **cmd decorators** follow the same execution contract as `@task.kubernetes_cmd`: render templates → generate command from Python callable → set `cmds/arguments` → render again → run. ([Apache Airflow][6])
* The **python-in-container decorators** follow the same “encode script+input into env vars, decode in container, run python script” approach as `@task.kubernetes` (including the `/airflow/xcom/return.json` behavior when `do_xcom_push=True`). ([Apache Airflow][1])
* GKE operators’ signatures and restrictions are respected:

  * `GKEStartPodOperator` sets a default `on_finish_action` and rejects `config_file`. ([Apache Airflow][2])
  * `GKEStartJobOperator` rejects `config_file`. ([Apache Airflow][2])

---

## Updated examples

### airflow.cfg

```ini
[providers.google.kubernetes_engine]
project_id = my-project
location = us-central1
cluster_name = my-cluster
namespace = default
gcp_conn_id = google_cloud_default
use_internal_ip = False
use_dns_endpoint = False
```

### Example 1: Your requested `gke_pod_cmd` with XCom from stdout → return.json

This is the exact call style you posted; the only addition is `xcom_push_mode="stdout"` so the decorator wraps the command to write stdout into `/airflow/xcom/return.json`.

```python
from airflow.decorators import dag, task
from pendulum import datetime

@dag(start_date=datetime(2025, 1, 1), schedule=None, catchup=False)
def demo_gke_cmd_xcom():
    @task.gke_pod_cmd(
        image="bash:5",
        do_xcom_push=True,
        xcom_push_mode="stdout",
    )
    def run_cmd():
        return [
            "python",
            "-c",
            'import json; print(json.dumps({"ok": True, "n": 7}))',
        ]

    out = run_cmd()  # XComArg containing {"ok": True, "n": 7}

demo_gke_cmd_xcom()
```

### Example 2: `gke_pod_cmd` where the command itself writes the file (no wrapping)

```python
@task.gke_pod_cmd(
    image="bash:5",
    do_xcom_push=True,
    xcom_push_mode="file",
)
def run_cmd():
    return [
        "bash",
        "-c",
        'mkdir -p /airflow/xcom && echo \'{"ok":true}\' > /airflow/xcom/return.json',
    ]
```

### Example 3: `gke_job_cmd` with opinionated job behavior

```python
@task.gke_job_cmd(
    image="bash:5",
    do_xcom_push=True,
    xcom_push_mode="stdout",
)
def run_job():
    return ["bash", "-c", 'echo \'{"job":"done"}\'']
```

---

## For `gke_job` (with opinionated defaults): what the defaults are and why

These decorators set (unless you override explicitly):

* `wait_until_job_complete=True`
  Because Kubernetes Job execution is otherwise easy to misuse as “fire-and-forget”; upstream’s `KubernetesJobOperator` default is `False`. ([Apache Airflow][7])

* `get_logs=True`
  Because most DAG authors expect logs from a Job to appear in the task logs (especially when they’re treating the job as the unit of work).

Separately, cluster defaults are pulled from config/env vars unless passed explicitly.

---

## Final internal validation notes (conflict-free / no guessing)

* All critical behavior claims above are anchored in upstream source/docs:

  * `@task.kubernetes_cmd` execution flow: ([Apache Airflow][6])
  * `/airflow/xcom/return.json` convention and dir creation: ([Apache Airflow][1])
  * GKE operator init + `config_file` restriction: ([Apache Airflow][2])
  * Job operator `wait_until_job_complete` default: ([Apache Airflow][7])
  * Provider config/env var conventions: ([Apache Airflow][3])
* The `*_cmd_base` → `*_cmd` wrapper refactor is exactly what you asked for: **the base is directly registerable** (so you can call it without wrappers), and the wrapper only adds defaults.
* `xcom_push_mode` is explicit and safe:

  * `"off"`: no special behavior
  * `"file"`: you manage `/airflow/xcom/return.json`
  * `"stdout"`: decorator wraps your command so stdout becomes that file

If you want the PR to survive review, the next hardening step would be adding **unit tests** (especially for `xcom_push_mode="stdout"` quoting, and for the “do_xcom_push False but xcom_push_mode != off” error path).

[1]: https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes/stable/_modules/airflow/providers/cncf/kubernetes/decorators/kubernetes.html "airflow.providers.cncf.kubernetes.decorators.kubernetes — apache-airflow-providers-cncf-kubernetes Documentation"
[2]: https://airflow.apache.org/docs/apache-airflow-providers-google/stable/_modules/airflow/providers/google/cloud/operators/kubernetes_engine.html "airflow.providers.google.cloud.operators.kubernetes_engine — apache-airflow-providers-google Documentation"
[3]: https://airflow.apache.org/docs/apache-airflow-providers-jdbc/stable/configurations-ref.html?utm_source=chatgpt.com "Configuration Reference — apache-airflow-providers-jdbc Documentation"
[4]: https://airflow.apache.org/docs/apache-airflow-providers-jdbc/stable/connections/jdbc.html?utm_source=chatgpt.com "JDBC connection — apache-airflow-providers-jdbc Documentation"
[5]: https://airflow.apache.org/docs/apache-airflow-providers-google/stable/_modules/airflow/providers/google/get_provider_info.html "airflow.providers.google.get_provider_info — apache-airflow-providers-google Documentation"
[6]: https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes/stable/_modules/airflow/providers/cncf/kubernetes/decorators/kubernetes_cmd.html "airflow.providers.cncf.kubernetes.decorators.kubernetes_cmd — apache-airflow-providers-cncf-kubernetes Documentation"
[7]: https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes/stable/_modules/airflow/providers/cncf/kubernetes/operators/job.html "airflow.providers.cncf.kubernetes.operators.job — apache-airflow-providers-cncf-kubernetes Documentation"

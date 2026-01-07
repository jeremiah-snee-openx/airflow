You **should register `_GkePodCmdBaseDecoratedOperator` directly** (via a `gke_pod_cmd_base_task()` that uses it as `decorated_operator_class`). That’s the upstream-established pattern: Airflow’s own `@task.kubernetes_cmd` is implemented as a `DecoratedOperator` + `KubernetesPodOperator` subclass and registered via `task_decorator_factory`—it does **not** register the raw operator class. ([Apache Airflow][1])

Below is a **clean, minimal, upstream-consistent** “full version” for the **cmd family** (pod + job) with:

* **base operator strategy** (`*_cmd_base` is the real engine; `*_cmd` is a thin defaults wrapper)
* **`xcom_push_mode`** including your requested `stdout -> /airflow/xcom/return.json` behavior (works only when `do_xcom_push=True`)
* **provider config defaults** (Airflow config/env vars, not Variables)
* **provider registration snippet**
* **updated examples**

This is intentionally modeled on the upstream `kubernetes_cmd` decorator flow (render → generate → set cmds/args → render → execute). ([Apache Airflow][1])

---

## Recommended package + module names

For the Airflow monorepo Google provider:

* `providers/google/src/airflow/providers/google/cloud/decorators/`
* `providers/google/src/airflow/providers/google/cloud/decorators/kubernetes_engine.py`

Reason: the Google provider already uses the “kubernetes_engine” service bucket for GKE operators (`airflow.providers.google.cloud.operators.kubernetes_engine`). ([Apache Airflow][2])

---

## Provider config options (airflow.cfg / environment variables)

### Config section

Use a provider-scoped section:

* `[providers.google.kubernetes_engine]`

### Keys

* `project_id`
* `location`
* `cluster_name`
* `namespace` (default `default`)
* `gcp_conn_id` (default `google_cloud_default`)
* `use_internal_ip` (default `False`)
* `use_dns_endpoint` (default `False`)

This matches GKE operator parameters (location, cluster_name, use_internal_ip/use_dns_endpoint, project_id, gcp_conn_id, namespace). ([Apache Airflow][2])

### Example `airflow.cfg`

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

### Env var equivalents

```bash
export AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__PROJECT_ID="my-project"
export AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__LOCATION="us-central1"
export AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__CLUSTER_NAME="my-cluster"
export AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__NAMESPACE="default"
export AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__GCP_CONN_ID="google_cloud_default"
export AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__USE_INTERNAL_IP="false"
export AIRFLOW__PROVIDERS_GOOGLE_KUBERNETES_ENGINE__USE_DNS_ENDPOINT="false"
```

---

## Source-validated module code (cmd decorators + base strategy + `xcom_push_mode`)

**File:** `providers/google/src/airflow/providers/google/cloud/decorators/kubernetes_engine.py`

```python
from __future__ import annotations

import shlex
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

from airflow.configuration import conf
from airflow.exceptions import AirflowException
from airflow.providers.common.compat.sdk import (
    DecoratedOperator,
    TaskDecorator,
    context_merge,
    task_decorator_factory,
)
from airflow.providers.google.cloud.operators.kubernetes_engine import (
    GKEStartJobOperator,
    GKEStartPodOperator,
)
from airflow.utils.operator_helpers import determine_kwargs

if TYPE_CHECKING:
    from airflow.sdk import Context


# ---------------------------------------------------------------------------
# Provider config defaults (airflow.cfg / env vars)
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
    Precedence: explicit args > [providers.google.kubernetes_engine] config > hard fallback.

    Note: project_id/location/cluster_name must be present after resolution.
    """
    resolved_project_id = project_id or _conf_get_trimmed(_CONF_SECTION, "project_id")
    resolved_location = location or _conf_get_trimmed(_CONF_SECTION, "location")
    resolved_cluster_name = cluster_name or _conf_get_trimmed(_CONF_SECTION, "cluster_name")

    missing = [
        k
        for k, v in (
            ("project_id", resolved_project_id),
            ("location", resolved_location),
            ("cluster_name", resolved_cluster_name),
        )
        if not v
    ]
    if missing:
        raise AirflowException(
            "Missing required GKE defaults: "
            + ", ".join(missing)
            + f". Provide them explicitly or set [{_CONF_SECTION}] in airflow.cfg / env vars."
        )

    resolved_namespace = namespace or _conf_get_trimmed(_CONF_SECTION, "namespace", fallback="default") or "default"
    resolved_gcp_conn_id = (
        gcp_conn_id
        or _conf_get_trimmed(_CONF_SECTION, "gcp_conn_id", fallback="google_cloud_default")
        or "google_cloud_default"
    )
    resolved_use_internal_ip = (
        use_internal_ip if use_internal_ip is not None else _conf_get_bool(_CONF_SECTION, "use_internal_ip", False)
    )
    resolved_use_dns_endpoint = (
        use_dns_endpoint if use_dns_endpoint is not None else _conf_get_bool(_CONF_SECTION, "use_dns_endpoint", False)
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
# cmd base utilities
# ---------------------------------------------------------------------------

XComPushMode = Literal["off", "file", "stdout"]


def _validate_cmd_list(out: Any, func_name: str) -> list[str]:
    if not isinstance(out, list):
        raise TypeError(f"Expected {func_name} to return list[str], got {type(out)}")
    if not out:
        raise ValueError(f"{func_name} returned an empty list[str]")
    if not all(isinstance(x, str) for x in out):
        raise TypeError(f"Expected {func_name} to return list[str], got {out}")
    return out


def _wrap_cmd_stdout_to_xcom_file(cmd: list[str]) -> list[str]:
    """
    Wrap cmd so that stdout becomes /airflow/xcom/return.json.

    This is compatible with KubernetesPodOperator's XCom sidecar behavior when do_xcom_push=True
    (the sidecar reads /airflow/xcom/return.json). This is the same convention used by the
    upstream kubernetes TaskFlow decorator implementation. :contentReference[oaicite:4]{index=4}
    """
    quoted = " ".join(shlex.quote(part) for part in cmd)
    wrapped = f"mkdir -p /airflow/xcom && {quoted} > /airflow/xcom/return.json"
    return ["bash", "-euc", wrapped]


# ---------------------------------------------------------------------------
# 1) gke_pod_cmd_base (no opinionated defaults; direct registration target)
# ---------------------------------------------------------------------------

class _GkePodCmdBaseDecoratedOperator(DecoratedOperator, GKEStartPodOperator):
    """
    Base implementation: mirrors @task.kubernetes_cmd but runs on GKE via GKEStartPodOperator.

    Flow matches upstream kubernetes_cmd:
      render -> generate list[str] -> set cmds/arguments -> render -> execute :contentReference[oaicite:5]{index=5}
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

        # Own cmds/arguments (same UX as upstream kubernetes_cmd). :contentReference[oaicite:6]{index=6}
        cmds = kwargs.pop("cmds", None)
        arguments = kwargs.pop("arguments", None)
        if cmds is not None or arguments is not None:
            warnings.warn(
                f"The `cmds` and `arguments` are unused in {self.custom_operator_name} decorator. "
                "Return a list[str] from the python_callable (or set args_only=True).",
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

        do_xcom_push = bool(getattr(self, "do_xcom_push", False))
        if self.xcom_push_mode != "off" and not do_xcom_push:
            raise AirflowException("xcom_push_mode requires do_xcom_push=True")

        if do_xcom_push and self.xcom_push_mode == "stdout":
            generated = _wrap_cmd_stdout_to_xcom_file(generated)
        # xcom_push_mode == "file": user is responsible for writing /airflow/xcom/return.json

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
        fn_kwargs = determine_kwargs(self.python_callable, self.op_args, context)
        out = self.python_callable(*self.op_args, **fn_kwargs)
        return _validate_cmd_list(out, self.python_callable.__name__)


def gke_pod_cmd_base_task(
    python_callable: Callable[..., list[str]] | None = None,
    *,
    args_only: bool = False,
    xcom_push_mode: XComPushMode = "off",
    **kwargs: Any,
) -> TaskDecorator:
    """
    Registers as @task.gke_pod_cmd_base.
    NOTE: This directly registers the base decorated operator class (no wrapper needed).
    """
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GkePodCmdBaseDecoratedOperator,
        args_only=args_only,
        xcom_push_mode=xcom_push_mode,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 2) gke_pod_cmd (opinionated defaults wrapper)
# ---------------------------------------------------------------------------

class _GkePodCmdDecoratedOperator(_GkePodCmdBaseDecoratedOperator):
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
    """Registers as @task.gke_pod_cmd."""
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GkePodCmdDecoratedOperator,
        args_only=args_only,
        xcom_push_mode=xcom_push_mode,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 3) gke_job_cmd_base (no opinionated defaults; direct registration target)
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
                f"The `cmds` and `arguments` are unused in {self.custom_operator_name} decorator. "
                "Return a list[str] from the python_callable (or set args_only=True).",
                UserWarning,
                stacklevel=3,
            )

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
            generated = _wrap_cmd_stdout_to_xcom_file(generated)

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
        fn_kwargs = determine_kwargs(self.python_callable, self.op_args, context)
        out = self.python_callable(*self.op_args, **fn_kwargs)
        return _validate_cmd_list(out, self.python_callable.__name__)


def gke_job_cmd_base_task(
    python_callable: Callable[..., list[str]] | None = None,
    *,
    args_only: bool = False,
    xcom_push_mode: XComPushMode = "off",
    **kwargs: Any,
) -> TaskDecorator:
    """Registers as @task.gke_job_cmd_base."""
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GkeJobCmdBaseDecoratedOperator,
        args_only=args_only,
        xcom_push_mode=xcom_push_mode,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 4) gke_job_cmd (opinionated defaults wrapper)
# ---------------------------------------------------------------------------

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
    """Registers as @task.gke_job_cmd."""
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GkeJobCmdDecoratedOperator,
        args_only=args_only,
        xcom_push_mode=xcom_push_mode,
        **kwargs,
    )
```

### Why this is correct for upstream

* The “base” classes implement the exact same **cmd-decorator control flow** as upstream `@task.kubernetes_cmd` (the reference implementation). ([Apache Airflow][1])
* The wrapper classes only add **policy** (defaults resolution).
* The defaults are aligned with the real GKE operator params/signatures. ([Apache Airflow][2])
* You are *not* attempting to register `GKEStartPodOperator` directly as a TaskFlow decorator (which would skip calling the function to generate commands, and would not be a `DecoratedOperator`-style TaskFlow operator). The upstream reference uses `DecoratedOperator` for exactly this reason. ([Apache Airflow][1])

---

## Provider registration snippet (task-decorators)

Airflow’s generated `get_provider_info.py` warns to edit the template. ([Apache Airflow][3])

Add to the Google provider’s `get_provider_info_TEMPLATE.py.jinja2`:

```python
"task-decorators": [
    {
        "name": "gke_pod_cmd_base",
        "class-name": "airflow.providers.google.cloud.decorators.kubernetes_engine.gke_pod_cmd_base_task",
    },
    {
        "name": "gke_pod_cmd",
        "class-name": "airflow.providers.google.cloud.decorators.kubernetes_engine.gke_pod_cmd_task",
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
```

And add the provider config section as shown earlier (same template supports `config`, as seen in other providers’ generated metadata). ([Apache Airflow][3])

---

## Updated examples (including your “stdout → XCom” requirement)

### Example: `gke_pod_cmd` returns JSON dict via stdout redirection

```python
from airflow.decorators import dag, task
from pendulum import datetime

@dag(start_date=datetime(2025, 1, 1), schedule=None, catchup=False)
def demo():
    @task.gke_pod_cmd(
        image="python:3.12-slim",
        do_xcom_push=True,
        xcom_push_mode="stdout",
    )
    def run_cmd():
        return [
            "python",
            "-c",
            'import json; print(json.dumps({"ok": True, "n": 7}))',
        ]

    out = run_cmd()  # XComArg resolving to {"ok": True, "n": 7}

demo()
```

### Example: `gke_pod_cmd_base` (no defaults) with explicit cluster args

```python
@task.gke_pod_cmd_base(
    image="bash:5",
    location="us-central1",
    cluster_name="my-cluster",
    project_id="my-project",
    namespace="default",
    gcp_conn_id="google_cloud_default",
    do_xcom_push=True,
    xcom_push_mode="stdout",
)
def run_cmd():
    return ["bash", "-c", 'echo \'{"hello":"world"}\'']
```

### Example: `xcom_push_mode="file"` (you write the file yourself)

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

---

## Quick review checklist (small chunks, sanity-focused)

1. **Decorator engine correctness**

   * Implements upstream `kubernetes_cmd` flow: render → generate → assign → render → execute. ([Apache Airflow][1])
   * Uses `DecoratedOperator` + concrete operator class (same pattern as upstream). ([Apache Airflow][1])

2. **GKE operator compatibility**

   * Uses supported params for `GKEStartPodOperator` / `GKEStartJobOperator` and doesn’t touch forbidden `config_file` handling (operator itself enforces that). ([Apache Airflow][2])

3. **XCom behavior**

   * `xcom_push_mode` is gated behind `do_xcom_push=True` (otherwise it’s a config error).
   * `stdout` mode writes exactly to `/airflow/xcom/return.json` and ensures directory exists.

4. **Upstream review expectations**

   * Base is directly registerable (`*_cmd_base_task` uses `_Gke*CmdBaseDecoratedOperator`).
   * Opinionated defaults live in wrapper (`*_cmd_task` uses `_resolve_gke_defaults`).
   * Defaults use Airflow config/env vars (provider section), not Variables.

If you want to maximize PR survivability, the next step is adding **unit tests** mirroring the cncf `kubernetes_cmd` decorator tests (especially for `stdout` wrapping + quoting).

[1]: https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes/stable/_modules/airflow/providers/cncf/kubernetes/decorators/kubernetes_cmd.html "airflow.providers.cncf.kubernetes.decorators.kubernetes_cmd — apache-airflow-providers-cncf-kubernetes Documentation"
[2]: https://airflow.apache.org/docs/apache-airflow-providers-google/stable/_modules/airflow/providers/google/cloud/operators/kubernetes_engine.html "airflow.providers.google.cloud.operators.kubernetes_engine — apache-airflow-providers-google Documentation"
[3]: https://airflow.apache.org/docs/apache-airflow-providers-standard/stable/_modules/airflow/providers/standard/get_provider_info.html?utm_source=chatgpt.com "airflow.providers.standard.get_provider_info — apache-airflow-providers ..."

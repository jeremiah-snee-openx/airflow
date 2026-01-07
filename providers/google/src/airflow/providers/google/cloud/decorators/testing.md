Got it — here’s the **minimal, upstream-style unit test suite** you’d add to the **Google provider** for the new `*_cmd_base` / `*_cmd` decorators, including the **`xcom_push_mode="stdout"` → `/airflow/xcom/return.json`** behavior.

This is deliberately modeled after the behavior/flow of the upstream `@task.kubernetes_cmd` decorated operator (render → generate → set → render → execute).

---

## 1) Add unit tests

### File

`providers/google/tests/unit/google/cloud/decorators/test_kubernetes_engine_cmd_decorator.py`

```python
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from airflow.exceptions import AirflowException
from airflow.providers.google.cloud.decorators import kubernetes_engine as mod
from airflow.providers.google.cloud.operators.kubernetes_engine import (
    GKEStartJobOperator,
    GKEStartPodOperator,
)


def _minimal_context():
    # render_template_fields() is called in execute(); most operators expect at least ti in context.
    return {"ti": MagicMock()}


# ---------------------------------------------------------------------------
# Pod cmd base
# ---------------------------------------------------------------------------

def test_gke_pod_cmd_base_sets_cmds_when_args_only_false(monkeypatch):
    # Arrange
    monkeypatch.setattr(GKEStartPodOperator, "execute", lambda self, context: "ok")

    def build_cmd():
        return ["echo", "hello"]

    op = mod._GkePodCmdBaseDecoratedOperator(
        task_id="t",
        python_callable=build_cmd,
        image="alpine:3.20",
        project_id="p",
        location="us-central1",
        cluster_name="c",
        namespace="default",
        gcp_conn_id="google_cloud_default",
        args_only=False,
        xcom_push_mode="off",
    )

    # Act
    result = op.execute(_minimal_context())

    # Assert
    assert result == "ok"
    assert op.cmds == ["echo", "hello"]
    assert op.arguments == []


def test_gke_pod_cmd_base_sets_arguments_when_args_only_true(monkeypatch):
    monkeypatch.setattr(GKEStartPodOperator, "execute", lambda self, context: "ok")

    def build_args():
        return ["--flag", "value"]

    op = mod._GkePodCmdBaseDecoratedOperator(
        task_id="t",
        python_callable=build_args,
        image="alpine:3.20",
        project_id="p",
        location="us-central1",
        cluster_name="c",
        namespace="default",
        gcp_conn_id="google_cloud_default",
        args_only=True,
        xcom_push_mode="off",
    )

    op.execute(_minimal_context())

    assert op.cmds == []
    assert op.arguments == ["--flag", "value"]


def test_gke_pod_cmd_base_validates_return_type(monkeypatch):
    monkeypatch.setattr(GKEStartPodOperator, "execute", lambda self, context: "ok")

    def bad():
        return "echo hello"  # not list[str]

    op = mod._GkePodCmdBaseDecoratedOperator(
        task_id="t",
        python_callable=bad,  # type: ignore[arg-type]
        image="alpine:3.20",
        project_id="p",
        location="us-central1",
        cluster_name="c",
        namespace="default",
        gcp_conn_id="google_cloud_default",
        args_only=False,
        xcom_push_mode="off",
    )

    with pytest.raises(TypeError):
        op.execute(_minimal_context())


def test_gke_pod_cmd_base_xcom_mode_requires_do_xcom_push(monkeypatch):
    monkeypatch.setattr(GKEStartPodOperator, "execute", lambda self, context: "ok")

    def build_cmd():
        return ["echo", '{"ok": true}']

    op = mod._GkePodCmdBaseDecoratedOperator(
        task_id="t",
        python_callable=build_cmd,
        image="alpine:3.20",
        project_id="p",
        location="us-central1",
        cluster_name="c",
        namespace="default",
        gcp_conn_id="google_cloud_default",
        args_only=False,
        xcom_push_mode="stdout",
        do_xcom_push=False,
    )

    with pytest.raises(AirflowException, match="xcom_push_mode requires do_xcom_push=True"):
        op.execute(_minimal_context())


def test_gke_pod_cmd_base_xcom_stdout_wraps_command(monkeypatch):
    """
    Verifies: do_xcom_push=True + xcom_push_mode='stdout' wraps the command so stdout is written
    into /airflow/xcom/return.json (the Kubernetes XCom sidecar convention). 
    """
    monkeypatch.setattr(GKEStartPodOperator, "execute", lambda self, context: "ok")

    # include spaces & quotes to exercise quoting
    def build_cmd():
        return ["python", "-c", 'print("{\\"ok\\": true, \\"msg\\": \\"a b\\"}")']

    op = mod._GkePodCmdBaseDecoratedOperator(
        task_id="t",
        python_callable=build_cmd,
        image="python:3.12-slim",
        project_id="p",
        location="us-central1",
        cluster_name="c",
        namespace="default",
        gcp_conn_id="google_cloud_default",
        args_only=False,
        xcom_push_mode="stdout",
        do_xcom_push=True,
    )

    op.execute(_minimal_context())

    assert op.cmds[:2] == ["bash", "-euc"]
    shell = op.cmds[2]
    assert "mkdir -p /airflow/xcom" in shell
    assert "> /airflow/xcom/return.json" in shell

    # sanity: the python invocation should be present (quoted)
    assert "python" in shell
    assert "-c" in shell


def test_gke_pod_cmd_base_xcom_file_does_not_wrap(monkeypatch):
    monkeypatch.setattr(GKEStartPodOperator, "execute", lambda self, context: "ok")

    def build_cmd():
        return ["bash", "-c", "echo '{\"ok\":true}' > /airflow/xcom/return.json"]

    op = mod._GkePodCmdBaseDecoratedOperator(
        task_id="t",
        python_callable=build_cmd,
        image="bash:5",
        project_id="p",
        location="us-central1",
        cluster_name="c",
        namespace="default",
        gcp_conn_id="google_cloud_default",
        args_only=False,
        xcom_push_mode="file",
        do_xcom_push=True,
    )

    op.execute(_minimal_context())

    # no wrapping => cmds should be exactly the returned command
    assert op.cmds == ["bash", "-c", "echo '{\"ok\":true}' > /airflow/xcom/return.json"]


# ---------------------------------------------------------------------------
# Pod cmd wrapper (opinionated defaults)
# ---------------------------------------------------------------------------

def test_gke_pod_cmd_wrapper_resolves_defaults_from_conf(monkeypatch):
    # Patch conf.get / conf.getboolean used by _resolve_gke_defaults
    def fake_get(section, key, fallback=None):
        if section != "providers.google.kubernetes_engine":
            return fallback
        return {
            "project_id": "proj",
            "location": "loc",
            "cluster_name": "clu",
            "namespace": "ns",
            "gcp_conn_id": "conn",
        }.get(key, fallback)

    def fake_getboolean(section, key, fallback=False):
        if section != "providers.google.kubernetes_engine":
            return fallback
        return {"use_internal_ip": True, "use_dns_endpoint": False}.get(key, fallback)

    monkeypatch.setattr(mod.conf, "get", fake_get)
    monkeypatch.setattr(mod.conf, "getboolean", fake_getboolean)

    monkeypatch.setattr(GKEStartPodOperator, "execute", lambda self, context: "ok")

    def build_cmd():
        return ["echo", "hi"]

    op = mod._GkePodCmdDecoratedOperator(
        task_id="t",
        python_callable=build_cmd,
        image="alpine:3.20",
        args_only=False,
        xcom_push_mode="off",
        # intentionally omit project_id/location/cluster_name/namespace/gcp_conn_id
    )

    op.execute(_minimal_context())

    assert op.project_id == "proj"
    assert op.location == "loc"
    assert op.cluster_name == "clu"
    assert op.namespace == "ns"
    assert op.gcp_conn_id == "conn"
    assert op.use_internal_ip is True
    assert op.use_dns_endpoint is False


# ---------------------------------------------------------------------------
# Job cmd base
# ---------------------------------------------------------------------------

def test_gke_job_cmd_base_sets_cmds(monkeypatch):
    monkeypatch.setattr(GKEStartJobOperator, "execute", lambda self, context: "ok")

    def build_cmd():
        return ["echo", "job"]

    op = mod._GkeJobCmdBaseDecoratedOperator(
        task_id="t",
        python_callable=build_cmd,
        image="alpine:3.20",
        project_id="p",
        location="us-central1",
        cluster_name="c",
        namespace="default",
        gcp_conn_id="google_cloud_default",
        args_only=False,
        xcom_push_mode="off",
    )

    op.execute(_minimal_context())

    assert op.cmds == ["echo", "job"]
    assert op.arguments == []


def test_gke_job_cmd_base_xcom_stdout_wraps(monkeypatch):
    monkeypatch.setattr(GKEStartJobOperator, "execute", lambda self, context: "ok")

    def build_cmd():
        return ["bash", "-c", "echo '{\"ok\":true}'"]

    op = mod._GkeJobCmdBaseDecoratedOperator(
        task_id="t",
        python_callable=build_cmd,
        image="bash:5",
        project_id="p",
        location="us-central1",
        cluster_name="c",
        namespace="default",
        gcp_conn_id="google_cloud_default",
        args_only=False,
        xcom_push_mode="stdout",
        do_xcom_push=True,
    )

    op.execute(_minimal_context())

    assert op.cmds[:2] == ["bash", "-euc"]
    assert "> /airflow/xcom/return.json" in op.cmds[2]
```

### What these tests cover

* **cmd generation + assignment** (`args_only` True/False)
* **type validation** of the decorated function return (`list[str]`)
* **`xcom_push_mode` gating** behind `do_xcom_push=True`
* **`stdout` wrapping** to `/airflow/xcom/return.json` (your requested fix)
* **defaults wrapper** pulls config from `airflow.cfg` / env via `conf.get`/`conf.getboolean`

And importantly: these tests **do not require a live GKE cluster**—they patch `GKEStartPodOperator.execute` / `GKEStartJobOperator.execute` to a no-op return.

---

## 2) Test execution target (what you run locally)

From the repo root (typical patterns), you’ll run provider unit tests scoped to Google provider. A common way is:

* `pytest providers/google/tests/unit/google/cloud/decorators/test_kubernetes_engine_cmd_decorator.py`

If you’re using Breeze, you’ll run the equivalent “providers tests” command for the google provider (Airflow’s dev tooling supports targeted pytest runs; exact wrappers vary by repo version).

---

## 3) Notes that help this survive upstream review

* You’re mirroring upstream `kubernetes_cmd` semantics (not inventing a new execution model).
* You’re testing only the new behavior you added (`xcom_push_mode`) and the contract of “decorated function produces a command list”.
* You’re not coupling the tests to Kubernetes connectivity or GCP auth.

---

Here’s the “extra mile” addition: **tests that `cmds` / `arguments` passed by the user are ignored and a warning is emitted**, matching the UX contract of upstream `kubernetes_cmd` decorators.

Add these tests to the same file:
`providers/google/tests/unit/google/cloud/decorators/test_kubernetes_engine_cmd_decorator.py`

```python
def test_gke_pod_cmd_base_warns_and_ignores_user_cmds_arguments(monkeypatch):
    """
    Passing cmds/arguments directly should be ignored by the decorator and should warn,
    like upstream @task.kubernetes_cmd. 
    """
    monkeypatch.setattr(GKEStartPodOperator, "execute", lambda self, context: "ok")

    def build_cmd():
        return ["echo", "from-callable"]

    with pytest.warns(UserWarning):
        op = mod._GkePodCmdBaseDecoratedOperator(
            task_id="t",
            python_callable=build_cmd,
            image="alpine:3.20",
            project_id="p",
            location="us-central1",
            cluster_name="c",
            namespace="default",
            gcp_conn_id="google_cloud_default",
            args_only=False,
            xcom_push_mode="off",
            # user tries to set these directly (should be ignored)
            cmds=["echo", "user-cmds"],
            arguments=["user-args"],
        )

    op.execute(_minimal_context())

    # Confirm decorator output wins
    assert op.cmds == ["echo", "from-callable"]
    assert op.arguments == []


def test_gke_job_cmd_base_warns_and_ignores_user_cmds_arguments(monkeypatch):
    """
    Same behavior for Job variant.
    """
    monkeypatch.setattr(GKEStartJobOperator, "execute", lambda self, context: "ok")

    def build_cmd():
        return ["echo", "from-callable"]

    with pytest.warns(UserWarning):
        op = mod._GkeJobCmdBaseDecoratedOperator(
            task_id="t",
            python_callable=build_cmd,
            image="alpine:3.20",
            project_id="p",
            location="us-central1",
            cluster_name="c",
            namespace="default",
            gcp_conn_id="google_cloud_default",
            args_only=False,
            xcom_push_mode="off",
            cmds=["echo", "user-cmds"],
            arguments=["user-args"],
        )

    op.execute(_minimal_context())

    assert op.cmds == ["echo", "from-callable"]
    assert op.arguments == []
```

### Notes (why this is PR-friendly)

* Uses `pytest.warns(UserWarning)` so the test is robust but not over-specific to exact wording.
* Verifies both the **warning** and the **override behavior**, which is exactly what reviewers care about: “Does the decorator own the command contract?”

Add these two tests to the same file
`providers/google/tests/unit/google/cloud/decorators/test_kubernetes_engine_cmd_decorator.py`

They specifically cover the “only one of `cmds` / `arguments` was provided” cases for both Pod and Job variants.

```python
@pytest.mark.parametrize(
    "kwargs",
    [
        {"cmds": ["echo", "user-cmds-only"]},
        {"arguments": ["--user-args-only"]},
    ],
)
def test_gke_pod_cmd_base_warns_when_only_one_of_cmds_or_arguments_is_set(monkeypatch, kwargs):
    """
    If a user provides only `cmds` OR only `arguments`, the decorator should still warn
    and still ignore those values in favor of the python_callable result.
    Mirrors upstream @task.kubernetes_cmd behavior. 
    """
    monkeypatch.setattr(GKEStartPodOperator, "execute", lambda self, context: "ok")

    def build_cmd():
        return ["echo", "from-callable"]

    with pytest.warns(UserWarning):
        op = mod._GkePodCmdBaseDecoratedOperator(
            task_id="t",
            python_callable=build_cmd,
            image="alpine:3.20",
            project_id="p",
            location="us-central1",
            cluster_name="c",
            namespace="default",
            gcp_conn_id="google_cloud_default",
            args_only=False,
            xcom_push_mode="off",
            **kwargs,
        )

    op.execute(_minimal_context())

    assert op.cmds == ["echo", "from-callable"]
    assert op.arguments == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cmds": ["echo", "user-cmds-only"]},
        {"arguments": ["--user-args-only"]},
    ],
)
def test_gke_job_cmd_base_warns_when_only_one_of_cmds_or_arguments_is_set(monkeypatch, kwargs):
    """
    Same contract for Job cmd base decorator.
    """
    monkeypatch.setattr(GKEStartJobOperator, "execute", lambda self, context: "ok")

    def build_cmd():
        return ["echo", "from-callable"]

    with pytest.warns(UserWarning):
        op = mod._GkeJobCmdBaseDecoratedOperator(
            task_id="t",
            python_callable=build_cmd,
            image="alpine:3.20",
            project_id="p",
            location="us-central1",
            cluster_name="c",
            namespace="default",
            gcp_conn_id="google_cloud_default",
            args_only=False,
            xcom_push_mode="off",
            **kwargs,
        )

    op.execute(_minimal_context())

    assert op.cmds == ["echo", "from-callable"]
    assert op.arguments == []
```

These tests are intentionally narrow:

* They validate the **warning** happens even when only one field is passed.
* They validate the decorator still **owns** `cmds/arguments` and the callable output wins.



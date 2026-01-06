from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from airflow.providers.google.cloud.operators.kubernetes_engine import GKEStartPodOperator
from airflow.providers.common.compat.sdk import (
    DecoratedOperator,
    TaskDecorator,
    context_merge,
    task_decorator_factory,
)
from airflow.utils.operator_helpers import determine_kwargs

if TYPE_CHECKING:
    from airflow.sdk import Context


class _GKECmdDecoratedOperator(DecoratedOperator, GKEStartPodOperator):
    """
    TaskFlow decorator-backed operator:

    - Your python_callable runs in the Airflow worker to *generate* a command list.
    - The pod runs that command in GKE via GKEStartPodOperator.
    """
    custom_operator_name = "@task.gke_cmd"

    # Include op_args/op_kwargs for TaskFlow argument passing, plus all GKEStartPodOperator template fields.
    template_fields: Sequence[str] = tuple({"op_args", "op_kwargs", *GKEStartPodOperator.template_fields})

    # Keep rendered task instance fields accurate after we mutate cmds/arguments at runtime.
    overwrite_rtif_after_execution: bool = True

    def __init__(
        self,
        *,
        python_callable: Callable[..., list[str]],
        args_only: bool = False,
        multiple_outputs: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.args_only = args_only
        self._multiple_outputs = bool(multiple_outputs)

        # Match kubernetes_cmd behavior: these are "owned" by the decorator, not user-supplied.
        cmds = kwargs.pop("cmds", None)
        arguments = kwargs.pop("arguments", None)
        if cmds is not None or arguments is not None:
            warnings.warn(
                f"The `cmds` and `arguments` are unused in {self.custom_operator_name}. "
                "Return a list[str] from the python_callable instead (or set args_only=True).",
                UserWarning,
                stacklevel=3,
            )

        # Sensible defaults, same spirit as @task.kubernetes_cmd.
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
        # 1) render templates so args/kwargs/macros are usable to generate the command
        self.render_template_fields(context)

        # 2) run python_callable in the worker to generate the command list
        generated = self._generate_cmds(context)

        # 3) inject into the operator fields
        if self.args_only:
            self.cmds = []
            self.arguments = generated
        else:
            self.cmds = generated
            self.arguments = []

        # 4) render again so returned strings can also contain templates/macros
        self.render_template_fields(context)

        # 5) run the actual pod in GKE
        result = super().execute(context)

        # Optional "multiple_outputs": push each dict key as its own XCom.
        if self._multiple_outputs and isinstance(result, dict):
            ti = context["ti"]
            for k, v in result.items():
                ti.xcom_push(key=k, value=v)

        return result

    def _generate_cmds(self, context: Context) -> list[str]:
        # Merge any context keys into op_kwargs (matches kubernetes_cmd pattern).
        context_merge(context, self.op_kwargs)

        # Filter kwargs based on the python_callable signature (so you can accept context params cleanly).
        kwargs = determine_kwargs(self.python_callable, self.op_args, context)

        generated_cmds = self.python_callable(*self.op_args, **kwargs)

        if not isinstance(generated_cmds, list) or not all(isinstance(x, str) for x in generated_cmds):
            raise TypeError("Expected python_callable to return list[str]")

        if not generated_cmds:
            raise ValueError("python_callable returned an empty command list")

        return generated_cmds


def gke_cmd_task(
    python_callable: Callable[..., list[str]] | None = None,
    *,
    args_only: bool = False,
    multiple_outputs: bool | None = None,
    **kwargs: Any,
) -> TaskDecorator:
    """
    Usage: @task.gke_cmd(...)(fn) or as decorator syntax.

    Accepts any GKEStartPodOperator kwargs via **kwargs, plus:
      - args_only: treat returned list as container arguments (keep image entrypoint)
      - multiple_outputs: if the operator returns a dict, push each key as separate XCom
    """
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GKECmdDecoratedOperator,
        args_only=args_only,
        multiple_outputs=multiple_outputs,
        **kwargs,
    )

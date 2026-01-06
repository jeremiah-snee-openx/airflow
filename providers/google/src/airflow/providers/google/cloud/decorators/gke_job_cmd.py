from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from airflow.providers.google.cloud.operators.kubernetes_engine import GKEStartJobOperator
from airflow.providers.common.compat.sdk import (
    DecoratedOperator,
    TaskDecorator,
    context_merge,
    task_decorator_factory,
)
from airflow.utils.operator_helpers import determine_kwargs

if TYPE_CHECKING:
    from airflow.sdk import Context


class _GKEJobCmdDecoratedOperator(DecoratedOperator, GKEStartJobOperator):
    """
    TaskFlow decorator-backed operator:

    - python_callable runs in the Airflow worker to *generate* a list[str]
    - that list becomes cmds (or arguments if args_only=True)
    - the Job runs in GKE via GKEStartJobOperator
    """

    custom_operator_name = "@task.gke_job_cmd"

    # Include op_args/op_kwargs for TaskFlow argument passing, plus operator template fields.
    template_fields: Sequence[str] = tuple(
        {"op_args", "op_kwargs", *GKEStartJobOperator.template_fields}
    )

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

        # Match kubernetes_cmd behavior: cmds/arguments are owned by the decorator.
        cmds = kwargs.pop("cmds", None)
        arguments = kwargs.pop("arguments", None)
        if cmds is not None or arguments is not None:
            warnings.warn(
                f"The `cmds` and `arguments` are unused in {self.custom_operator_name}. "
                "Return list[str] from the python_callable instead (or set args_only=True).",
                UserWarning,
                stacklevel=3,
            )

        super().__init__(
            python_callable=python_callable,
            cmds=None,
            arguments=None,
            **kwargs,
        )

    def execute(self, context: Context):
        # 1) Render templates so op_kwargs can include templated/macros values
        self.render_template_fields(context)

        # 2) Generate the command list in the Airflow worker
        generated = self._generate_cmds(context)

        # 3) Inject into the Job container
        if self.args_only:
            self.cmds = []
            self.arguments = generated
        else:
            self.cmds = generated
            self.arguments = []

        # 4) Render again so returned strings can contain templates/macros too
        self.render_template_fields(context)

        # 5) Run the real operator
        result = super().execute(context)

        # Optional: emulate TaskFlow multiple_outputs by splitting dict keys into separate XComs.
        if self._multiple_outputs and isinstance(result, dict):
            ti = context["ti"]
            for k, v in result.items():
                ti.xcom_push(key=str(k), value=v)

        return result

    def _generate_cmds(self, context: Context) -> list[str]:
        # Merge context into op_kwargs (same idea as upstream kubernetes_cmd)
        context_merge(context, self.op_kwargs)

        # Pass only relevant kwargs based on python_callable signature
        kwargs = determine_kwargs(self.python_callable, self.op_args, context)

        generated_cmds = self.python_callable(*self.op_args, **kwargs)

        if not isinstance(generated_cmds, list) or not all(
            isinstance(x, str) for x in generated_cmds
        ):
            raise TypeError("Expected python_callable to return list[str]")

        if not generated_cmds:
            raise ValueError("python_callable returned an empty command list")

        return generated_cmds


def gke_job_cmd_task(
    python_callable: Callable[..., list[str]] | None = None,
    *,
    args_only: bool = False,
    multiple_outputs: bool | None = None,
    **kwargs: Any,
) -> TaskDecorator:
    """
    Registerable provider TaskFlow decorator.
    Accepts all GKEStartJobOperator kwargs via **kwargs (except config_file). :contentReference[oaicite:2]{index=2}

    Extra kwargs:
      - args_only: treat returned list[str] as container arguments (use image entrypoint)
      - multiple_outputs: if result is a dict, split into separate XCom keys
    """
    return task_decorator_factory(
        python_callable=python_callable,
        decorated_operator_class=_GKEJobCmdDecoratedOperator,
        args_only=args_only,
        multiple_outputs=multiple_outputs,
        **kwargs,
    )

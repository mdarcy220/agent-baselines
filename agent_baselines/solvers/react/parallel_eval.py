"""Subprocess-based parallel evaluation for inspect_ai.

This module provides a subprocess wrapper around inspect_ai.eval() to enable
parallel evaluation when using DSPy optimizers. Each eval runs in a separate
subprocess to avoid inspect_ai's concurrent call restriction.
"""

import logging
import multiprocessing as mp
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


def _run_eval_worker(args):
    """Worker function that runs in a subprocess via Pool.map.

    Args:
        args: Tuple of (sample_id, system_message, continue_message, model_name,
              task_name, tool_config_dict)

    Returns:
        Tuple of ("success", score_value, sample_id, scorer_name) or
        ("error", error_message)
    """
    sample_id, system_message, continue_message, model_name, task_name, tool_config_dict = args

    try:
        print(f"Starting eval worker for sample {sample_id}", file=sys.stderr)
        # Import inside worker to ensure fresh state
        from astabench.evals.sqa import sqa_dev
        from astabench.tools import ToolsetConfig
        from inspect_ai import eval as inspect_eval

        from agent_baselines.solvers.react.dspy_agent import (
            create_agent_with_dspy_prompts,
        )

        # Recreate task and tool config
        if task_name == "sqa_dev":
            base_task = sqa_dev()
        else:
            raise ValueError(f"Unknown task: {task_name}")

        tool_config = ToolsetConfig.model_validate(tool_config_dict)

        # Create solver with candidate prompts
        agent_solver = create_agent_with_dspy_prompts(
            system_message_text=system_message,
            continue_message_text=continue_message,
            max_steps=10,
            tools=tool_config.create_tools(),
            add_submit_tool=not tool_config.with_editor_submit,
        )

        # Run eval on just this sample
        logs = inspect_eval(
            tasks=[base_task],
            model=model_name,
            solver=agent_solver,
            sample_id=sample_id,
            log_dir=".dspy_cache",
            log_level="warning",
            display="plain",
        )
        print(f"Eval worker for sample {sample_id} completed", file=sys.stderr)

        # Extract score
        assert logs and len(logs) > 0, f"No logs returned for sample {sample_id}"
        eval_log = logs[0]

        assert (
            eval_log.samples and len(eval_log.samples) > 0
        ), f"No samples in eval log for {sample_id}"

        sample = eval_log.samples[0]
        assert sample.scores, f"No scores for sample {sample_id}"

        print(f"Scores for sample {sample_id} are complete", file=sys.stderr)

        # Extract global_avg from the sample's scores
        for scorer_name, score in sample.scores.items():
            if isinstance(score.value, dict) and "global_avg" in score.value:
                score_value = float(score.value["global_avg"])
                print(
                    f"Eval worker for sample {sample_id} returning success",
                    file=sys.stderr,
                )
                return ("success", score_value, sample_id, scorer_name)

        print(f"No global_avg found in scores for sample {sample_id}", file=sys.stderr)

        # If we get here, score format is wrong
        score_info = {
            name: type(score.value).__name__ for name, score in sample.scores.items()
        }

        error_msg = f"No global_avg found in scores for sample {sample_id}. Available scores: {score_info}"
        print(f"Eval worker returning error: {error_msg}", file=sys.stderr)
        return ("error", error_msg)

    except Exception as e:
        # Return exception info
        import traceback

        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"Exception in eval worker for sample {sample_id}: {error_msg}", file=sys.stderr)
        return ("error", error_msg)


def eval_in_subprocess(
    sample_id: str,
    system_message: str,
    continue_message: str,
    model_name: str,
    task_name: str = "sqa_dev",
    tool_config_dict: dict[str, Any] | None = None,
    timeout: int = 600,
) -> float:
    """Run inspect_ai.eval() in an isolated subprocess.

    Args:
        sample_id: ID of the sample to evaluate
        system_message: System message prompt to use
        continue_message: Continue message prompt to use
        model_name: Model to use for evaluation
        task_name: Task name (default: "sqa_dev")
        tool_config_dict: Tool configuration as dict (default: search_tools + report_editor)
        timeout: Timeout in seconds (default: 600)

    Returns:
        Score (global_avg) for this sample

    Raises:
        TimeoutError: If evaluation exceeds timeout
        RuntimeError: If evaluation fails
    """
    if tool_config_dict is None:
        tool_config_dict = {
            "with_search_tools": True,
            "with_report_editor": True,
        }

    # Use Pool.apply() to run worker in subprocess
    # This properly handles return values without needing Queue
    with mp.Pool(processes=1) as pool:
        try:
            logger.info(f"Starting subprocess for sample {sample_id}")
            async_result = pool.apply_async(
                _run_eval_worker,
                args=((sample_id, system_message, continue_message, model_name, task_name, tool_config_dict),),
            )
            # Wait for result with timeout
            result_data = async_result.get(timeout=timeout)
            logger.info(f"Subprocess completed for sample {sample_id}")

        except mp.TimeoutError:
            pool.terminate()
            pool.join()
            raise TimeoutError(
                f"Evaluation of sample {sample_id} timed out after {timeout}s"
            )
        except Exception as e:
            pool.terminate()
            pool.join()
            raise RuntimeError(f"Subprocess failed for sample {sample_id}: {e}")

    # Parse result
    status, *result = result_data

    if status == "error":
        raise RuntimeError(f"Evaluation failed for sample {sample_id}: {result[0]}")

    # Unpack success result: (score, sample_id, scorer_name)
    score_value = result[0]
    return score_value

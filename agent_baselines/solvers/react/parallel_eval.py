"""Subprocess-based parallel evaluation for inspect_ai.

This module provides a subprocess wrapper around inspect_ai.eval() to enable
parallel evaluation when using DSPy optimizers. Each eval runs in a separate
subprocess to avoid inspect_ai's concurrent call restriction.
"""

import logging
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def redirect_output(std_log_file: str | None):
    """Context manager to redirect stdout/stderr to a log file at the FD level.

    Uses os.dup2() to redirect at the file descriptor level, which ensures that
    subprocess-spawned processes (like Docker builds) also have their output
    redirected, not just Python print statements.

    Args:
        std_log_file: Path to log file, or None to keep default output
    """
    if std_log_file is None:
        # No redirection needed
        yield
        return

    # Create log directory if needed
    log_dir = os.path.dirname(std_log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Save original file descriptors by duplicating them
    old_stdout_fd = os.dup(sys.stdout.fileno())
    old_stderr_fd = os.dup(sys.stderr.fileno())

    # Open log file for writing
    log_file = None
    try:
        log_file = open(std_log_file, "w")

        # Redirect at file descriptor level (affects all subprocesses)
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())

        yield

    finally:
        # Flush any remaining output
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore original file descriptors
        os.dup2(old_stdout_fd, sys.stdout.fileno())
        os.dup2(old_stderr_fd, sys.stderr.fileno())

        # Close saved file descriptors
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

        # Close log file
        if log_file is not None:
            log_file.close()


def _do_evaluation(
    sample_id: str,
    system_message: str,
    continue_message: str,
    model_names: list[str],
    task_path: str,
    primary_metric: str,
) -> tuple[str, float, str, str]:
    """Perform the actual evaluation work.

    Args:
        sample_id: ID of the sample to evaluate
        system_message: System message prompt
        continue_message: Continue message prompt
        model_names: List of models to evaluate on
        task_path: Task path (e.g., "astabench/sqa_dev")
        primary_metric: Primary metric to extract (format: "scorer_name/metric_name")

    Returns:
        Tuple of ("success", avg_score_value, sample_id, task_path)

    Raises:
        RuntimeError: If all models fail (systemic issue)
    """
    print(
        f"Starting eval worker for sample {sample_id} on task {task_path}",
        file=sys.stderr,
    )
    # Import inside worker to ensure fresh state
    from inspect_ai import eval as inspect_eval

    from agent_baselines.solvers.react.dspy_agent import (
        create_agent_with_dspy_prompts,
    )

    # Parse primary_metric format: "scorer_name/metric_name"
    # E.g., "global_avg/mean" or "score_discoverybench/mean"
    metric_parts = primary_metric.split("/")
    assert len(metric_parts) == 2, f"Invalid primary_metric format: {primary_metric}"
    scorer_name, metric_name = metric_parts

    # Create solver with candidate prompts (without tools - task provides them)
    agent_solver = create_agent_with_dspy_prompts(
        system_message_text=system_message,
        continue_message_text=continue_message,
        max_steps=10,
    )

    # Evaluate on all models and collect scores
    scores = []
    for model_name in model_names:
        print(f"Evaluating sample {sample_id} on model {model_name}", file=sys.stderr)

        # Run eval on this sample with this model
        # inspect_eval can load tasks by path string (e.g., "astabench/sqa_dev")
        logs = inspect_eval(
            tasks=task_path,  # Pass task path as string
            model=model_name,
            solver=agent_solver,
            sample_id=sample_id,
            log_dir=".dspy_cache",
            log_level="warning",
            display="plain",
        )

        # Extract score from results
        assert logs and len(logs) > 0, f"No logs returned for sample {sample_id}"
        eval_log = logs[0]

        # Handle evaluation failures gracefully
        if not eval_log.results or not eval_log.results.scores:
            error_msg = (
                f"Evaluation failed for sample {sample_id} on model {model_name}"
            )
            if eval_log.error:
                error_msg += f": {eval_log.error}"
            print(f"WARNING: {error_msg}", file=sys.stderr)
            print(f"Treating as score 0.0 for this model", file=sys.stderr)
            scores.append(0.0)
            continue

        # Find the scorer with matching name
        scorer = None
        for score in eval_log.results.scores:
            if score.name == scorer_name:
                scorer = score
                break

        if not scorer:
            print(
                f"WARNING: Scorer '{scorer_name}' not found in results for {sample_id}. "
                f"Available: {[s.name for s in eval_log.results.scores]}. Treating as 0.0.",
                file=sys.stderr,
            )
            scores.append(0.0)
            continue

        if metric_name not in scorer.metrics:
            print(
                f"WARNING: Metric '{metric_name}' not found in scorer '{scorer_name}' for {sample_id}. "
                f"Available: {list(scorer.metrics.keys())}. Treating as 0.0.",
                file=sys.stderr,
            )
            scores.append(0.0)
            continue

        score_value = float(scorer.metrics[metric_name].value)
        scores.append(score_value)
        print(f"Model {model_name} score: {score_value:.4f}", file=sys.stderr)

    # Average scores across models
    if not scores:
        error_msg = (
            f"All {len(model_names)} models failed for sample {sample_id} on task {task_path}. "
            f"This likely indicates a systemic issue with the prompts or sample."
        )
        print(f"ERROR: {error_msg}", file=sys.stderr)
        raise RuntimeError(error_msg)

    avg_score = sum(scores) / len(scores)
    print(f"Sample {sample_id} average score: {avg_score:.4f}", file=sys.stderr)

    return ("success", avg_score, sample_id, task_path)


def _run_eval_worker(args):
    """Worker function that runs in a subprocess via Pool.map.

    Args:
        args: Tuple of (sample_id, system_message, continue_message, model_names,
              task_path, primary_metric, std_log_file)

    Returns:
        Tuple of ("success", avg_score_value, sample_id, task_path) or
        ("error", error_message)
    """
    (
        sample_id,
        system_message,
        continue_message,
        model_names,
        task_path,
        primary_metric,
        std_log_file,
    ) = args

    with redirect_output(std_log_file):
        try:
            return _do_evaluation(
                sample_id=sample_id,
                system_message=system_message,
                continue_message=continue_message,
                model_names=model_names,
                task_path=task_path,
                primary_metric=primary_metric,
            )
        except Exception as e:
            # Return exception info
            import traceback

            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(
                f"Exception in eval worker for sample {sample_id}: {error_msg}",
                file=sys.stderr,
            )
            return ("error", error_msg)


def eval_in_subprocess(
    sample_id: str,
    system_message: str,
    continue_message: str,
    model_names: list[str],
    task_path: str,
    primary_metric: str,
    timeout: int = 600,
    std_log_file: str | None = None,
) -> float:
    """Run inspect_ai.eval() in an isolated subprocess with multi-model support.

    Args:
        sample_id: ID of the sample to evaluate
        system_message: System message prompt to use
        continue_message: Continue message prompt to use
        model_names: List of models to evaluate on (scores will be averaged)
        task_path: Task path (e.g., "astabench/sqa_dev")
        primary_metric: Primary metric to extract (e.g., "global_avg/mean")
        timeout: Timeout in seconds (default: 600)
        std_log_file: Optional path to redirect subprocess output (default: None)

    Returns:
        Average score across all models for this sample

    Raises:
        TimeoutError: If evaluation exceeds timeout
        RuntimeError: If evaluation fails
    """
    # Use Pool.apply() to run worker in subprocess
    # This properly handles return values without needing Queue
    import time

    time.time()

    with mp.Pool(processes=1) as pool:
        try:
            logger.info(
                f"Starting subprocess for sample {sample_id} on task {task_path}"
            )
            subprocess_start = time.time()

            async_result = pool.apply_async(
                _run_eval_worker,
                args=(
                    (
                        sample_id,
                        system_message,
                        continue_message,
                        model_names,
                        task_path,
                        primary_metric,
                        std_log_file,
                    ),
                ),
            )
            # Wait for result with timeout
            result_data = async_result.get(timeout=timeout)

            actual_duration = time.time() - subprocess_start
            logger.info(
                f"Subprocess completed for sample {sample_id} (took {actual_duration:.1f}s)"
            )

        except mp.TimeoutError:
            actual_duration = time.time() - subprocess_start
            pool.terminate()
            pool.join()
            raise TimeoutError(
                f"Evaluation of sample {sample_id} timed out after {actual_duration:.0f}s "
                f"(timeout setting: {timeout}s)"
            )
        except Exception as e:
            actual_duration = time.time() - subprocess_start
            pool.terminate()
            pool.join()
            raise RuntimeError(
                f"Subprocess failed for sample {sample_id} after {actual_duration:.1f}s: {e}"
            )

    # Parse result
    status, *result = result_data

    if status == "error":
        raise RuntimeError(f"Evaluation failed for sample {sample_id}: {result[0]}")

    # Unpack success result: (avg_score, sample_id, task_path)
    score_value = result[0]
    return score_value

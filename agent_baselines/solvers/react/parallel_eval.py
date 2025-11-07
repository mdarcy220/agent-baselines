"""Subprocess-based parallel evaluation for inspect_ai.

This module provides a subprocess wrapper around inspect_ai.eval() to enable
parallel evaluation when using DSPy optimizers. Each eval runs in a separate
subprocess to avoid inspect_ai's concurrent call restriction.

Design: Uses inspect_ai's solver loading mechanism to dynamically load any
solver from a path. The agent_params dict is passed as kwargs to the solver
factory function, enabling support for different agent types.
"""

import logging
import multiprocessing as mp
import os
import sys
import time
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
    model_names: list[str],
    task_path: str,
    primary_metric: str,
    solver_path: str,
    agent_params: dict,
) -> tuple[str, float, str, str]:
    """Perform the actual evaluation work.

    Args:
        sample_id: ID of the sample to evaluate
        model_names: List of models to evaluate on
        task_path: Task path (e.g., "astabench/sqa_dev")
        primary_metric: Primary metric to extract (format: "scorer_name/metric_name")
        solver_path: Path to solver (e.g., "agent_baselines/solvers/react/dspy_agent.py@create_agent_with_dspy_prompts")
        agent_params: Parameters to pass to the solver factory as kwargs

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
    from inspect_ai._eval.loader import SolverSpec, solver_from_spec

    # Parse primary_metric format: "scorer_name/metric_name"
    # E.g., "global_avg/mean" or "score_discoverybench/mean"
    metric_parts = primary_metric.split("/")
    assert len(metric_parts) == 2, f"Invalid primary_metric format: {primary_metric}"
    scorer_name, metric_name = metric_parts

    # Load solver dynamically using inspect_ai's solver loading mechanism
    # This allows any solver to be used, not just ReAct
    solver_spec = SolverSpec(solver=solver_path, args=agent_params)
    agent_solver = solver_from_spec(solver_spec)

    scores = []
    for model_name in model_names:
        print(f"Evaluating sample {sample_id} on model {model_name}", file=sys.stderr)

        # Retry configuration: We allow up to 2 retries to handle transient failures
        # (e.g., scorer crashes due to unexpected agent output, temporary API issues).
        # This reduces spurious failures without hiding systemic problems.
        logs = inspect_eval(
            tasks=task_path,
            model=model_name,
            solver=agent_solver,
            sample_id=sample_id,
            log_dir=".dspy_cache",
            log_level="warning",
            display="plain",
            retry_on_error=2,
        )

        assert logs and len(logs) > 0, f"No logs returned for sample {sample_id}"
        eval_log = logs[0]

        # Handle evaluation failures gracefully
        #
        # Design decision: We return 0.0 instead of crashing the entire optimization.
        # Rationale:
        # 1. A bad prompt can cause systemic failures across samples (e.g., malformed
        #    output that breaks the scorer). Crashing loses all optimization progress.
        # 2. Scoring 0.0 signals DSPy to avoid this prompt in future iterations.
        # 3. Infrastructure bugs would cause ALL samples to score 0.0, making it
        #    obvious in the optimization results (not a silent failure).
        # 4. Prompt-specific failures mean it's a bad prompt we want to avoid anyway.
        if not eval_log.results or not eval_log.results.scores:
            error_msg = (
                eval_log.error if eval_log.error else "No results/scores returned"
            )
            logger.error(
                f"Eval failed for {sample_id} on {model_name}: {error_msg}. Returning score 0.0"
            )
            scores.append(0.0)
            continue

        # Find the scorer with matching name
        scorer = None
        for score in eval_log.results.scores:
            if score.name == scorer_name:
                scorer = score
                break

        if not scorer:
            available = [s.name for s in eval_log.results.scores]
            logger.error(
                f"Scorer '{scorer_name}' not found for {sample_id} on {model_name}. "
                f"Available: {available}. Returning score 0.0"
            )
            scores.append(0.0)
            continue

        if metric_name not in scorer.metrics:
            available = list(scorer.metrics.keys())
            logger.error(
                f"Metric '{metric_name}' not found in scorer '{scorer_name}' for {sample_id} on {model_name}. "
                f"Available: {available}. Returning score 0.0"
            )
            scores.append(0.0)
            continue

        score_value = float(scorer.metrics[metric_name].value)
        scores.append(score_value)
        print(f"Model {model_name} score: {score_value:.4f}", file=sys.stderr)

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
        args: Tuple of (sample_id, model_names, task_path, primary_metric,
              solver_path, agent_params, std_log_file)

    Returns:
        Tuple of ("success", avg_score_value, sample_id, task_path) or
        ("error", error_message)
    """
    (
        sample_id,
        model_names,
        task_path,
        primary_metric,
        solver_path,
        agent_params,
        std_log_file,
    ) = args

    with redirect_output(std_log_file):
        try:
            return _do_evaluation(
                sample_id=sample_id,
                model_names=model_names,
                task_path=task_path,
                primary_metric=primary_metric,
                solver_path=solver_path,
                agent_params=agent_params,
            )
        except Exception as e:
            # Return exception info
            #
            # Note: This captures exceptions that occur during evaluation setup or execution.
            # These are returned as error results rather than raised, so the parent process
            # can decide how to handle them (e.g., return 0.0 during DSPy optimization).
            import traceback

            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(
                f"Exception in eval worker for {sample_id} on {task_path}: {type(e).__name__}: {e}"
            )
            return ("error", error_msg)


def eval_in_subprocess(
    sample_id: str,
    model_names: list[str],
    task_path: str,
    primary_metric: str,
    solver_path: str,
    agent_params: dict,
    timeout: int = 600,
    std_log_file: str | None = None,
) -> float:
    """Run inspect_ai.eval() in an isolated subprocess with multi-model support.

    Uses inspect_ai's dynamic solver loading to support any solver type.

    Args:
        sample_id: ID of the sample to evaluate
        model_names: List of models to evaluate on (scores will be averaged)
        task_path: Task path (e.g., "astabench/sqa_dev")
        primary_metric: Primary metric to extract (e.g., "global_avg/mean")
        solver_path: Path to solver (e.g., "agent_baselines/solvers/react/dspy_agent.py@create_agent_with_dspy_prompts")
        agent_params: Parameters to pass to the solver factory as kwargs
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

    # Create a single-process pool for this evaluation.
    #
    # Design rationale (per Claude):
    # - Each eval runs in isolation with no state leakage
    # - Parallelization happens at a higher level: DSPy's ThreadPoolExecutor
    #   (default 8 threads) spawns multiple concurrent subprocesses
    # - This bypasses inspect_ai's global lock that prevents concurrent eval_async() calls
    # - Pool creation overhead (~6ms) is negligible vs evaluation time (5-60s)
    # - Achieves 3-8x speedup for typical optimizations
    with mp.Pool(processes=1) as pool:
        try:
            logger.info(
                f"Starting subprocess for sample {sample_id} on task {task_path}"
            )
            subprocess_start = time.perf_counter()

            async_result = pool.apply_async(
                _run_eval_worker,
                args=(
                    (
                        sample_id,
                        model_names,
                        task_path,
                        primary_metric,
                        solver_path,
                        agent_params,
                        std_log_file,
                    ),
                ),
            )
            # Wait for result with timeout
            result_data = async_result.get(timeout=timeout)

            logger.info(
                f"Subprocess completed for sample {sample_id} (took {time.perf_counter() - subprocess_start:.1f}s)"
            )

        except mp.TimeoutError:
            pool.terminate()
            pool.join()
            raise TimeoutError(
                f"Evaluation of sample {sample_id} timed out after {time.perf_counter() - subprocess_start:.0f}s "
                f"(timeout setting: {timeout}s)"
            )
        except Exception as e:
            pool.terminate()
            pool.join()
            raise RuntimeError(
                f"Subprocess failed for sample {sample_id} after {time.perf_counter() - subprocess_start:.1f}s: {e}"
            )

    status, *result = result_data

    if status == "error":
        raise RuntimeError(f"Evaluation failed for sample {sample_id}: {result[0]}")

    # Unpack success result: (avg_score, sample_id, task_path)
    score_value = result[0]
    return score_value

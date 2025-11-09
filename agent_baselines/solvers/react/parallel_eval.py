"""Subprocess-based parallel evaluation for inspect_ai.

This module provides a subprocess wrapper around inspect_ai.eval() to enable
parallel evaluation when using DSPy optimizers. Each eval runs in a separate
subprocess to avoid inspect_ai's concurrent call restriction.

Design: Uses inspect_ai's solver loading mechanism to dynamically load any
solver from a path. The agent_params dict is passed as kwargs to the solver
factory function, enabling support for different agent types.
"""

import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Result from evaluating a single model."""

    eval_path: str
    score: float | None  # None indicates infrastructure failure
    answer: str | None

    def float_score(self) -> float:
        """Convert score to float, using 0.0 for infrastructure failures."""
        return 0.0 if self.score is None else self.score


@dataclass
class SummaryFile:
    """Multi-model evaluation summary.

    This is the schema for summary JSON files created during optimization.
    Each summary contains results from evaluating multiple models on a single sample.
    """

    sample_id: str
    task_path: str
    primary_metric: str
    timestamp: str
    model_names: list[str]  # Preserve original order
    models: dict[str, ModelResult]  # model_name -> result
    average_score: float


@dataclass
class EvalSuccess:
    """Successful evaluation result."""

    score: float
    sample_id: str
    task_path: str
    summary_path: str


@dataclass
class EvalError:
    """Failed evaluation result."""

    error_message: str


# Union type for evaluation results
EvalResult = EvalSuccess | EvalError


def load_summary_file(summary_path: str) -> SummaryFile:
    """Load and parse a summary JSON file.

    Args:
        summary_path: Path to the summary JSON file

    Returns:
        Parsed SummaryFile object

    Raises:
        FileNotFoundError: If summary file doesn't exist
        KeyError: If required fields are missing from summary
        ValueError: If summary format is invalid
    """

    with open(summary_path, "r") as f:
        data = json.load(f)

    models = {}
    for model_name, model_data in data["models"].items():
        models[model_name] = ModelResult(
            eval_path=model_data["eval_path"],
            score=model_data["score"],
            answer=model_data["answer"],
        )

    return SummaryFile(
        sample_id=data["sample_id"],
        task_path=data["task_path"],
        primary_metric=data["primary_metric"],
        timestamp=data["timestamp"],
        model_names=data["model_names"],
        models=models,
        average_score=data["average_score"],
    )


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
        yield
        return

    log_dir = os.path.dirname(std_log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Must save FDs before redirection to restore them later
    old_stdout_fd = os.dup(sys.stdout.fileno())
    old_stderr_fd = os.dup(sys.stderr.fileno())

    log_file = None
    try:
        log_file = open(std_log_file, "w")

        # Use FD-level redirection so subprocesses (like Docker) also get redirected
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())

        yield

    finally:
        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(old_stdout_fd, sys.stdout.fileno())
        os.dup2(old_stderr_fd, sys.stderr.fileno())

        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

        if log_file is not None:
            log_file.close()


def _do_evaluation(
    sample_id: str,
    model_names: list[str],
    task_path: str,
    primary_metric: str,
    solver_path: str,
    agent_params: dict,
    inspect_log_dir: str,
) -> EvalSuccess:
    """Perform the actual evaluation work using parallel multi-model evaluation.

    Args:
        sample_id: ID of the sample to evaluate
        model_names: List of models to evaluate on (runs in parallel)
        task_path: Task path (e.g., "astabench/sqa_dev")
        primary_metric: Primary metric to extract (format: "scorer_name/metric_name")
        solver_path: Path to solver
        agent_params: Parameters to pass to the solver factory as kwargs
        inspect_log_dir: Directory for inspect_ai evaluation logs

    Returns:
        EvalSuccess with score, sample_id, task_path, and summary_path

    Raises:
        RuntimeError: If all models fail (systemic issue)
    """

    print(
        f"Starting parallel eval for sample {sample_id} on {len(model_names)} models",
        file=sys.stderr,
    )

    # Import inside worker to ensure fresh state (avoid leaking across subprocesses)
    from inspect_ai import eval as inspect_eval
    from inspect_ai._eval.loader import SolverSpec, solver_from_spec

    metric_parts = primary_metric.split("/")
    assert len(metric_parts) == 2, f"Invalid primary_metric format: {primary_metric}"
    scorer_name, metric_name = metric_parts

    solver_spec = SolverSpec(solver=solver_path, args=agent_params)
    agent_solver = solver_from_spec(solver_spec)

    # inspect_ai runs multiple models in parallel when passed a list
    print(f"Running parallel evaluation on models: {model_names}", file=sys.stderr)
    logs = inspect_eval(
        tasks=task_path,
        model=model_names,
        solver=agent_solver,
        sample_id=sample_id,
        log_dir=inspect_log_dir,
        log_level="warning",
        display="plain",
        retry_on_error=2,
    )

    assert logs and len(logs) > 0, f"No logs returned for sample {sample_id}"
    assert len(logs) == len(
        model_names
    ), f"Expected {len(model_names)} logs, got {len(logs)}"

    model_results = {}

    for eval_log in logs:
        model_name = eval_log.eval.model

        score_value = None
        answer = None
        eval_path = eval_log.location or ""

        if not eval_log.results or not eval_log.results.scores:
            error_msg = eval_log.error if eval_log.error else "No results/scores"
            logger.error(f"Eval failed for {sample_id} on {model_name}: {error_msg}")
        else:
            scorer = None
            for score in eval_log.results.scores:
                if score.name == scorer_name:
                    scorer = score
                    break

            if not scorer:
                available = [s.name for s in eval_log.results.scores]
                logger.error(
                    f"Scorer '{scorer_name}' not found for {sample_id} on {model_name}. "
                    f"Available: {available}"
                )
            elif metric_name not in scorer.metrics:
                available = list(scorer.metrics.keys())
                logger.error(
                    f"Metric '{metric_name}' not found in scorer '{scorer_name}' for {sample_id} on {model_name}. "
                    f"Available: {available}"
                )
            else:
                score_value = float(scorer.metrics[metric_name].value)

            if eval_log.samples and len(eval_log.samples) > 0:
                sample = eval_log.samples[0]
                if sample.output and sample.output.completion:
                    answer = sample.output.completion

        model_results[model_name] = ModelResult(
            eval_path=eval_path,
            score=score_value,
            answer=answer,
        )

        score_str = f"{score_value:.4f}" if score_value is not None else "FAILED"
        print(f"Model {model_name} score: {score_str}", file=sys.stderr)

    had_legitimate_scores = any(r.score is not None for r in model_results.values())
    if not had_legitimate_scores:
        error_msg = (
            f"All {len(model_names)} models failed for sample {sample_id} on task {task_path}. "
            f"This likely indicates a systemic issue with the evaluation infrastructure."
        )
        print(f"ERROR: {error_msg}", file=sys.stderr)
        raise RuntimeError(error_msg)

    scores = [r.float_score() for r in model_results.values()]
    avg_score = sum(scores) / len(scores)
    print(f"Sample {sample_id} average score: {avg_score:.4f}", file=sys.stderr)

    summary = SummaryFile(
        sample_id=sample_id,
        task_path=task_path,
        primary_metric=primary_metric,
        timestamp=datetime.now().isoformat(),
        model_names=model_names,
        models=model_results,
        average_score=avg_score,
    )

    # Convert sample_id to string first (some datasets use int IDs like DS-1000's `822`)
    summary_filename = (
        f"{str(sample_id).replace('|', '_').replace('/', '_')}_summary.json"
    )
    summary_path = Path(inspect_log_dir) / summary_filename
    with open(summary_path, "w") as f:
        # asdict() recursively converts nested dataclasses (ModelResult)
        json.dump(asdict(summary), f, indent=2)

    print(f"Wrote summary to: {summary_path}", file=sys.stderr)

    return EvalSuccess(
        score=avg_score,
        sample_id=sample_id,
        task_path=task_path,
        summary_path=str(summary_path),
    )


def _run_eval_worker(args) -> EvalResult:
    """Worker function that runs in a subprocess via Pool.map.

    Args:
        args: Tuple of (sample_id, model_names, task_path, primary_metric,
              solver_path, agent_params, inspect_log_dir, std_log_file)

    Returns:
        EvalSuccess on success or EvalError on failure
    """
    (
        sample_id,
        model_names,
        task_path,
        primary_metric,
        solver_path,
        agent_params,
        inspect_log_dir,
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
                inspect_log_dir=inspect_log_dir,
            )
        except Exception as e:
            # Return exception info
            #
            # Note: This captures exceptions that occur during evaluation setup or execution.
            # These are returned as error results rather than raised, so the parent process
            # can decide how to handle them (e.g., return 0.0 during DSPy optimization).
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.exception(
                f"Exception in eval worker for {sample_id} on {task_path}: {type(e).__name__}:"
            )
            return EvalError(error_message=error_msg)


def eval_in_subprocess(
    sample_id: str,
    model_names: list[str],
    task_path: str,
    primary_metric: str,
    solver_path: str,
    agent_params: dict,
    inspect_log_dir: str,
    timeout: int = 600,
) -> tuple[float, str]:
    """Run inspect_ai.eval() in an isolated subprocess with multi-model support.

    Uses inspect_ai's dynamic solver loading to support any solver type.

    Args:
        sample_id: ID of the sample to evaluate
        model_names: List of models to evaluate on (scores will be averaged)
        task_path: Task path (e.g., "astabench/sqa_dev")
        primary_metric: Primary metric to extract (e.g., "global_avg/mean")
        solver_path: Path to solver (e.g., "agent_baselines/solvers/react/dspy_agent.py@create_agent_with_dspy_prompts")
        agent_params: Parameters to pass to the solver factory as kwargs
        inspect_log_dir: Directory for logs
        timeout: Timeout in seconds (default: 600)

    Returns:
        Tuple of (average_score, eval_path) where:
        - average_score: Average score across all models for this sample
        - eval_path: Path to the .eval file created (empty string if not found)

    Raises:
        TimeoutError: If evaluation exceeds timeout
        RuntimeError: If evaluation fails
    """
    os.makedirs(inspect_log_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    std_log_file = f"{inspect_log_dir}/{sample_id}_{timestamp}_stdout.log"

    # Use single-process pool for subprocess isolation (no state leakage)
    # Parallelization happens at higher level: DSPy's ThreadPoolExecutor spawns
    # multiple concurrent subprocesses, bypassing inspect_ai's global lock
    #
    # Use 'spawn' instead of 'fork' to avoid bugs when mixing multiprocessing
    # with threading (DSPy's ThreadPoolExecutor + tqdm progress bars).
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
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
                        inspect_log_dir,
                        std_log_file,
                    ),
                ),
            )
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

    if isinstance(result_data, EvalError):
        raise RuntimeError(
            f"Evaluation failed for sample {sample_id}: {result_data.error_message}"
        )

    if isinstance(result_data, EvalSuccess):
        return result_data.score, result_data.summary_path

    # Fail loudly if we get an unexpected type
    raise TypeError(
        f"Unexpected result type from worker: {type(result_data)}. "
        f"Expected EvalSuccess or EvalError."
    )

"""Simplified DSPy optimizer for ReAct agent prompts.

Clean architecture with clear separation of concerns:
1. Task Data - Load and prepare samples from multiple tasks
2. Agent Wrapper - Encapsulate tunable params, fixed params, and metric
3. Optimizer - Configure DSPy optimizer (MIPRO, GEPA, Bootstrap)
4. Logging - Optional infrastructure for debugging
5. CLI - Wire everything together

Key design principles:
- Import data loading utilities from optimize.py (don't duplicate)
- Use parallel_eval.py for subprocess-based evaluation
- Keep metric function pure (no filesystem I/O)
- Separate logging concerns from evaluation logic
- Make it easy to extend to other agents beyond ReAct
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import dspy
import numpy as np
from dspy.utils.callback import BaseCallback

from agent_baselines.solvers.react.basic_agent import DEFAULT_SUBMIT_NAME
from agent_baselines.solvers.react.dspy_agent import DSPyReActPrompts
from agent_baselines.solvers.react.optimize import (
    create_mixed_dspy_examples,
    interleave_tasks,
    load_samples_from_tasks,
    load_tasks_from_config,
)
from agent_baselines.solvers.react.parallel_eval import eval_in_subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Generic task description for universal prompts (multi-task optimization)
GENERIC_TASK_DESCRIPTION = """You will be given a task to complete. Use the available tools to help you solve the task, doing reasoning before each action to explain your approach."""

# Target samples per task for MIPRO minibatch sizing
SAMPLES_PER_TASK_FOR_MINIBATCH = 5


# ============================================================================
# Agent Wrapper Abstraction
# ============================================================================


@dataclass
class TaskLoadingConfig:
    """Configuration for loading task data."""

    config_path: str | None
    task_split: str | None
    tasks: list[str] | None
    samples_per_task: int
    train_ratio: float


@dataclass
class OptimizerConfig:
    """Configuration for DSPy optimizer."""

    optimizer_type: str
    optimizer_model: str | None
    num_candidates: int
    num_trials: int | None
    max_bootstrapped_demos: int
    max_labeled_demos: int
    temperature: float


@dataclass
class RunConfig:
    """Runtime configuration."""

    run_dir: str | None
    output_file: str
    verbose_llm: bool


@dataclass
class ReactAgentConfig:
    """Encapsulates configuration for ReAct agent wrapper.

    This separates tunable parameters (what DSPy optimizes) from fixed
    parameters (what stays constant during optimization).
    """

    # Fixed parameters (not optimized)
    eval_models: list[str]  # Models to evaluate prompts on
    agent_kwargs: dict  # Agent config (e.g., {"max_steps": 10})
    eval_timeout: int  # Timeout per sample evaluation

    # Metadata for logging (not used during eval)
    log_dir: str | None = None


class ReactInspectAgent:
    """Agent wrapper that bridges DSPy and inspect_ai.

    This abstraction encapsulates:
    1. Tunable parameters - what DSPy optimizes (system/continue messages)
    2. Fixed parameters - what stays constant (models, agent config)
    3. Metric function - how to score a configuration

    Design: The metric function is pure evaluation logic. Logging is handled
    separately via optional wrapper to maintain separation of concerns.
    """

    def __init__(self, config: ReactAgentConfig):
        """Initialize agent wrapper with fixed configuration.

        Args:
            config: Fixed agent configuration (models, timeouts, etc.)
        """
        self.config = config

    def create_metric(self) -> Callable:
        """Create metric function for DSPy optimization.

        Returns a pure evaluation function that:
        1. Extracts prompts from DSPy prediction
        2. Extracts metadata from DSPy example
        3. Formulates agent_params dict with all parameters for the ReAct agent
        4. Runs eval_in_subprocess with those inputs
        5. Returns averaged score across models

        The metric has no side effects (no logging, no I/O) - it's just
        evaluation logic. This makes it testable and reusable.

        Design principle: The agent wrapper is responsible for knowing what
        parameters its specific agent type needs and packaging them into the
        agent_params dict. This makes it easy to add new agent types later.
        """

        def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
            """Pure metric function for DSPy optimization.

            Args:
                example: DSPy Example with sample_id, task_path, primary_metric
                prediction: DSPy Prediction with system_message, continue_message
                trace: Optional trace (unused)
                pred_name: Optional predictor name (DSPy internal)
                pred_trace: Optional predictor trace (DSPy internal)

            Returns:
                Score averaged across all evaluation models
            """
            system_message = prediction.system_message
            continue_message = prediction.continue_message

            sample_id = example.sample_id
            task_path = example.task_path
            primary_metric = example.primary_metric

            agent_params = {
                "system_message": system_message,
                "continue_message": continue_message,
                **self.config.agent_kwargs,
            }

            std_log_file = None
            if self.config.log_dir:
                os.makedirs(self.config.log_dir, exist_ok=True)
                timestamp = int(time.time() * 1000)
                std_log_file = f"{self.config.log_dir}/{sample_id}_{timestamp}.log"

            # Run evaluation in subprocess (enables parallelization)
            # This handles multi-model evaluation and returns averaged score
            score_value = eval_in_subprocess(
                sample_id=sample_id,
                model_names=self.config.eval_models,
                task_path=task_path,
                primary_metric=primary_metric,
                agent_params=agent_params,
                timeout=self.config.eval_timeout,
                std_log_file=std_log_file,
            )

            return score_value

        return metric

    def create_logging_metric(self) -> Callable:
        """Create metric function with logging wrapper.

        This wraps the pure metric with logging behavior for user visibility.
        Only used when log_dir is configured.

        Returns:
            Metric function that logs progress and results
        """
        base_metric = self.create_metric()

        def logging_metric(
            example, prediction, trace=None, pred_name=None, pred_trace=None
        ):
            """Metric with logging for visibility during optimization."""
            sample_id = example.sample_id
            task_name = example.task_name

            try:
                # Call base metric for evaluation
                score_value = base_metric(
                    example, prediction, trace, pred_name, pred_trace
                )

                # Log result concisely
                logger.info(f"✓ {sample_id} ({task_name}): {score_value:.4f}")
                return score_value

            except Exception as e:
                # Log failure
                logger.info(f"✗ {sample_id} ({task_name}): {e}")
                raise

        return logging_metric

    def get_tunable_module(self) -> DSPyReActPrompts:
        """Get the DSPy module with tunable parameters.

        For ReAct agent, this is the DSPyReActPrompts module with
        system_message_generator and continue_message_generator.
        """
        return DSPyReActPrompts()


# ============================================================================
# LLM Logging Infrastructure (Optional)
# ============================================================================


class LLMCallLogger(BaseCallback):
    """Callback to log all LLM calls during DSPy optimization.

    This is optional infrastructure for debugging optimizer behavior.
    Separate from evaluation logic to maintain separation of concerns.
    """

    def __init__(self, log_file: str):
        """Initialize logger with output file path.

        Args:
            log_file: Path to file where LLM calls will be logged
        """
        self.log_file = log_file
        self.call_count = 0

        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        with open(log_file, "w") as f:
            f.write(f"DSPy LLM Call Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def on_lm_start(self, call_id, instance, inputs):
        """Log LLM call inputs."""
        self.call_count += 1

        with open(self.log_file, "a") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(
                f"LLM CALL #{self.call_count} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Model: {instance.model}\n")
            f.write(f"Call ID: {call_id}\n")
            f.write(f"{'=' * 80}\n\n")

            # Log messages or prompt
            if inputs.get("messages"):
                f.write("MESSAGES:\n")
                for msg in inputs["messages"]:
                    f.write(f"\n[{msg['role'].upper()}]\n")
                    f.write(f"{msg['content']}\n")
            elif inputs.get("prompt"):
                f.write("PROMPT:\n")
                f.write(f"{inputs['prompt']}\n")

            # Log other parameters
            other_params = {
                k: v for k, v in inputs.items() if k not in ["prompt", "messages"]
            }
            if other_params:
                f.write(f"\nPARAMETERS:\n{json.dumps(other_params, indent=2)}\n")

    def on_lm_end(self, call_id, outputs, exception):
        """Log LLM call outputs."""
        with open(self.log_file, "a") as f:
            f.write(f"\nOUTPUT (Call ID: {call_id}):\n")

            if exception:
                f.write(f"ERROR: {exception}\n")
            else:
                for output in outputs:
                    if isinstance(output, dict):
                        f.write(f"{output.get('text', str(output))}\n")
                    else:
                        f.write(f"{output}\n")

            f.write(f"\n{'=' * 80}\n\n")


def setup_dspy_with_logging(
    optimizer_model: str,
    run_dir: str,
    verbose_llm: bool,
) -> None:
    """Configure DSPy language model with optional logging.

    Args:
        optimizer_model: Model to use for DSPy optimization
        run_dir: Run directory for logs
        verbose_llm: Whether to enable LLM call logging
    """
    lm = dspy.LM(model=optimizer_model)

    if verbose_llm:
        llm_log_file = f"{run_dir}/llm_calls.log"
        llm_logger = LLMCallLogger(llm_log_file)
        dspy.settings.configure(lm=lm, callbacks=[llm_logger])
        logger.info(f"LLM call logging enabled: {llm_log_file}")
    else:
        dspy.settings.configure(lm=lm)


# ============================================================================
# Optimization Pipeline Functions
# ============================================================================


def setup_optimization_run(
    optimizer_config: OptimizerConfig,
    agent_config: ReactAgentConfig,
    run_config: RunConfig,
) -> str:
    """Setup run directory and configure DSPy.

    Returns:
        run_dir: Path to the run directory
    """
    run_dir = run_config.run_dir
    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = f".dspy_cache/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    logger.info(f"Run directory: {run_dir}")

    optimizer_model = optimizer_config.optimizer_model or agent_config.eval_models[0]
    setup_dspy_with_logging(optimizer_model, run_dir, run_config.verbose_llm)

    logger.info(f"Eval models: {', '.join(agent_config.eval_models)}")
    logger.info(f"Optimizer model: {optimizer_model}")

    return run_dir


def load_and_prepare_data(
    task_config: TaskLoadingConfig,
) -> tuple[list, list, list]:
    """Load task data and create train/val splits.

    Returns:
        Tuple of (train_examples, val_examples, task_configs)
    """
    task_configs = load_tasks_from_config(
        config_path=task_config.config_path,
        split=task_config.task_split,
        task_paths=task_config.tasks,
    )

    logger.info(
        f"Loading {len(task_configs)} tasks: {', '.join(tc.name for tc in task_configs)}"
    )

    sample_tuples = load_samples_from_tasks(
        task_configs=task_configs,
        samples_per_task=task_config.samples_per_task,
    )

    train_examples, val_examples = create_mixed_dspy_examples(
        sample_tuples=sample_tuples,
        train_ratio=task_config.train_ratio,
    )
    logger.info(
        f"Loaded {len(sample_tuples)} samples, split into {len(train_examples)} train / {len(val_examples)} val"
    )

    # Interleave training examples (critical for multi-task optimization)
    # MIPRO's dataset observation looks at first ~10 examples, so we need
    # to ensure all tasks appear early to signal multi-task nature
    import random

    rng = random.Random(42)
    train_examples = interleave_tasks(train_examples, rng=rng)

    return train_examples, val_examples, task_configs


def run_optimization(
    agent_wrapper: ReactInspectAgent,
    train_examples: list,
    val_examples: list,
    optimizer_config: OptimizerConfig,
    num_tasks: int,
    run_dir: str,
) -> tuple[Any, str]:
    """Configure optimizer and run optimization.

    Returns:
        Tuple of (optimized_module, optimizer_name)
    """
    metric = agent_wrapper.create_logging_metric()

    compile_kwargs = {
        "trainset": train_examples,
        "max_bootstrapped_demos": optimizer_config.max_bootstrapped_demos,
        "max_labeled_demos": optimizer_config.max_labeled_demos,
    }

    optimizer_type = optimizer_config.optimizer_type

    if optimizer_type == "gepa":
        try:
            from dspy.propose import GEPA

            optimizer = GEPA(
                metric=metric,
                breadth=optimizer_config.num_candidates,
                depth=3,
                init_temperature=optimizer_config.temperature,
            )
            optimizer_name = "GEPA"
        except ImportError:
            raise ImportError(
                "GEPA optimizer not available. Install with: pip install dspy-ai[gepa]"
            )

    elif optimizer_type == "mipro":
        optimizer = dspy.MIPROv2(
            metric=metric,
            auto=None,
            num_candidates=optimizer_config.num_candidates,
            init_temperature=optimizer_config.temperature,
            log_dir=run_dir,
        )
        optimizer_name = "MIPROv2"

        # Calculate num_trials using DSPy's formula if not provided
        # Formula: max(2 * num_vars * log2(N), 1.5 * N)
        # where num_vars = num_predictors * 2 (system + continue message)
        num_trials = optimizer_config.num_trials
        if num_trials is None:
            react_prompts = agent_wrapper.get_tunable_module()
            num_predictors = len(react_prompts.predictors())
            num_vars = num_predictors * 2
            num_trials = int(
                max(
                    2 * num_vars * np.log2(optimizer_config.num_candidates),
                    1.5 * optimizer_config.num_candidates,
                )
            )

        compile_kwargs["num_trials"] = num_trials

        # Calculate adaptive minibatch size
        # Ensures adequate coverage across all tasks during optimization
        desired_minibatch = num_tasks * SAMPLES_PER_TASK_FOR_MINIBATCH
        compile_kwargs["minibatch_size"] = min(
            len(train_examples), len(val_examples), desired_minibatch
        )
        compile_kwargs["minibatch"] = True

        if len(val_examples) < desired_minibatch:
            logger.warning(
                f"Validation set size ({len(val_examples)}) limits minibatch size. "
                f"Consider increasing samples_per_task for better multi-task coverage."
            )

    elif optimizer_type == "bootstrap":
        optimizer = dspy.BootstrapFewShot(metric=metric)
        optimizer_name = "BootstrapFewShot"

    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Choose from: 'gepa', 'mipro', 'bootstrap'"
        )

    logger.info(f"Optimizer config: {json.dumps(asdict(optimizer_config))}")
    logger.info(
        f"Compile kwargs: {json.dumps({k: v for k, v in compile_kwargs.items() if k != 'trainset'})}"
    )
    logger.info(
        f"Starting {optimizer_name} optimization with {len(train_examples)} train samples..."
    )

    react_prompts = agent_wrapper.get_tunable_module()
    optimized_module = optimizer.compile(react_prompts, **compile_kwargs)

    logger.info("Optimization complete")

    return optimized_module, optimizer_name


# ============================================================================
# Main Optimization Function
# ============================================================================


def optimize_react_prompts(
    task_config: TaskLoadingConfig,
    agent_config: ReactAgentConfig,
    optimizer_config: OptimizerConfig,
    run_config: RunConfig,
) -> dict:
    """Optimize ReAct agent prompts using DSPy across multiple tasks and models.

    This is the main entry point that orchestrates the optimization pipeline.

    Args:
        task_config: Configuration for loading task data
        agent_config: Configuration for ReAct agent wrapper
        optimizer_config: Configuration for DSPy optimizer
        run_config: Runtime configuration

    Returns:
        Dictionary with optimized prompts and metadata
    """
    run_dir = setup_optimization_run(optimizer_config, agent_config, run_config)
    train_examples, val_examples, task_configs = load_and_prepare_data(task_config)

    agent_config.log_dir = f"{run_dir}/eval_logs"
    agent_wrapper = ReactInspectAgent(agent_config)
    logger.info(f"Agent config: {json.dumps(asdict(agent_config))}")

    optimized_module, optimizer_name = run_optimization(
        agent_wrapper=agent_wrapper,
        train_examples=train_examples,
        val_examples=val_examples,
        optimizer_config=optimizer_config,
        num_tasks=len(task_configs),
        run_dir=run_dir,
    )

    optimized_prediction = optimized_module(
        task_description=GENERIC_TASK_DESCRIPTION,
        submit_function_name=DEFAULT_SUBMIT_NAME,
    )

    optimizer_model = optimizer_config.optimizer_model or agent_config.eval_models[0]

    optimized_prompts = {
        "system_message": optimized_prediction.system_message,
        "continue_message": optimized_prediction.continue_message,
        "metadata": {
            "eval_models": agent_config.eval_models,
            "optimizer_model": optimizer_model,
            "optimizer": optimizer_name,
            "num_candidates": (
                optimizer_config.num_candidates
                if optimizer_config.optimizer_type != "bootstrap"
                else None
            ),
            "train_samples": len(train_examples),
            "val_samples": len(val_examples),
            "tasks": [tc.name for tc in task_configs],
            "task_paths": [tc.path for tc in task_configs],
            "task_split": task_config.task_split,
            "samples_per_task": task_config.samples_per_task,
        },
    }

    output_path = Path(__file__).parent / run_config.output_file
    with open(output_path, "w") as f:
        json.dump(optimized_prompts, f, indent=2)

    logger.info(f"Saved optimized prompts to: {output_path}")

    if val_examples:
        metric = agent_wrapper.create_logging_metric()
        val_score = metric(val_examples[0], optimized_prediction)
        logger.info(f"Validation score: {val_score:.4f}")

    logger.info(f"Run directory: {run_dir}")

    return optimized_prompts


# ============================================================================
# CLI Interface
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize ReAct agent prompts using DSPy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data loading
    data_group = parser.add_argument_group("Data Loading")
    data_group.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to astabench config YAML (defaults to astabench v1.0.0)",
    )
    data_group.add_argument(
        "--task-split",
        type=str,
        default="validation",
        help="Which split to use from config. Mutually exclusive with --tasks.",
    )
    data_group.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task paths (e.g., 'astabench/sqa_dev'). Mutually exclusive with --task-split.",
    )
    data_group.add_argument(
        "--samples-per-task",
        type=int,
        default=5,
        help="Max samples per task",
    )
    data_group.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (0.8 = 80/20)",
    )

    # Agent configuration
    agent_group = parser.add_argument_group("Agent Configuration")
    agent_group.add_argument(
        "--models",
        type=str,
        default="openai/gpt-4o",
        help="Comma-separated models for evaluation (scores averaged)",
    )
    agent_group.add_argument(
        "--agent-max-steps",
        type=int,
        default=10,
        help="Maximum agent steps",
    )

    # Optimizer configuration
    opt_group = parser.add_argument_group("Optimizer Configuration")
    opt_group.add_argument(
        "--optimizer",
        type=str,
        default="mipro",
        choices=["gepa", "mipro", "bootstrap"],
        help="DSPy optimizer to use",
    )
    opt_group.add_argument(
        "--optimizer-model",
        type=str,
        default=None,
        help="Model for DSPy optimization (defaults to first eval model)",
    )
    opt_group.add_argument(
        "--num-candidates",
        type=int,
        default=5,
        help="Number of candidate prompts",
    )
    opt_group.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="MIPRO trials (defaults to DSPy formula)",
    )
    opt_group.add_argument(
        "--optimizer-temperature",
        type=float,
        default=1.0,
        help="Temperature for prompt generation",
    )

    # Evaluation configuration
    eval_group = parser.add_argument_group("Evaluation Configuration")
    eval_group.add_argument(
        "--eval-timeout",
        type=int,
        default=1200,
        help="Timeout per sample evaluation (seconds)",
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output",
        type=str,
        default="optimized_prompts.json",
        help="Output file for optimized prompts",
    )
    output_group.add_argument(
        "--verbose-llm",
        action="store_true",
        help="Enable LLM call logging",
    )

    args = parser.parse_args()

    if args.task_split != "validation" and args.tasks:
        parser.error("Cannot specify both --task-split and --tasks")

    models = [m.strip() for m in args.models.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    task_split = None if tasks else args.task_split

    agent_kwargs = {"max_steps": args.agent_max_steps}

    task_config = TaskLoadingConfig(
        config_path=args.config_path,
        task_split=task_split,
        tasks=tasks,
        samples_per_task=args.samples_per_task,
        train_ratio=args.train_ratio,
    )

    agent_config = ReactAgentConfig(
        eval_models=models,
        agent_kwargs=agent_kwargs,
        eval_timeout=args.eval_timeout,
        log_dir=None,  # Will be set in optimize_react_prompts based on run_dir
    )

    optimizer_config = OptimizerConfig(
        optimizer_type=args.optimizer,
        optimizer_model=args.optimizer_model,
        num_candidates=args.num_candidates,
        num_trials=args.num_trials,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
        temperature=args.optimizer_temperature,
    )

    run_config = RunConfig(
        run_dir=None,  # Will be created in optimize_react_prompts
        output_file=args.output,
        verbose_llm=args.verbose_llm,
    )

    optimize_react_prompts(
        task_config=task_config,
        agent_config=agent_config,
        optimizer_config=optimizer_config,
        run_config=run_config,
    )

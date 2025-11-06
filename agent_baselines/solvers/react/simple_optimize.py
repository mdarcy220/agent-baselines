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
from dataclasses import dataclass
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
        3. Runs eval_in_subprocess with those inputs
        4. Returns averaged score across models

        The metric has no side effects (no logging, no I/O) - it's just
        evaluation logic. This makes it testable and reusable.
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
            # Extract prompts from prediction
            system_message = prediction.system_message
            continue_message = prediction.continue_message

            # Extract metadata from example
            sample_id = example.sample_id
            task_path = example.task_path
            primary_metric = example.primary_metric

            # Run evaluation in subprocess (enables parallelization)
            # This handles multi-model evaluation and returns averaged score
            score_value = eval_in_subprocess(
                sample_id=sample_id,
                system_message=system_message,
                continue_message=continue_message,
                model_names=self.config.eval_models,
                task_path=task_path,
                primary_metric=primary_metric,
                timeout=self.config.eval_timeout,
                agent_kwargs=self.config.agent_kwargs,
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
            task_path = example.task_path

            # Log evaluation start
            logger.info("=" * 80)
            logger.info(f"DSPy Metric Evaluation:")
            logger.info(f"  Sample: {sample_id}")
            logger.info(f"  Task: {task_name} ({task_path})")
            logger.info(f"  Models: {', '.join(self.config.eval_models)}")
            logger.info(f"  Testing candidate prompts...")

            # Optional: write detailed logs to file
            if self.config.log_dir:
                os.makedirs(self.config.log_dir, exist_ok=True)
                timestamp = int(time.time() * 1000)
                std_log_file = f"{self.config.log_dir}/{sample_id}_{timestamp}.log"
                logger.info(f"  (Detailed logs → {std_log_file})")

            logger.info("=" * 80)

            try:
                # Call base metric for evaluation
                score_value = base_metric(
                    example, prediction, trace, pred_name, pred_trace
                )

                # Log success
                logger.info(
                    f"✓ Sample {sample_id} completed: avg score = {score_value:.4f}\n"
                )

                return score_value

            except Exception as e:
                # Log failure
                logger.info(f"✗ Sample {sample_id} failed: {e}\n")
                logger.error(
                    f"Failed to evaluate sample {sample_id} on task {task_path}: {e}"
                )
                raise

        return logging_metric


# ============================================================================
# Optimizer Configuration
# ============================================================================


def create_optimizer(
    optimizer_type: str,
    metric: Callable,
    num_candidates: int,
    temperature: float,
    run_dir: str,
) -> tuple[Any, str]:
    """Create and configure DSPy optimizer.

    Args:
        optimizer_type: "mipro", "gepa", or "bootstrap"
        metric: Metric function for evaluation
        num_candidates: Number of candidate prompts
        temperature: Temperature for prompt generation
        run_dir: Directory for optimizer logs

    Returns:
        Tuple of (optimizer instance, optimizer name)
    """
    if optimizer_type == "gepa":
        try:
            from dspy.propose import GEPA

            optimizer = GEPA(
                metric=metric,
                breadth=num_candidates,
                depth=3,
                init_temperature=temperature,
            )
            return optimizer, "GEPA"
        except ImportError:
            raise ImportError(
                "GEPA optimizer not available. Install with: pip install dspy-ai[gepa]"
            )

    elif optimizer_type == "mipro":
        optimizer = dspy.MIPROv2(
            metric=metric,
            auto=None,
            num_candidates=num_candidates,
            init_temperature=temperature,
            log_dir=run_dir,
        )
        return optimizer, "MIPROv2"

    elif optimizer_type == "bootstrap":
        optimizer = dspy.BootstrapFewShot(metric=metric)
        return optimizer, "BootstrapFewShot"

    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Choose from: 'gepa', 'mipro', 'bootstrap'"
        )


def calculate_compile_kwargs(
    optimizer_type: str,
    train_examples: list,
    val_examples: list,
    num_tasks: int,
    num_candidates: int,
    num_trials: int | None,
    max_bootstrapped_demos: int,
    max_labeled_demos: int,
) -> dict:
    """Calculate compilation kwargs for optimizer.

    This handles optimizer-specific parameter calculation, particularly for
    MIPRO's num_trials and minibatch_size formulas.

    Args:
        optimizer_type: Type of optimizer
        train_examples: Training examples
        val_examples: Validation examples
        num_tasks: Number of tasks being optimized
        num_candidates: Number of candidate prompts
        num_trials: Manual override for num_trials (MIPRO only)
        max_bootstrapped_demos: Max bootstrapped demonstrations
        max_labeled_demos: Max labeled demonstrations

    Returns:
        Dictionary of kwargs for optimizer.compile()
    """
    compile_kwargs = {
        "trainset": train_examples,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
    }

    if optimizer_type == "mipro":
        # Calculate num_trials using DSPy's formula if not provided
        # Formula: max(2 * num_vars * log2(N), 1.5 * N)
        # where num_vars = num_predictors * 2 (system + continue message)
        if num_trials is None:
            # Create temporary module to count predictors
            # NOTE: This creates a second instance (first is in optimization phase)
            # but is necessary to calculate trials before optimization starts
            react_prompts = DSPyReActPrompts()
            num_predictors = len(react_prompts.predictors())
            num_vars = num_predictors * 2
            num_trials = int(
                max(2 * num_vars * np.log2(num_candidates), 1.5 * num_candidates)
            )

        compile_kwargs["num_trials"] = num_trials

        # Calculate adaptive minibatch size
        # Ensures adequate coverage across all tasks during optimization
        desired_minibatch = num_tasks * SAMPLES_PER_TASK_FOR_MINIBATCH
        minibatch_size = min(len(train_examples), len(val_examples), desired_minibatch)

        compile_kwargs["minibatch_size"] = minibatch_size
        compile_kwargs["minibatch"] = True

        # Log configuration
        logger.info(f"MIPRO num_trials: {num_trials} (DSPy formula)")
        logger.info(
            f"MIPRO minibatch_size: {minibatch_size} "
            f"(min of train={len(train_examples)}, val={len(val_examples)}, "
            f"desired={desired_minibatch} [{num_tasks} tasks × {SAMPLES_PER_TASK_FOR_MINIBATCH}])"
        )

        if len(val_examples) < desired_minibatch:
            logger.warning(
                f"Validation set size ({len(val_examples)}) limits minibatch size. "
                f"Consider increasing samples_per_task for better multi-task coverage."
            )

    return compile_kwargs


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

        # Create log directory
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Initialize log file
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
# Main Optimization Function
# ============================================================================


def optimize_react_prompts(
    # Data loading
    config_path: str | None = None,
    task_split: str | None = "validation",
    tasks: list[str] | None = None,
    samples_per_task: int = 5,
    train_ratio: float = 0.8,
    # Agent configuration
    eval_models: list[str] | str = "openai/gpt-4o",
    agent_kwargs: dict | None = None,
    # Optimizer configuration
    optimizer_type: str = "mipro",
    optimizer_model: str | None = None,
    num_candidates: int = 5,
    num_trials: int | None = None,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 3,
    optimizer_temperature: float = 1.0,
    # Evaluation configuration
    eval_timeout: int = 600,
    # Output configuration
    output_file: str = "optimized_prompts.json",
    run_dir: str | None = None,
    verbose_llm: bool = False,
) -> dict:
    """Optimize ReAct agent prompts using DSPy across multiple tasks and models.

    This is the main entry point that orchestrates the optimization pipeline.

    Args:
        # Data loading
        config_path: Path to astabench config (defaults to astabench v1.0.0)
        task_split: Which split to use (e.g., "validation"). Mutually exclusive with tasks.
        tasks: Specific task paths (e.g., ["astabench/sqa_dev"]). Mutually exclusive with task_split.
        samples_per_task: Max samples per task
        train_ratio: Train/val split ratio (default 0.8 = 80/20)

        # Agent configuration
        eval_models: Model(s) for agent evaluation (string or list)
        agent_kwargs: Agent config dict (e.g., {"max_steps": 10})

        # Optimizer configuration
        optimizer_type: "mipro", "gepa", or "bootstrap"
        optimizer_model: Model for DSPy optimization (defaults to first eval model)
        num_candidates: Number of candidate prompts
        num_trials: MIPRO trials (defaults to DSPy formula)
        max_bootstrapped_demos: Max bootstrapped demonstrations
        max_labeled_demos: Max labeled demonstrations
        optimizer_temperature: Temperature for prompt generation

        # Evaluation configuration
        eval_timeout: Timeout per sample evaluation (seconds)

        # Output configuration
        output_file: Where to save optimized prompts
        run_dir: Run directory (defaults to timestamped .dspy_cache/run_*)
        verbose_llm: Enable LLM call logging

    Returns:
        Dictionary with optimized prompts and metadata
    """
    # ========================================
    # Setup and Validation
    # ========================================

    # Default configurations
    if agent_kwargs is None:
        agent_kwargs = {"max_steps": 10}

    # Normalize eval_models to list
    if isinstance(eval_models, str):
        eval_models = [eval_models]

    # Create run directory
    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = f".dspy_cache/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"RUN DIRECTORY: {run_dir}")
    logger.info("=" * 80)

    # Configure DSPy
    optimizer_model = optimizer_model or eval_models[0]
    setup_dspy_with_logging(optimizer_model, run_dir, verbose_llm)

    logger.info(f"Eval models: {', '.join(eval_models)}")
    logger.info(f"Optimizer model: {optimizer_model}")

    # ========================================
    # Phase 1: Load Task Data
    # ========================================

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: Loading task data")
    logger.info("=" * 80)

    # Load task configurations
    task_configs = load_tasks_from_config(
        config_path=config_path,
        split=task_split,
        task_paths=tasks,
    )

    logger.info(f"Loaded {len(task_configs)} task configs")
    for tc in task_configs:
        logger.info(f"  - {tc.name} ({tc.path})")

    # Load samples from tasks
    sample_tuples = load_samples_from_tasks(
        task_configs=task_configs,
        samples_per_task=samples_per_task,
    )
    logger.info(f"Loaded {len(sample_tuples)} total samples across all tasks")

    # Create DSPy examples and split train/val
    train_examples, val_examples = create_mixed_dspy_examples(
        sample_tuples=sample_tuples,
        train_ratio=train_ratio,
    )
    logger.info(f"Split: {len(train_examples)} train, {len(val_examples)} val")

    # Interleave training examples (critical for multi-task optimization)
    # MIPRO's dataset observation looks at first ~10 examples, so we need
    # to ensure all tasks appear early to signal multi-task nature
    import random

    rng = random.Random(42)
    train_examples = interleave_tasks(train_examples, rng=rng)

    # ========================================
    # Phase 2: Create Agent Wrapper
    # ========================================

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Creating agent wrapper")
    logger.info("=" * 80)

    agent_config = ReactAgentConfig(
        eval_models=eval_models,
        agent_kwargs=agent_kwargs,
        eval_timeout=eval_timeout,
        log_dir=f"{run_dir}/eval_logs",
    )

    agent_wrapper = ReactInspectAgent(agent_config)

    # Create metric function (with logging for visibility)
    metric = agent_wrapper.create_logging_metric()

    logger.info("Agent wrapper created")
    logger.info(f"  Eval models: {eval_models}")
    logger.info(f"  Agent kwargs: {agent_kwargs}")
    logger.info(f"  Eval timeout: {eval_timeout}s")
    logger.info(f"  Eval logs: {agent_config.log_dir}")

    # ========================================
    # Phase 3: Configure Optimizer
    # ========================================

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: Configuring optimizer")
    logger.info("=" * 80)

    optimizer, optimizer_name = create_optimizer(
        optimizer_type=optimizer_type,
        metric=metric,
        num_candidates=num_candidates,
        temperature=optimizer_temperature,
        run_dir=run_dir,
    )

    logger.info(f"Optimizer: {optimizer_name}")
    logger.info(f"  Candidates: {num_candidates}")
    logger.info(f"  Temperature: {optimizer_temperature}")

    compile_kwargs = calculate_compile_kwargs(
        optimizer_type=optimizer_type,
        train_examples=train_examples,
        val_examples=val_examples,
        num_tasks=len(task_configs),
        num_candidates=num_candidates,
        num_trials=num_trials,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )

    logger.info(f"Compile kwargs: {list(compile_kwargs.keys())}")

    # ========================================
    # Phase 4: Run Optimization
    # ========================================

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: Running optimization")
    logger.info("=" * 80)

    logger.info(f"Training samples: {len(train_examples)}")
    logger.info(f"Validation samples: {len(val_examples)}")
    logger.info("Starting DSPy optimizer.compile()...")

    # Create the DSPy module to optimize
    react_prompts = DSPyReActPrompts()

    # Run optimization
    optimized_module = optimizer.compile(react_prompts, **compile_kwargs)

    logger.info("Optimization complete!")

    # ========================================
    # Phase 5: Save and Evaluate Results
    # ========================================

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: Saving results")
    logger.info("=" * 80)

    # Generate final optimized prompts
    optimized_prediction = optimized_module(
        task_description=GENERIC_TASK_DESCRIPTION,
        submit_function_name=DEFAULT_SUBMIT_NAME,
    )

    # Package results
    optimized_prompts = {
        "system_message": optimized_prediction.system_message,
        "continue_message": optimized_prediction.continue_message,
        "metadata": {
            "eval_models": eval_models,
            "optimizer_model": optimizer_model,
            "optimizer": optimizer_name,
            "num_candidates": num_candidates if optimizer_type != "bootstrap" else None,
            "train_samples": len(train_examples),
            "val_samples": len(val_examples),
            "tasks": [tc.name for tc in task_configs],
            "task_paths": [tc.path for tc in task_configs],
            "task_split": task_split,
            "samples_per_task": samples_per_task,
        },
    }

    # Save to file
    output_path = Path(__file__).parent / output_file
    with open(output_path, "w") as f:
        json.dump(optimized_prompts, f, indent=2)

    logger.info(f"Saved optimized prompts to: {output_path}")
    logger.info(f"\nOptimized System Message:\n{optimized_prediction.system_message}")
    logger.info(
        f"\nOptimized Continue Message:\n{optimized_prediction.continue_message}"
    )

    # Validate on first val example
    if val_examples:
        logger.info("\nValidating on first val example...")
        val_score = metric(val_examples[0], optimized_prediction)
        logger.info(f"Validation score: {val_score:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Output file: {output_path}")

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

    # Validate mutually exclusive args
    if args.task_split != "validation" and args.tasks:
        parser.error("Cannot specify both --task-split and --tasks")

    # Parse comma-separated lists
    models = [m.strip() for m in args.models.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    task_split = None if tasks else args.task_split

    # Build agent_kwargs
    agent_kwargs = {"max_steps": args.agent_max_steps}

    # Run optimization
    optimize_react_prompts(
        # Data loading
        config_path=args.config_path,
        task_split=task_split,
        tasks=tasks,
        samples_per_task=args.samples_per_task,
        train_ratio=args.train_ratio,
        # Agent configuration
        eval_models=models,
        agent_kwargs=agent_kwargs,
        # Optimizer configuration
        optimizer_type=args.optimizer,
        optimizer_model=args.optimizer_model,
        num_candidates=args.num_candidates,
        num_trials=args.num_trials,
        optimizer_temperature=args.optimizer_temperature,
        # Evaluation configuration
        eval_timeout=args.eval_timeout,
        # Output configuration
        output_file=args.output,
        verbose_llm=args.verbose_llm,
    )

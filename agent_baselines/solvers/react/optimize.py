"""Script to optimize ReAct agent prompts using DSPy GEPA optimizer.

This script:
1. Loads the sqa_dev dataset
2. Splits it into train (80%) and validation (20%) sets
3. Uses GEPA to optimize system and continue message prompts
4. Saves the optimized prompts to a JSON file

Key architecture:
- Each DSPy Example represents one sample from sqa_dev
- The metric function runs inspect_ai.eval() on that specific sample
- DSPy optimizes prompts based on per-sample scores
"""

import json
import logging
import os
from pathlib import Path

import dspy

from agent_baselines.solvers.react.basic_agent import (
    DEFAULT_SUBMIT_NAME,
)
from agent_baselines.solvers.react.dspy_agent import (
    DSPyReActPrompts,
)
from agent_baselines.solvers.react.parallel_eval import eval_in_subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Default task description for SQA
SQA_TASK_DESCRIPTION = """Generate a report answering research questions with inline citations.
The agent should use search tools to find relevant papers, read them, and synthesize a well-cited response."""


def load_tasks_from_config(
    config_path: str | None = None,
    split: str | None = "validation",
    task_paths: list[str] | None = None,
) -> list[dict]:
    """Load task configurations from astabench config.

    Args:
        config_path: Path to astabench config YAML (default: uses astabench's default)
        split: Which split to load tasks from (default: "validation"). Mutually exclusive with task_paths.
        task_paths: Specific task paths to load (e.g., ["astabench/sqa_dev"]). Mutually exclusive with split.

    Returns:
        List of task configs with name, path, primary_metric, tags

    Raises:
        ValueError: If both split and task_paths are specified, or if a task path is not found in config
    """
    import importlib.resources
    import os

    import yaml
    from agenteval.models import SuiteConfig

    # Validate mutually exclusive args
    if split is not None and task_paths is not None:
        raise ValueError(
            "Cannot specify both 'split' and 'task_paths' - they are mutually exclusive"
        )

    if split is None and task_paths is None:
        raise ValueError("Must specify either 'split' or 'task_paths'")

    # Use astabench's default config if not specified
    if config_path is None:
        with importlib.resources.path("astabench.config", "v1.0.0.yml") as path:
            config_path = os.path.abspath(path)

    # Load and parse config
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    suite_config = SuiteConfig.model_validate(config_data)

    # Get all tasks from all splits (we need to search across splits for specific paths)
    all_tasks = []
    for split_data in config_data.get("splits", []):
        split_name = split_data.get("name")
        tasks_in_split = suite_config.get_tasks(split_name)
        all_tasks.extend(tasks_in_split)

    # If specific task paths requested, filter to those
    if task_paths:
        task_configs = []
        for task_path in task_paths:
            # Find task with matching path
            matching_task = None
            for task in all_tasks:
                if task.path == task_path:
                    matching_task = task
                    break

            if matching_task is None:
                raise ValueError(
                    f"Task path '{task_path}' not found in config. "
                    f"Available paths: {[t.path for t in all_tasks]}"
                )

            task_configs.append(matching_task)

        return task_configs

    # Otherwise return all tasks from the specified split
    return suite_config.get_tasks(split)


def create_metric_function(model_names: list[str], optimizer_type: str = "mipro"):
    """Create a metric function that evaluates agent performance across tasks and models.

    This metric function:
    1. Takes a DSPy prediction (containing candidate prompts)
    2. Runs inspect_ai.eval() in a subprocess (for parallelization)
    3. Evaluates on all specified models and averages the scores
    4. Returns the averaged score (or score + feedback for GEPA)

    Args:
        model_names: List of model names to evaluate on (scores will be averaged)
        optimizer_type: Type of optimizer ("mipro" or "gepa") for feedback support

    Returns:
        A function that takes (example, prediction, trace, pred_name, pred_trace) and returns:
        - float score for MIPROv2
        - dict with {"score": float, "feedback": str} for GEPA
    """

    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        """DSPy metric function for multi-task, multi-model evaluation.

        Args:
            example: DSPy Example containing sample_id, task_path, and primary_metric
            prediction: DSPy Prediction with system_message and continue_message
            trace: Optional trace information

        Returns:
            Score averaged across models for this sample
        """
        # Get the prompts from the prediction
        system_message = prediction.system_message
        continue_message = prediction.continue_message

        # Extract task info from example
        sample_id = example.sample_id
        task_path = example.task_path
        primary_metric = example.primary_metric
        task_name = example.task_name

        # Create log directory for this evaluation
        log_dir = ".dspy_cache/eval_logs"
        os.makedirs(log_dir, exist_ok=True)

        # Create unique log file for this sample evaluation
        # Use timestamp to avoid collisions if same sample is evaluated multiple times
        import time

        timestamp = int(time.time() * 1000)
        std_log_file = f"{log_dir}/{sample_id}_{timestamp}.log"

        # Print clearer DSPy-level logging
        logger.info(f"\n{'='*80}")
        logger.info(f"DSPy Metric Evaluation:")
        logger.info(f"  Sample: {sample_id}")
        logger.info(f"  Task: {task_name} ({task_path})")
        logger.info(f"  Models: {', '.join(model_names)}")
        logger.info(f"  Candidate prompts being tested...")
        logger.info(f"  (Detailed eval output → {std_log_file})")
        logger.info(f"{'='*80}\n")

        # Run eval in subprocess (enables parallelization)
        try:
            score_value = eval_in_subprocess(
                sample_id=sample_id,
                system_message=system_message,
                continue_message=continue_message,
                model_names=model_names,
                task_path=task_path,
                primary_metric=primary_metric,
                timeout=600,  # 10 minute timeout per sample
                std_log_file=std_log_file,
            )

            # Print result
            logger.info(
                f"✓ Sample {sample_id} completed: avg score = {score_value:.4f}\n"
            )
            return score_value

        except Exception as e:
            logger.info(f"✗ Sample {sample_id} failed: {e}\n")
            logger.error(
                f"Failed to evaluate sample {sample_id} on task {task_path}: {e}"
            )
            raise

    return metric


def load_samples_from_tasks(
    task_configs: list[dict],
    samples_per_task: int | None = None,
) -> list[tuple]:
    """Load samples from multiple tasks.

    Args:
        task_configs: List of task configs from load_tasks_from_config()
        samples_per_task: Optional limit on samples per task

    Returns:
        List of (task_path, primary_metric, task_name, sample) tuples
    """
    from inspect_ai._eval.loader import load_tasks

    all_sample_tuples = []

    for task_config in task_configs:
        task_path = task_config.path
        primary_metric = task_config.primary_metric
        task_name = task_config.name

        logger.info(f"Loading task: {task_name} ({task_path})")

        # Load task using inspect_ai's loader
        tasks = load_tasks([task_path], task_args={})
        assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"
        task = tasks[0]

        # Get samples from dataset
        samples = list(task.dataset)

        # Limit samples if specified
        if samples_per_task:
            samples = samples[:samples_per_task]

        logger.info(f"  Loaded {len(samples)} samples from {task_name}")

        # Create tuples with task metadata
        for sample in samples:
            all_sample_tuples.append((task_path, primary_metric, task_name, sample))

    return all_sample_tuples


def create_mixed_dspy_examples(
    sample_tuples: list[tuple],
    train_ratio: float = 0.8,
) -> tuple[list, list]:
    """Convert multi-task samples to DSPy examples and split into train/val.

    Each DSPy example contains:
    - task_description: Actual task content including the question/problem from sample.input
    - submit_function_name: Name of the submit function
    - sample: Full Sample object (for metric evaluation)
    - sample_id: The ID of the sample (for running eval on just this sample)
    - task_path: Path to the task (e.g., "astabench/sqa_dev")
    - primary_metric: Primary metric for this task (e.g., "global_avg/mean")
    - task_name: Human-readable task name

    This provides optimizers with actual task content they need for bootstrapping
    and instruction generation, while still using sample_id for evaluation.

    Args:
        sample_tuples: List of (task_path, primary_metric, task_name, sample) tuples
        train_ratio: Ratio of data to use for training (default 0.8)

    Returns:
        Tuple of (train_examples, val_examples)
    """
    # Split into train/val
    split_idx = int(len(sample_tuples) * train_ratio)
    train_tuples = sample_tuples[:split_idx]
    val_tuples = sample_tuples[split_idx:]

    logger.info(
        f"Creating DSPy examples: {len(train_tuples)} train, {len(val_tuples)} val"
    )

    def create_examples(tuples):
        examples = []
        for task_path, primary_metric, task_name, sample in tuples:
            # Create task description that includes actual content
            # This gives DSPy optimizers real task data to work with
            task_description = f"Task: {task_name}\n\nQuestion: {sample.input}"

            # Create a DSPy example with full sample data
            example = dspy.Example(
                task_description=task_description,  # Now includes actual question!
                submit_function_name=DEFAULT_SUBMIT_NAME,
                sample=sample,  # Full sample object for metric if needed
                sample_id=sample.id,
                task_path=task_path,
                primary_metric=primary_metric,
                task_name=task_name,
            ).with_inputs("task_description", "submit_function_name")

            examples.append(example)
        return examples

    train_examples = create_examples(train_tuples)
    val_examples = create_examples(val_tuples)

    return train_examples, val_examples


def optimize_prompts(
    models: list[str] | str = "openai/gpt-4o",
    optimizer_model: str | None = None,
    optimizer_type: str = "mipro",
    samples_per_task: int = 5,
    train_ratio: float = 0.8,
    num_candidates: int = 5,
    num_trials: int | None = None,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 3,
    output_file: str = "optimized_prompts.json",
    config_path: str | None = None,
    task_split: str | None = "validation",
    tasks: list[str] | None = None,
):
    """Run DSPy optimization on ReAct agent prompts across multiple tasks and models.

    Args:
        models: Model(s) to use for agent evaluation (can be single string or list)
        optimizer_model: Model to use for DSPy optimization (defaults to first model)
        optimizer_type: Which optimizer to use: "gepa", "mipro", or "bootstrap"
        samples_per_task: Maximum number of samples to use per task
        train_ratio: Ratio of data to use for training
        num_candidates: Number of candidate prompts (for GEPA/MIPRO)
        num_trials: Number of trials for MIPRO (defaults to ~3.6 * num_candidates)
        max_bootstrapped_demos: Max bootstrapped demonstrations
        max_labeled_demos: Max labeled demonstrations
        output_file: Where to save the optimized prompts
        config_path: Path to astabench config (default: uses astabench default)
        task_split: Which split to use from config (default: "validation"). Mutually exclusive with tasks.
        tasks: Specific task paths to use (e.g., ["astabench/sqa_dev"]). Mutually exclusive with task_split.
    """
    # Convert single model to list
    if isinstance(models, str):
        models = [models]

    # Set up DSPy language model for generating candidate prompts
    optimizer_model = optimizer_model or models[0]
    lm = dspy.LM(model=optimizer_model)
    # Allow DSPy to use default parallelization (we use subprocesses for eval isolation)
    dspy.settings.configure(lm=lm)

    logger.info(f"Agent models: {', '.join(models)}")
    logger.info(f"Optimizer model: {optimizer_model}")
    logger.info("Loading tasks from config...")
    logger.info(
        "Running evaluations in parallel using subprocess isolation (bypasses inspect_ai limitation)"
    )

    # Load task configs from astabench
    task_configs = load_tasks_from_config(
        config_path=config_path,
        split=task_split,
        task_paths=tasks,
    )

    if tasks:
        logger.info(f"Loaded {len(task_configs)} specified tasks")
    else:
        logger.info(f"Loaded {len(task_configs)} tasks from {task_split} split")

    # Load samples from all tasks
    sample_tuples = load_samples_from_tasks(
        task_configs=task_configs,
        samples_per_task=samples_per_task,
    )
    logger.info(f"Loaded {len(sample_tuples)} total samples across all tasks")

    # Create mixed DSPy examples and split into train/val
    train_examples, val_examples = create_mixed_dspy_examples(
        sample_tuples=sample_tuples,
        train_ratio=train_ratio,
    )

    logger.info(f"Using {len(train_examples)} train, {len(val_examples)} val examples")

    # Create the DSPy module
    react_prompts = DSPyReActPrompts()

    # Create metric function that uses subprocess-based eval
    logger.info("Creating metric function (subprocess-based for parallelization)...")
    metric = create_metric_function(models)

    # Set up optimizer based on type
    logger.info(f"Setting up {optimizer_type} optimizer...")

    # Initialize mipro_num_trials (only used for MIPRO)
    mipro_num_trials = None

    if optimizer_type == "gepa":
        try:
            from dspy.propose import GEPA

            optimizer = GEPA(
                metric=metric,
                breadth=num_candidates,
                depth=3,
                init_temperature=1.0,
            )
            optimizer_name = "GEPA"
        except ImportError:
            raise ImportError(
                "GEPA optimizer not available. Install with: pip install dspy-ai[gepa]"
            )

    elif optimizer_type == "mipro":
        # MIPRO has two modes:
        # 1. Auto mode: MIPROv2 decides num_candidates/num_trials
        # 2. Manual mode (auto=None): must provide num_candidates and num_trials
        # We use manual mode to allow user control
        optimizer = dspy.MIPROv2(
            metric=metric,
            auto=None,  # Enable manual mode
            num_candidates=num_candidates,
            init_temperature=1.0,
        )
        optimizer_name = "MIPROv2"
        # Calculate num_trials for compile() call (MIPRO needs it there, not in __init__)
        mipro_num_trials = num_trials or int(num_candidates * 3.6)
        logger.info(
            f"Using manual mode with num_candidates={num_candidates}, num_trials={mipro_num_trials}"
        )

    elif optimizer_type == "bootstrap":
        # BootstrapFewShot: max_bootstrapped_demos and max_labeled_demos
        # are passed to compile(), not __init__()
        optimizer = dspy.BootstrapFewShot(metric=metric)
        optimizer_name = "BootstrapFewShot"

    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Choose from: 'gepa', 'mipro', 'bootstrap'"
        )

    logger.info(f"Using {optimizer_name} optimizer")

    # Run optimization
    logger.info("\n" + "=" * 80)
    logger.info("STARTING DSPY OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"\nOptimizer: {optimizer_name}")
    logger.info(f"Training samples: {len(train_examples)}")
    logger.info(f"Models being optimized for: {', '.join(models)}")

    # Prepare compile() arguments based on optimizer type
    compile_kwargs = {
        "trainset": train_examples,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
    }

    # MIPRO needs num_trials and minibatch settings in compile() call
    if optimizer_type == "mipro":
        compile_kwargs["num_trials"] = mipro_num_trials
        # Adaptive minibatch sizing for multi-task optimization
        # Ensure adequate coverage across all task types (aim for ~5 examples per task)
        num_tasks = len(task_configs)
        minibatch_size = min(len(train_examples), num_tasks * 5)
        compile_kwargs["minibatch_size"] = minibatch_size
        compile_kwargs["minibatch"] = True
        logger.info(
            f"Minibatch size: {minibatch_size} (adaptive: {num_tasks} tasks × 5)"
        )
        logger.info(f"Number of trials: {mipro_num_trials}")
        logger.info(
            f"\nEach trial will evaluate {minibatch_size} sample(s) on {len(models)} model(s)"
        )
        logger.info(
            f"Estimated total evaluations: ~{mipro_num_trials * minibatch_size * len(models)}"
        )

    logger.info("\nDSPy will now generate and test candidate prompts...")
    logger.info("Watch for 'DSPy Metric Evaluation' blocks below to see progress.")
    logger.info("Detailed per-sample logs are being written to: .dspy_cache/eval_logs/")
    logger.info("=" * 80 + "\n")

    optimized_module = optimizer.compile(react_prompts, **compile_kwargs)

    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE!")
    logger.info("=" * 80 + "\n")

    # Generate optimized prompts
    logger.info("\nGenerating optimized prompts...")
    optimized_prediction = optimized_module(
        task_description=SQA_TASK_DESCRIPTION, submit_function_name=DEFAULT_SUBMIT_NAME
    )

    # Collect task information for metadata
    task_names = [config.name for config in task_configs]
    task_paths = [config.path for config in task_configs]

    optimized_prompts = {
        "system_message": optimized_prediction.system_message,
        "continue_message": optimized_prediction.continue_message,
        "metadata": {
            "agent_models": models,
            "optimizer_model": optimizer_model,
            "train_samples": len(train_examples),
            "val_samples": len(val_examples),
            "optimizer": optimizer_name,
            "num_candidates": num_candidates if optimizer_type != "bootstrap" else None,
            "tasks": task_names,
            "task_paths": task_paths,
            "task_split": task_split,
            "samples_per_task": samples_per_task,
        },
    }

    # Save to file
    output_path = Path(__file__).parent / output_file
    with open(output_path, "w") as f:
        json.dump(optimized_prompts, f, indent=2)

    # Print results to stdout (logger may be suppressed)
    logger.info(f"\n{'='*60}")
    logger.info("✓ Multi-Task, Multi-Model Optimization Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"\n📁 Optimized prompts saved to: {output_path}")
    logger.info(f"\n📝 Optimized System Message:")
    logger.info("-" * 60)
    logger.info(optimized_prompts["system_message"])
    logger.info(f"\n📝 Optimized Continue Message:")
    logger.info("-" * 60)
    logger.info(optimized_prompts["continue_message"])
    logger.info(f"\n{'='*60}")

    # Evaluate on validation set
    if val_examples:
        logger.info("\n📊 Evaluating first validation example...")
        val_score = metric(val_examples[0], optimized_prediction)
        logger.info(f"✓ Validation score (first sample): {val_score:.4f}")

    logger.info(f"\n💡 Optimization Details:")
    logger.info(f"   Agent models (for evaluation): {', '.join(models)}")
    logger.info(f"   Optimizer model (for prompt generation): {optimizer_model}")
    logger.info(f"   Tasks optimized: {len(task_configs)} tasks ({task_split} split)")
    logger.info(f"   Total training samples: {len(train_examples)} across all tasks")

    logger.info(f"\n✓ To use the optimized agent on any task, run:")
    logger.info(f"  uv run astabench eval <task_path> \\")
    logger.info(
        f"    --solver agent_baselines/solvers/react/optimized_agent.py@instantiated_optimized_agent \\"
    )
    logger.info(f"    --model <model_name>")
    logger.info(
        f"\n  (The optimized agent will automatically load prompts from {output_path})"
    )

    logger.info(
        f"\n✓ Multi-task optimization completed successfully using subprocess isolation!"
    )
    logger.info("")

    return optimized_prompts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize ReAct agent prompts across multiple tasks and models using DSPy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default="openai/gpt-4o",
        help="Comma-separated list of models to evaluate on (scores will be averaged)",
    )
    parser.add_argument(
        "--optimizer-model",
        type=str,
        default=None,
        help="Model to use for DSPy optimization (defaults to first model)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="mipro",
        choices=["gepa", "mipro", "bootstrap"],
        help="Which DSPy optimizer to use",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=5,
        help="Max samples to use per task",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of samples to use for training",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=5,
        help="Number of candidate prompts (for GEPA/MIPRO)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Number of trials for MIPRO (defaults to ~3.6 * num_candidates)",
    )
    parser.add_argument(
        "--output", type=str, default="optimized_prompts.json", help="Output file"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to astabench config YAML (defaults to astabench's v1.0.0.yml)",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="validation",
        help="Which split to use from config (default: validation). Mutually exclusive with --tasks.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of specific task paths (e.g., 'astabench/sqa_dev,astabench/litqa2_validation'). Mutually exclusive with --task-split.",
    )

    args = parser.parse_args()

    # Validate mutually exclusive args
    if args.task_split != "validation" and args.tasks:
        parser.error("Cannot specify both --task-split and --tasks")

    # Parse models list
    models = [m.strip() for m in args.models.split(",")]

    # Parse tasks list if provided
    tasks = None
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]

    # Set task_split to None if using --tasks
    task_split = None if tasks else args.task_split

    optimize_prompts(
        models=models,
        optimizer_model=args.optimizer_model,
        optimizer_type=args.optimizer,
        samples_per_task=args.samples_per_task,
        train_ratio=args.train_ratio,
        num_candidates=args.num_candidates,
        num_trials=args.num_trials,
        output_file=args.output,
        config_path=args.config_path,
        task_split=task_split,
        tasks=tasks,
    )

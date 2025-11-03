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
from pathlib import Path

import dspy
from astabench.evals.sqa import sqa_dev
from astabench.tools import ToolsetConfig

from agent_baselines.solvers.react.basic_agent import (
    DEFAULT_SUBMIT_NAME,
)
from agent_baselines.solvers.react.dspy_agent import (
    DSPyReActPrompts,
)
from agent_baselines.solvers.react.parallel_eval import eval_in_subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Default task description for SQA
SQA_TASK_DESCRIPTION = """Generate a report answering research questions with inline citations.
The agent should use search tools to find relevant papers, read them, and synthesize a well-cited response."""


def create_metric_function(model_name: str, tool_config: ToolsetConfig):
    """Create a metric function that evaluates agent performance on SQA.

    This metric function:
    1. Takes a DSPy prediction (containing candidate prompts)
    2. Runs inspect_ai.eval() in a subprocess (for parallelization)
    3. Returns the global_avg score

    Args:
        model_name: Name of the model to use for the agent
        tool_config: Tool configuration to use

    Returns:
        A function that takes (example, prediction, trace) and returns a score
    """
    # Convert tool config to dict for subprocess serialization
    tool_config_dict = tool_config.model_dump()

    def metric(example, prediction, trace=None) -> float:
        """DSPy metric function.

        Args:
            example: DSPy Example containing sample_id
            prediction: DSPy Prediction with system_message and continue_message
            trace: Optional trace information

        Returns:
            Score between 0 and 1 (global_avg for this sample)
        """
        # Get the prompts from the prediction
        system_message = prediction.system_message
        continue_message = prediction.continue_message
        sample_id = example.sample_id

        logger.info(f"Evaluating sample {sample_id} in subprocess")

        # Run eval in subprocess (enables parallelization)
        try:
            score_value = eval_in_subprocess(
                sample_id=sample_id,
                system_message=system_message,
                continue_message=continue_message,
                model_name=model_name,
                task_name="sqa_dev",
                tool_config_dict=tool_config_dict,
                timeout=600,  # 10 minute timeout per sample
            )

            logger.info(
                f"Sample {sample_id} score: {score_value:.4f} "
                f"(system_msg: {system_message[:50]}...)"
            )
            return score_value

        except Exception as e:
            logger.error(f"Failed to evaluate sample {sample_id}: {e}")
            raise

    return metric


def load_and_split_data(base_task, train_ratio: float = 0.8, limit: int = None):
    """Load sqa_dev data and split into train/val sets.

    Args:
        base_task: The sqa_dev task
        train_ratio: Ratio of data to use for training (default 0.8)
        limit: Optional limit on total number of samples to use

    Returns:
        Tuple of (train_samples, val_samples) where each sample has an id
    """
    all_samples = list(base_task.dataset)

    if limit:
        all_samples = all_samples[:limit]

    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    logger.info(f"Loaded {len(all_samples)} total samples")
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    return train_samples, val_samples


def create_dspy_examples(samples):
    """Convert inspect_ai samples to DSPy examples.

    Each DSPy example contains:
    - task_description: What the agent should do
    - submit_function_name: Name of the submit function
    - sample_id: The ID of the sample (for running eval on just this sample)

    Args:
        samples: List of inspect_ai samples from the dataset

    Returns:
        List of DSPy examples
    """
    examples = []
    for sample in samples:
        # Create a DSPy example with inputs that DSPy will pass to the module
        example = dspy.Example(
            task_description=SQA_TASK_DESCRIPTION,
            submit_function_name=DEFAULT_SUBMIT_NAME,
            sample_id=sample.id,  # Store sample ID for targeted evaluation
        ).with_inputs("task_description", "submit_function_name")

        examples.append(example)

    return examples


def optimize_prompts(
    model_name: str = "openai/gpt-4o",
    optimizer_model: str | None = None,
    optimizer_type: str = "mipro",
    train_limit: int = 20,
    val_limit: int = 10,
    train_ratio: float = 0.8,
    num_candidates: int = 5,
    num_trials: int | None = None,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 3,
    output_file: str = "optimized_prompts.json",
):
    """Run DSPy optimization on ReAct agent prompts.

    Args:
        model_name: Model to use for running the agent during evaluation
        optimizer_model: Model to use for DSPy optimization (defaults to model_name)
        optimizer_type: Which optimizer to use: "gepa", "mipro", or "bootstrap"
        train_limit: Maximum number of training samples
        val_limit: Maximum number of validation samples
        train_ratio: Ratio of data to use for training
        num_candidates: Number of candidate prompts (for GEPA/MIPRO)
        num_trials: Number of trials for MIPRO (defaults to ~3.6 * num_candidates)
        max_bootstrapped_demos: Max bootstrapped demonstrations
        max_labeled_demos: Max labeled demonstrations
        output_file: Where to save the optimized prompts
    """
    # Set up DSPy language model for generating candidate prompts
    optimizer_model = optimizer_model or model_name
    lm = dspy.LM(model=optimizer_model)
    # Allow DSPy to use default parallelization (we use subprocesses for eval isolation)
    dspy.settings.configure(lm=lm)

    logger.info(f"Agent model: {model_name}")
    logger.info(f"Optimizer model: {optimizer_model}")
    logger.info("Loading data...")
    logger.info(
        "Running evaluations in parallel using subprocess isolation (bypasses inspect_ai limitation)"
    )

    # Load base task
    base_task = sqa_dev()

    # Load and split data
    train_samples, val_samples = load_and_split_data(
        base_task, train_ratio=train_ratio, limit=train_limit
    )

    if val_limit:
        val_samples = val_samples[:val_limit]

    logger.info(f"Using {len(train_samples)} train, {len(val_samples)} val samples")

    # Convert to DSPy examples
    train_examples = create_dspy_examples(train_samples)
    val_examples = create_dspy_examples(val_samples)

    # Create tool configuration
    tool_config = ToolsetConfig(
        with_search_tools=True,
        with_report_editor=True,
    )

    # Create the DSPy module
    react_prompts = DSPyReActPrompts()

    # Create metric function that uses subprocess-based eval
    logger.info("Creating metric function (subprocess-based for parallelization)...")
    metric = create_metric_function(model_name, tool_config)

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
    logger.info(f"\nStarting optimization on {len(train_examples)} training examples")
    logger.info("This may take a while (each candidate runs inspect_ai.eval)...")

    # Prepare compile() arguments based on optimizer type
    compile_kwargs = {
        "trainset": train_examples,
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
    }

    # MIPRO needs num_trials and minibatch settings in compile() call
    if optimizer_type == "mipro":
        compile_kwargs["num_trials"] = mipro_num_trials
        # Set minibatch_size to not exceed valset size
        # Use min of (default 25, half of trainset size)
        minibatch_size = min(25, max(1, len(train_examples) // 2))
        compile_kwargs["minibatch_size"] = minibatch_size
        compile_kwargs["minibatch"] = True
        logger.info(f"Using minibatch_size={minibatch_size}")

    optimized_module = optimizer.compile(react_prompts, **compile_kwargs)

    # Generate optimized prompts
    logger.info("\nGenerating optimized prompts...")
    optimized_prediction = optimized_module(
        task_description=SQA_TASK_DESCRIPTION, submit_function_name=DEFAULT_SUBMIT_NAME
    )

    optimized_prompts = {
        "system_message": optimized_prediction.system_message,
        "continue_message": optimized_prediction.continue_message,
        "metadata": {
            "agent_model": model_name,
            "optimizer_model": optimizer_model,
            "train_samples": len(train_examples),
            "val_samples": len(val_examples),
            "optimizer": optimizer_name,
            "num_candidates": num_candidates if optimizer_type != "bootstrap" else None,
        },
    }

    # Save to file
    output_path = Path(__file__).parent / output_file
    with open(output_path, "w") as f:
        json.dump(optimized_prompts, f, indent=2)

    # Print results to stdout (logger may be suppressed)
    print(f"\n{'='*60}")
    print("✓ Optimization complete!")
    print(f"{'='*60}")
    print(f"\n📁 Optimized prompts saved to: {output_path}")
    print(f"\n📝 Optimized System Message:")
    print("-" * 60)
    print(optimized_prompts["system_message"])
    print(f"\n📝 Optimized Continue Message:")
    print("-" * 60)
    print(optimized_prompts["continue_message"])
    print(f"\n{'='*60}")

    # Evaluate on validation set
    if val_examples:
        print("\n📊 Evaluating first validation example...")
        val_score = metric(val_examples[0], optimized_prediction)
        print(f"✓ Validation score (first sample): {val_score:.4f}")

    print(f"\n💡 Optimization Details:")
    print(f"   Agent model (for evaluation): {model_name}")
    print(f"   Optimizer model (for prompt generation): {optimizer_model}")

    print(f"\n✓ To use the optimized agent, run:")
    print(f"  uv run astabench eval astabench/sqa_dev \\")
    print(
        f"    --solver agent_baselines/solvers/react/optimized_agent.py@instantiated_optimized_agent \\"
    )
    print(f"    --model {model_name}")
    print(
        f"\n  (The optimized agent will automatically load prompts from {output_path})"
    )

    print(f"\n✓ Parallel optimization completed successfully using subprocess isolation!")
    print()

    return optimized_prompts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize ReAct agent prompts using DSPy GEPA optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model to use for running the agent during evaluation",
    )
    parser.add_argument(
        "--optimizer-model",
        type=str,
        default=None,
        help="Model to use for DSPy optimization (defaults to --model)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="mipro",
        choices=["gepa", "mipro", "bootstrap"],
        help="Which DSPy optimizer to use",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=20,
        help="Max total samples to load (will be split into train/val)",
    )
    parser.add_argument(
        "--val-limit", type=int, default=10, help="Max validation samples"
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

    args = parser.parse_args()

    optimize_prompts(
        model_name=args.model,
        optimizer_model=args.optimizer_model,
        optimizer_type=args.optimizer,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        train_ratio=args.train_ratio,
        num_candidates=args.num_candidates,
        num_trials=args.num_trials,
        output_file=args.output,
    )

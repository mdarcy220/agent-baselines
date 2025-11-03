# Recommended Fixes for DSPy Optimization Code

This document provides specific code changes to address the critical and high-priority issues identified in the code review.

## Quick Summary

**Files to modify:**
- `agent_baselines/solvers/react/optimize.py` (7 changes)
- `agent_baselines/solvers/react/dspy_agent.py` (2 changes)

**Testing:**
- Run the provided `demonstrate_issues.py` to see the issues
- After fixes, run a small test optimization (3 samples, 2 candidates)

---

## Fix #1: Add Validation in Metric Function (CRITICAL)

**File**: `agent_baselines/solvers/react/optimize.py`

**Location**: Lines 60-131 (the `metric` function inside `create_metric_function`)

**Current code**:
```python
def metric(example, prediction, trace=None) -> float:
    # ...existing code...

    logs = inspect_eval(
        tasks=[task],
        model=model_name,
        sample_id=sample_id,
        log_dir=".dspy_cache",
        log_level="warning",
    )

    # Extract score from the specific sample
    if not logs or len(logs) == 0:
        logger.warning(f"No logs returned for sample {sample_id}")
        return 0.0

    eval_log = logs[0]

    # Get the score from the sample
    if not eval_log.samples or len(eval_log.samples) == 0:
        logger.warning(f"No samples in eval log for {sample_id}")
        return 0.0

    sample = eval_log.samples[0]

    # Extract global_avg from the sample's scores
    if not sample.scores:
        logger.warning(f"No scores for sample {sample_id}")
        return 0.0

    # The scores should have global_avg in the value dict
    for score in sample.scores:
        if isinstance(score.value, dict) and "global_avg" in score.value:
            score_value = float(score.value["global_avg"])
            logger.info(
                f"Sample {sample_id} score: {score_value:.4f} "
                f"(system_msg: {system_message[:50]}...)"
            )
            return score_value

    # Fallback: if no global_avg, return 0
    logger.warning(f"No global_avg found in scores for sample {sample_id}")
    return 0.0
```

**Replace with**:
```python
def metric(example, prediction, trace=None) -> float:
    """DSPy metric function.

    Args:
        example: DSPy Example containing sample_id
        prediction: DSPy Prediction with system_message and continue_message
        trace: Optional trace information

    Returns:
        Score between 0 and 1 (global_avg for this sample)

    Raises:
        RuntimeError: If evaluation fails or sample doesn't complete
        ValueError: If scores are missing or malformed
    """
    # Validate prediction structure
    if not hasattr(prediction, 'system_message'):
        raise ValueError(
            f"DSPy prediction missing 'system_message' attribute. "
            f"Prediction type: {type(prediction)}"
        )
    if not hasattr(prediction, 'continue_message'):
        raise ValueError(
            f"DSPy prediction missing 'continue_message' attribute. "
            f"Prediction type: {type(prediction)}"
        )

    # Get the prompts from the prediction
    system_message = prediction.system_message
    continue_message = prediction.continue_message
    sample_id = example.sample_id

    # Validate prompts are non-empty strings
    if not system_message or not isinstance(system_message, str):
        raise ValueError(f"Invalid system_message: {type(system_message)}")
    if not continue_message or not isinstance(continue_message, str):
        raise ValueError(f"Invalid continue_message: {type(continue_message)}")

    logger.info(f"Evaluating sample {sample_id} with candidate prompts")

    # Create solver with candidate prompts
    agent_solver = create_agent_with_dspy_prompts(
        system_message_text=system_message,
        continue_message_text=continue_message,
        max_steps=10,
        tools=tool_config.create_tools(),
        add_submit_tool=not tool_config.with_editor_submit,
    )

    # Create task with this solver
    task = base_task.model_copy(update={"solver": agent_solver})

    # Run eval on just this sample
    logs = inspect_eval(
        tasks=[task],
        model=model_name,
        sample_id=sample_id,
        log_dir=".dspy_cache",
        log_level="warning",
    )

    # Validate logs were returned
    if not logs or len(logs) == 0:
        raise RuntimeError(
            f"No evaluation logs returned for sample {sample_id}. "
            f"This indicates an infrastructure failure."
        )

    eval_log = logs[0]

    # Check evaluation status
    if hasattr(eval_log, 'status') and eval_log.status != "success":
        error_msg = getattr(eval_log, 'error', 'Unknown error')
        raise RuntimeError(
            f"Evaluation failed for sample {sample_id}: status={eval_log.status}. "
            f"Error: {error_msg}"
        )

    # Validate samples exist
    if not eval_log.samples or len(eval_log.samples) == 0:
        raise ValueError(
            f"No samples in evaluation log for {sample_id}. "
            f"This indicates a problem with inspect_ai evaluation."
        )

    sample = eval_log.samples[0]

    # Check if sample completed
    if hasattr(sample, 'completed') and not sample.completed:
        raise RuntimeError(
            f"Sample {sample_id} did not complete successfully. "
            f"Agent may have timed out, hit token limits, or encountered errors."
        )

    # Validate scores exist
    if not sample.scores:
        raise ValueError(
            f"No scores for sample {sample_id}. "
            f"Check that the task's scorer is configured correctly."
        )

    # Find global_avg score (fail-loud)
    score_obj = None
    for score in sample.scores:
        if isinstance(score.value, dict) and "global_avg" in score.value:
            score_obj = score
            break

    if score_obj is None:
        # Log available scores for debugging
        available_scores = []
        for s in sample.scores:
            if isinstance(s.value, dict):
                available_scores.append(list(s.value.keys()))
            else:
                available_scores.append(f"non-dict: {type(s.value).__name__}")

        raise ValueError(
            f"No 'global_avg' found in scores for sample {sample_id}. "
            f"Available score keys: {available_scores}. "
            f"Verify that the sqa_dev task produces global_avg scores."
        )

    # Extract score using fail-loud dict access
    score_value = float(score_obj.value["global_avg"])

    logger.info(
        f"Sample {sample_id} score: {score_value:.4f} "
        f"(system_msg: {system_message[:50]}...)"
    )
    return score_value
```

**Why**: This change eliminates silent failures and provides clear error messages for different failure modes, making debugging much easier and preventing corruption of optimization data.

---

## Fix #2: Correct BootstrapFewShot Parameters (CRITICAL)

**File**: `agent_baselines/solvers/react/optimize.py`

**Location**: Lines 293-299

**Current code**:
```python
elif optimizer_type == "bootstrap":
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )
    optimizer_name = "BootstrapFewShot"
```

**Replace with**:
```python
elif optimizer_type == "bootstrap":
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        # Note: max_bootstrapped_demos and max_labeled_demos are passed to
        # compile(), not __init__(). See compile_kwargs below.
    )
    optimizer_name = "BootstrapFewShot"
```

**Why**: BootstrapFewShot.__init__() doesn't accept these parameters. They must be passed to compile() instead, which the code already does correctly later.

---

## Fix #3: Add Shuffling to Data Split (CRITICAL)

**File**: `agent_baselines/solvers/react/optimize.py`

**Location**: Lines 135-158

**Add import at top of file**:
```python
import random  # Add to existing imports
```

**Current code**:
```python
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
```

**Replace with**:
```python
def load_and_split_data(
    base_task,
    train_ratio: float = 0.8,
    limit: int = None,
    shuffle: bool = True,
    seed: int = 42
):
    """Load sqa_dev data and split into train/val sets.

    Args:
        base_task: The sqa_dev task
        train_ratio: Ratio of data to use for training (default 0.8)
        limit: Optional limit on total number of samples to use
        shuffle: Whether to shuffle data before splitting (default True).
            Shuffling prevents bias from dataset ordering.
        seed: Random seed for reproducible shuffling (default 42)

    Returns:
        Tuple of (train_samples, val_samples) where each sample has an id
    """
    all_samples = list(base_task.dataset)

    # Shuffle to prevent ordering bias (e.g., easy samples first, hard last)
    if shuffle:
        random.seed(seed)
        random.shuffle(all_samples)
        logger.info(f"Shuffled {len(all_samples)} samples (seed={seed})")

    if limit:
        all_samples = all_samples[:limit]

    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    logger.info(f"Loaded {len(all_samples)} total samples")
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    return train_samples, val_samples
```

**Also update the call site** (line 230):
```python
# Load and split data
train_samples, val_samples = load_and_split_data(
    base_task,
    train_ratio=train_ratio,
    limit=train_limit,
    shuffle=True,  # Explicitly enable shuffling
    seed=42  # Reproducible results
)
```

**Why**: Prevents bias from dataset ordering. The sqa_dev dataset may have samples ordered by difficulty, topic, or date. Shuffling ensures both train and val sets have representative distributions.

---

## Fix #4: Add Explicit tool_call_format Parameter (HIGH)

**File**: `agent_baselines/solvers/react/dspy_agent.py`

**Location**: Lines 78-105

**Add import at top**:
```python
from typing import Literal  # Add to existing imports
```

**Current code**:
```python
def create_agent_with_dspy_prompts(
    system_message_text: str,
    continue_message_text: str,
    max_steps: int = 10,
    **kwargs,
) -> Solver:
    """Create a basic_agent solver with custom DSPy-optimized prompts.

    Args:
        system_message_text: The optimized system message
        continue_message_text: The optimized continue message
        max_steps: Maximum number of agent steps
        **kwargs: Additional arguments to pass to basic_agent

    Returns:
        A configured basic_agent solver
    """

    @solver
    def custom_system_message() -> Solver:
        return system_message(system_message_text, submit=DEFAULT_SUBMIT_NAME)

    return basic_agent(
        init=custom_system_message(),
        continue_message=continue_message_text,
        max_steps=max_steps,
        **kwargs,
    )
```

**Replace with**:
```python
def create_agent_with_dspy_prompts(
    system_message_text: str,
    continue_message_text: str,
    max_steps: int = 10,
    tool_call_format: Literal["text", "native"] = "native",
    **kwargs,
) -> Solver:
    """Create a basic_agent solver with custom DSPy-optimized prompts.

    Args:
        system_message_text: The optimized system message
        continue_message_text: The optimized continue message
        max_steps: Maximum number of agent steps
        tool_call_format: Format for tool calls ("text" or "native").
            IMPORTANT: Prompts optimized for one format may not work well
            with the other format. Ensure you use the same format during
            optimization and deployment.
        **kwargs: Additional arguments to pass to basic_agent

    Returns:
        A configured basic_agent solver
    """

    @solver
    def custom_system_message() -> Solver:
        return system_message(system_message_text, submit=DEFAULT_SUBMIT_NAME)

    return basic_agent(
        init=custom_system_message(),
        continue_message=continue_message_text,
        max_steps=max_steps,
        tool_call_format=tool_call_format,
        **kwargs,
    )
```

**Why**: Makes tool_call_format explicit and documents the importance of consistency between optimization and deployment.

---

## Fix #5: Evaluate Full Validation Set (HIGH)

**File**: `agent_baselines/solvers/react/optimize.py`

**Location**: Lines 368-372

**Current code**:
```python
# Evaluate on validation set
if val_examples:
    logger.info("\nEvaluating first validation example...")
    val_score = metric(val_examples[0], optimized_prediction)
    logger.info(f"Validation score (first sample): {val_score:.4f}")
```

**Replace with**:
```python
# Evaluate on validation set
if val_examples:
    logger.info(f"\nEvaluating on {len(val_examples)} validation examples...")
    val_scores = []

    for i, val_example in enumerate(val_examples):
        try:
            score = metric(val_example, optimized_prediction)
            val_scores.append(score)
            logger.info(f"  Val sample {i+1}/{len(val_examples)}: {score:.4f}")
        except Exception as e:
            logger.error(
                f"  Val sample {i+1}/{len(val_examples)} failed: {e}",
                exc_info=True
            )
            # Don't include failed samples in average

    if val_scores:
        avg_val_score = sum(val_scores) / len(val_scores)
        min_val_score = min(val_scores)
        max_val_score = max(val_scores)

        logger.info(f"\nValidation Results:")
        logger.info(f"  Average: {avg_val_score:.4f}")
        logger.info(f"  Min: {min_val_score:.4f}")
        logger.info(f"  Max: {max_val_score:.4f}")
        logger.info(f"  Scores: {[f'{s:.4f}' for s in val_scores]}")
    else:
        logger.warning("No validation scores computed (all failed)")
```

**Why**: Evaluating only one validation sample doesn't give a good sense of generalization. Full validation set evaluation provides more reliable metrics.

---

## Fix #6: Make Hard-coded Constants Configurable (MEDIUM)

**File**: `agent_baselines/solvers/react/optimize.py`

**Location**: Lines 189-216 (function signature)

**Current code**:
```python
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
```

**Replace with**:
```python
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
    max_steps: int = 10,  # NEW: Make max_steps configurable
    cache_dir: str = ".dspy_cache",  # NEW: Make cache dir configurable
    log_level: str = "warning",  # NEW: Make log level configurable
    gepa_depth: int = 3,  # NEW: Make GEPA depth configurable
    shuffle_data: bool = True,  # NEW: Make shuffling configurable
    random_seed: int = 42,  # NEW: Make random seed configurable
    output_file: str = "optimized_prompts.json",
):
```

Then update the function body to use these parameters:

**Line 82** (in create_metric_function):
```python
max_steps=max_steps,  # Use parameter instead of hard-coded 10
```

**Line 95** (in create_metric_function):
```python
log_dir=cache_dir,  # Use parameter instead of hard-coded ".dspy_cache"
log_level=log_level,  # Use parameter instead of hard-coded "warning"
```

**Line 230** (load_and_split_data call):
```python
train_samples, val_samples = load_and_split_data(
    base_task,
    train_ratio=train_ratio,
    limit=train_limit,
    shuffle=shuffle_data,  # Use parameter
    seed=random_seed  # Use parameter
)
```

**Line 268** (GEPA depth):
```python
optimizer = GEPA(
    metric=metric,
    breadth=num_candidates,
    depth=gepa_depth,  # Use parameter instead of hard-coded 3
    init_temperature=1.0,
)
```

**Why**: Makes the function more flexible for experimentation without requiring code changes.

---

## Fix #7: Add Type Hints (MEDIUM)

**File**: `agent_baselines/solvers/react/optimize.py`

**Add imports at top**:
```python
from typing import Callable, Any
from inspect_ai.dataset import Sample
from inspect_ai.task import Task
```

**Update function signatures**:

```python
def create_metric_function(
    model_name: str,
    base_task: Task,  # Add type hint
    tool_config: ToolsetConfig
) -> Callable[[Any, Any, Any], float]:  # Add return type
    """..."""

def create_dspy_examples(
    samples: list[Sample]  # Add type hint
) -> list[dspy.Example]:  # Add return type
    """..."""
```

**Why**: Better type safety and IDE support.

---

## Testing the Fixes

After applying the fixes, test with a small optimization run:

```bash
# Run with minimal parameters to test the fixes
python agent_baselines/solvers/react/optimize.py \
    --model openai/gpt-4o-mini \
    --optimizer-model openai/gpt-4o-mini \
    --train-limit 3 \
    --val-limit 2 \
    --num-candidates 2 \
    --optimizer mipro \
    --output test_optimized_prompts.json
```

This should:
1. Complete without errors
2. Show shuffling in the logs
3. Evaluate full validation set (2 samples)
4. Raise clear errors if anything goes wrong
5. Create a valid JSON output file

---

## Summary of Changes

| Fix | Priority | File | Lines | Risk |
|-----|----------|------|-------|------|
| #1: Metric validation | CRITICAL | optimize.py | 60-131 | Low - improves robustness |
| #2: Bootstrap params | CRITICAL | optimize.py | 293-299 | Low - fixes bug |
| #3: Shuffle data | CRITICAL | optimize.py | 135-158 | Low - improves generalization |
| #4: tool_call_format | HIGH | dspy_agent.py | 78-105 | Low - adds parameter |
| #5: Full val eval | HIGH | optimize.py | 368-372 | Low - improves metrics |
| #6: Configurable constants | MEDIUM | optimize.py | Various | Low - adds parameters |
| #7: Type hints | MEDIUM | optimize.py | Various | Very Low - documentation |

**Total lines changed**: ~150 (mostly in optimize.py)

**Risk assessment**: LOW - All changes are additive or improve error handling. No breaking changes to existing functionality.

**Testing effort**: 1-2 hours
- Run demonstrate_issues.py to verify issues exist
- Apply fixes
- Run small test optimization
- Verify output and logs

**Estimated time to apply all fixes**: 30-60 minutes

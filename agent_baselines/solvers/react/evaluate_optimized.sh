#!/bin/bash
# Compare baseline ReAct agent vs DSPy-optimized agent on sqa_dev
#
# Usage:
#   ./evaluate_optimized.sh [--limit N] [--model MODEL]
#
# Options:
#   --limit N: Limit evaluation to N samples (default: 10)
#   --model MODEL: Model to use (default: openai/gpt-4o)
#
# This script runs both the baseline and optimized agents on sqa_dev
# and compares their performance on the global_avg metric.

set -euo pipefail

# Default values
LIMIT=10
MODEL="openai/gpt-4o"
BASELINE_LOG_DIR="logs/baseline"
OPTIMIZED_LOG_DIR="logs/optimized"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--limit N] [--model MODEL]"
            exit 1
            ;;
    esac
done

echo "=================================="
echo "Evaluating ReAct Agents on SQA_dev"
echo "=================================="
echo "Model: $MODEL"
echo "Limit: $LIMIT samples"
echo ""

# Check if ASTA_TOOL_KEY is set
if [ -z "${ASTA_TOOL_KEY:-}" ]; then
    echo "WARNING: ASTA_TOOL_KEY is not set. Search tools may not work."
fi

# Check if optimized prompts exist
PROMPTS_FILE="agent_baselines/solvers/react/optimized_prompts.json"
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "ERROR: Optimized prompts not found at $PROMPTS_FILE"
    echo "Please run optimization first:"
    echo "  python agent_baselines/solvers/react/optimize.py"
    exit 1
fi

echo "Found optimized prompts at $PROMPTS_FILE"
echo ""

# Run baseline agent
echo "=================================="
echo "Running BASELINE agent..."
echo "=================================="
uv run astabench eval \
    astabench/sqa_dev \
    --solver agent_baselines/solvers/react/basic_agent.py@instantiated_basic_agent \
    --model "$MODEL" \
    --limit "$LIMIT" \
    --log-dir "$BASELINE_LOG_DIR" \
    -S max_steps=10 \
    -S with_search_tools=1 \
    -S with_report_editor=1

echo ""
echo "Baseline evaluation complete!"
echo ""

# Run optimized agent
echo "=================================="
echo "Running OPTIMIZED agent..."
echo "=================================="
uv run astabench eval \
    astabench/sqa_dev \
    --solver agent_baselines/solvers/react/optimized_agent.py@instantiated_optimized_agent \
    --model "$MODEL" \
    --limit "$LIMIT" \
    --log-dir "$OPTIMIZED_LOG_DIR" \
    -S max_steps=10 \
    -S with_search_tools=1 \
    -S with_report_editor=1

echo ""
echo "Optimized evaluation complete!"
echo ""

# Compare results
echo "=================================="
echo "Comparison Summary"
echo "=================================="
echo ""
echo "Baseline logs: $BASELINE_LOG_DIR"
echo "Optimized logs: $OPTIMIZED_LOG_DIR"
echo ""
echo "To view detailed results, use:"
echo "  uv run inspect view $BASELINE_LOG_DIR"
echo "  uv run inspect view $OPTIMIZED_LOG_DIR"
echo ""
echo "Or compare specific metrics from the logs:"
echo "  uv run inspect eval-log $BASELINE_LOG_DIR/*.json"
echo "  uv run inspect eval-log $OPTIMIZED_LOG_DIR/*.json"

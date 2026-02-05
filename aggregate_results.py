#!/usr/bin/env python3
"""
Aggregate and display results from language exploration experiments.

This script reads results.json files from experiment directories and
computes statistics across tasks and seeds.

Usage:
    # Single aggregation:
    python aggregate_results.py --results_dir ./logs/language_exploration

    # Watch mode for live monitoring:
    python aggregate_results.py --results_dir ./logs/language_exploration --watch

    # Save aggregated stats to JSON:
    python aggregate_results.py --results_dir ./logs/language_exploration --save_json stats.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np


def find_result_files(results_dir: str) -> List[Path]:
    """Find all results.json files in the results directory."""
    results_path = Path(results_dir)
    return sorted(results_path.glob("**/results.json"))


def load_result(result_path: Path) -> Optional[Dict[str, Any]]:
    """Load a single result file."""
    try:
        with open(result_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {result_path}: {e}")
        return None


def aggregate_results(results_dir: str) -> Optional[Dict[str, Any]]:
    """
    Aggregate all results from a results directory.

    Args:
        results_dir: Directory containing experiment subdirectories with results.json

    Returns:
        Aggregated statistics dictionary or None if no results found
    """
    result_files = find_result_files(results_dir)

    if not result_files:
        return None

    # Group by task
    results_by_task = defaultdict(list)

    for result_path in result_files:
        result = load_result(result_path)
        if result is None:
            continue

        task = result.get("task", "unknown")
        results_by_task[task].append(result)

    # Compute statistics
    stats = {
        "total_runs": len(result_files),
        "total_successes": 0,
        "total_failures": 0,
        "tasks": {},
        "overall": {},
    }

    all_iterations = []
    all_successes = []

    for task, task_results in sorted(results_by_task.items()):
        successes = [r for r in task_results if r.get("success", False)]
        failures = [r for r in task_results if not r.get("success", False)]

        # Iterations to success for successful runs
        iterations_to_success = []
        for r in successes:
            n_iters = r.get("total_iterations", len(r.get("history", [])) + 1)
            iterations_to_success.append(n_iters)

        successful_goals = [r.get("final_goal", "") for r in successes]

        task_stats = {
            "n_runs": len(task_results),
            "n_successes": len(successes),
            "n_failures": len(failures),
            "success_rate": len(successes) / len(task_results) if task_results else 0,
            "mean_iterations_to_success": float(np.mean(iterations_to_success)) if iterations_to_success else None,
            "std_iterations_to_success": float(np.std(iterations_to_success)) if len(iterations_to_success) > 1 else None,
            "min_iterations_to_success": int(min(iterations_to_success)) if iterations_to_success else None,
            "max_iterations_to_success": int(max(iterations_to_success)) if iterations_to_success else None,
            "successful_goals": successful_goals,
            "initial_goal": task_results[0].get("initial_goal", "") if task_results else "",
        }

        stats["tasks"][task] = task_stats
        stats["total_successes"] += len(successes)
        stats["total_failures"] += len(failures)

        all_iterations.extend(iterations_to_success)
        all_successes.extend([1] * len(successes) + [0] * len(failures))

    # Overall statistics
    stats["overall"] = {
        "success_rate": stats["total_successes"] / stats["total_runs"] if stats["total_runs"] > 0 else 0,
        "mean_iterations_to_success": float(np.mean(all_iterations)) if all_iterations else None,
        "std_iterations_to_success": float(np.std(all_iterations)) if len(all_iterations) > 1 else None,
    }

    return stats


def print_stats(stats: Optional[Dict[str, Any]], verbose: bool = True):
    """Print statistics in a formatted way."""
    if stats is None:
        print("No results found.")
        return

    print("\n" + "=" * 80)
    print("LANGUAGE EXPLORATION EXPERIMENT RESULTS")
    print("=" * 80)

    # Overall summary
    print(f"\n{'OVERALL SUMMARY':^80}")
    print("-" * 80)
    print(f"Total runs:       {stats['total_runs']}")
    print(f"Total successes:  {stats['total_successes']}")
    print(f"Total failures:   {stats['total_failures']}")
    print(f"Success rate:     {stats['overall']['success_rate']:.1%}")

    if stats['overall']['mean_iterations_to_success'] is not None:
        mean_iter = stats['overall']['mean_iterations_to_success']
        std_iter = stats['overall']['std_iterations_to_success']
        if std_iter is not None:
            print(f"Iterations to success: {mean_iter:.2f} ± {std_iter:.2f}")
        else:
            print(f"Iterations to success: {mean_iter:.2f}")

    # Per-task breakdown
    print(f"\n{'PER-TASK BREAKDOWN':^80}")
    print("-" * 80)
    print(f"{'Task':<20} {'Runs':<6} {'Success':<8} {'Rate':<8} {'Iters (mean±std)':<20} {'Min/Max':<10}")
    print("-" * 80)

    for task, task_stats in sorted(stats["tasks"].items()):
        rate = f"{task_stats['success_rate']:.0%}"

        if task_stats['mean_iterations_to_success'] is not None:
            mean = task_stats['mean_iterations_to_success']
            std = task_stats['std_iterations_to_success']
            if std is not None:
                iters_str = f"{mean:.1f} ± {std:.1f}"
            else:
                iters_str = f"{mean:.1f}"
            min_max = f"{task_stats['min_iterations_to_success']}/{task_stats['max_iterations_to_success']}"
        else:
            iters_str = "N/A"
            min_max = "N/A"

        print(f"{task:<20} {task_stats['n_runs']:<6} {task_stats['n_successes']:<8} {rate:<8} {iters_str:<20} {min_max:<10}")

    # Successful goals (verbose mode)
    if verbose:
        print(f"\n{'SUCCESSFUL GOALS BY TASK':^80}")
        print("-" * 80)

        for task, task_stats in sorted(stats["tasks"].items()):
            if task_stats['successful_goals']:
                initial = task_stats['initial_goal']
                initial_preview = initial[:50] + "..." if len(initial) > 50 else initial
                print(f"\nTask {task} (initial: '{initial_preview}'):")

                # Count unique goals
                goal_counts = defaultdict(int)
                for goal in task_stats['successful_goals']:
                    goal_counts[goal] += 1

                for goal, count in sorted(goal_counts.items(), key=lambda x: -x[1]):
                    goal_preview = goal[:70] + "..." if len(goal) > 70 else goal
                    print(f"  [{count}x] {goal_preview}")

    print("\n" + "=" * 80)


def watch_results(results_dir: str, interval: float = 10.0):
    """Continuously watch and update results."""
    print(f"Watching {results_dir} for results (Ctrl+C to stop)...")
    print(f"Refreshing every {interval} seconds\n")

    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')

            stats = aggregate_results(results_dir)
            print_stats(stats, verbose=False)

            print(f"\n[Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}]")
            print(f"[Watching: {results_dir}]")
            print("[Press Ctrl+C to stop]")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped watching.")


def save_stats_json(stats: Dict[str, Any], output_path: str):
    """Save aggregated statistics to JSON."""
    def convert(obj):
        """Convert numpy types to Python types."""
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    stats_converted = convert(stats)

    with open(output_path, 'w') as f:
        json.dump(stats_converted, f, indent=2)

    print(f"Saved aggregated statistics to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate language exploration results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch and update results",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Refresh interval in seconds (for --watch mode)",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Save aggregated statistics to JSON file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show summary, not detailed goals",
    )

    args = parser.parse_args()

    if args.watch:
        watch_results(args.results_dir, args.interval)
    else:
        stats = aggregate_results(args.results_dir)
        print_stats(stats, verbose=not args.quiet)

        if args.save_json and stats:
            save_stats_json(stats, args.save_json)


if __name__ == "__main__":
    main()

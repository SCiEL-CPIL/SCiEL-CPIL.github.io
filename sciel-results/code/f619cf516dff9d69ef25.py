"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard variables
_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

def _time_exceeded():
    """Check if we are approaching the time limit."""
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def dispatch(ready_operations, time):
    # Explicit deadline guard
    if _time_exceeded():
        if not ready_operations:
            return 0
        return 0

    if not ready_operations:
        return 0

    best_score = -1.0
    best_index = 0
    best_machine = None

    # Pre-compute normalization constants
    max_remaining_work = max(op.get("remaining_work", 0.0) for op in ready_operations)
    if max_remaining_work <= 0:
        max_remaining_work = 1.0

    max_remaining_ops = max(op.get("remaining_ops", 0) for op in ready_operations)
    if max_remaining_ops <= 0:
        max_remaining_ops = 1

    for i, op in enumerate(ready_operations):
        # Check deadline again in case of extremely large input (unlikely here)
        if _time_exceeded():
            break

        remaining_work = op.get("remaining_work", 0.0)
        machine_options = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)

        if not machine_options:
            score = 0.0
            best_machine_local = 0
        else:
            # Find the machine with the minimum duration for this operation
            min_duration = float('inf')
            best_machine_local = next(iter(machine_options.keys()))

            for m, dur in machine_options.items():
                if dur < min_duration:
                    min_duration = dur
                    best_machine_local = m

            if min_duration <= 0:
                min_duration = 1.0

            # Strategy from DIRECTIVE:
            # Priority is inversely proportional to the minimum available processing time.
            # This prioritizes operations that can be executed most efficiently.
            
            # Efficiency score: 1/min_duration
            # Shorter processing times get higher priority
            eff_score = 1.0 / min_duration

            # Normalize remaining work for criticality
            norm_work = remaining_work / max_remaining_work

            # Tail score: prioritize jobs closer to completion
            if remaining_ops > 0:
                tail_score = 1.0 / remaining_ops
            else:
                tail_score = 0.0

            # Combined score with emphasis on efficiency:
            # - 0.5 * efficiency: heavily favor short-duration operations
            # - 0.3 * criticality: consider remaining workload
            # - 0.2 * tail: help finish jobs with few remaining ops
            score = 0.5 * eff_score + 0.3 * norm_work + 0.2 * tail_score

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_machine_local

    # Return the index and the explicitly selected best machine
    return (int(best_index), best_machine)
# EVOLVE-BLOCK-END

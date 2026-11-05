"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random
import math

_DEADLINE = None
_DEADLINE_LIMIT = 145.0  # seconds wall-clock safety margin (evaluator kills at 150s)


def _elapsed():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = _time.monotonic() + _DEADLINE_LIMIT
    return _time.monotonic() - _DEADLINE


def _remaining_time():
    return _DEADLINE_LIMIT - _elapsed()


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Strategy: Prioritize operations with highest remaining work (critical path),
    with tie-breaking by shortest processing time. Returns (index, machine)
    to guide machine selection for earliest completion.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If only one option, return it with its best machine
    if n == 1:
        op = ready_operations[0]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options, key=machine_options.get)
            return (0, best_machine)
        return 0
    
    best_score = -1e18
    best_idx = 0
    best_machine = None
    
    # Find max remaining work for normalization
    max_rw = 0
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        if rw > max_rw:
            max_rw = rw
    
    norm_rw = max_rw if max_rw > 0 else 1.0
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            score = 0
            best_machine_for_op = None
        else:
            # Select machine with shortest duration (earliest completion)
            best_machine_for_op = min(machine_options, key=machine_options.get)
            min_duration = machine_options[best_machine_for_op]
            
            # Score components:
            # 1. Critical path: remaining work (normalized) - dominant factor
            cp_score = remaining_work / norm_rw
            
            # 2. Urgency: operations closer to completion get slight boost
            urgency = 1.0 / (1.0 + remaining_ops)
            
            # 3. Shorter processing time reduces blocking
            time_bonus = 1.0 / (1.0 + min_duration * 0.1)
            
            # 4. Deterministic tiebreaker
            job = op.get("job", 0)
            op_num = op.get("op", 0)
            tie_break = ((job * 31 + op_num * 17) % 97) * 0.0001
            
            # Weighted combination - critical path dominates
            score = (
                0.70 * cp_score +
                0.15 * urgency +
                0.15 * time_bonus +
                tie_break
            )
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op
    
    # Return (index, machine) to guide machine selection
    if best_machine is not None:
        return (int(best_idx), best_machine)
    return int(best_idx)
# EVOLVE-BLOCK-END

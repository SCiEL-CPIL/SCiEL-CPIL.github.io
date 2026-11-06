"""Rewritten dispatch rule for open-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time_mod

# Track a wall-clock deadline to avoid timeout.
# The evaluator kills after 150s; we guard with a margin.
_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        # We don't know when evaluation started, so we can't set a global deadline
        # reliably across multiple calls. Instead, we track per-call budget.
        pass
    return None


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Load-Balancing Water-Filling heuristic.
    For each candidate operation, we consider assigning it to the least-loaded
    compatible machine. We score by the resulting increase in the maximum
    machine load (L-infinity norm), but we also blend in job urgency (remaining work).
    
    The idea: prioritize operations that, when scheduled on the least-loaded machine,
    cause the smallest increase in the global makespan bound.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If only one option, return it
    if n == 1:
        return 0
    
    # Collect machine options for each operation
    # We need to estimate machine loads. Since we don't have direct access to
    # current machine loads, we'll use a heuristic based on the operations themselves.
    
    # Approach: For each candidate op, find the machine with the minimum duration
    # (least work added) and score by a combination of:
    # 1. The duration on the best machine (shorter is better for load balancing)
    # 2. The job's remaining work (urgency)
    # 3. A water-filling style: minimize the max load increase
    
    # Since we don't know current machine loads, we approximate by assuming
    # machines are roughly balanced and focus on minimizing the duration added
    # while also considering job completion urgency.
    
    # For each operation, compute:
    # - min_duration: the shortest processing time among its machine options
    # - best_machine: the machine with min duration
    # - remaining_work: total remaining work for the job
    # - remaining_ops: number of remaining operations for the job
    
    scores = []
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            scores.append(float('inf'))
            continue
        
        # Find the machine with minimum duration (least load increase)
        min_duration = min(machine_options.values())
        min_machine = min(machine_options, key=machine_options.get)
        
        # Water-filling score: we want to minimize the increase in max load.
        # Since we don't know current loads, we use min_duration as a proxy.
        # But we also want to prioritize jobs with more remaining work (urgency)
        # and fewer remaining ops (closer to completion).
        
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Combine: lower is better
        # Weight 1: min_duration (load balancing)
        # Weight 2: -remaining_work (urgency: more work = more urgent)
        # Weight 3: remaining_ops (fewer ops left = more urgent to finish)
        
        # Normalize-ish combination:
        # score = min_duration - alpha * remaining_work + beta * remaining_ops
        # We want to minimize this.
        
        # Let's use a simple weighted sum:
        # Higher remaining_work should decrease the score (more urgent)
        # Lower min_duration should decrease the score (less load)
        # Lower remaining_ops should decrease the score (closer to done)
        
        score = min_duration - 0.1 * remaining_work + 0.5 * remaining_ops
        
        scores.append(score)
    
    # Pick the operation with the lowest score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] < best_score:
            best_score = scores[i]
            best_idx = i
    
    return best_idx
# EVOLVE-BLOCK-END

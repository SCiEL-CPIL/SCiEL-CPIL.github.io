"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = None
_START = None

def _ensure_deadline():
    global _DEADLINE, _START
    if _DEADLINE is None:
        _START = _time.monotonic()
        _DEADLINE = _START + 145.0  # safety margin before 150s kill
    return _DEADLINE

def _time_left():
    return max(0.0, _DEADLINE - _time.monotonic())


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Strategy: 
    1. Select operation based on a weighted score of Remaining Work (MWR) and Machine Scarcity.
    2. Select the machine with the Shortest Processing Time (SPT) for that operation.
    3. Return (index, machine).
    
    This combines the critical-path clearing benefit of MWR with the load-reduction benefit of SPT machine assignment.
    """
    deadline = _ensure_deadline()
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        # If only one op, pick it and its best machine
        op = ready_operations[0]
        # Find machine with min duration
        best_machine = min(op["machine_options"], key=lambda m: op["machine_options"][m])
        return (0, best_machine)
    
    # Fast path: if we're close to deadline, use simple heuristic
    if _time_left() < 0.5:
        # Simple: Max remaining work, min duration machine
        best_idx = int(np.argmax([op["remaining_work"] for op in ready_operations]))
        op = ready_operations[best_idx]
        best_machine = min(op["machine_options"], key=lambda m: op["machine_options"][m])
        return (best_idx, best_machine)
    
    scores = np.zeros(n)
    best_machines = [None] * n
    min_durations = [0.0] * n
    
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        machine_opts = op["machine_options"]
        
        # 1. Machine Scarcity: Fewer options = more constrained
        num_options = len(machine_opts)
        scarcity = 1.0 / max(num_options, 1)
        
        # 2. Duration metrics
        durations = list(machine_opts.values())
        min_duration = min(durations)
        avg_duration = sum(durations) / len(durations)
        
        # Store best machine (SPT) for this operation
        best_machine = min(machine_opts, key=lambda m: machine_opts[m])
        best_machines[i] = best_machine
        min_durations[i] = min_duration
        
        # Composite score for OPERATION selection:
        # - Dominant factor: Remaining Work (MWR). Jobs with more work should be processed to clear the critical path.
        # - Secondary factor: Scarcity. Operations with fewer machine options are more urgent to schedule to avoid deadlock/bottlenecks.
        # - Tertiary factor: Min Duration. If remaining work is similar, prefer shorter operations to free up resources quickly.
        
        # Weights:
        # rem_work is usually in the range of total job time.
        # scarcity is 1/num_options.
        # min_duration is processing time.
        
        # To balance scales, we normalize or use small weights for secondary factors.
        # Since rem_work is the most important for makespan in open shop, give it weight 1.0.
        # Scarcity: 0.1 * scarcity * rem_work (scaled by work to be significant only if work is high? or just 0.1 * scarcity?)
        # Let's use a direct additive model with tuned weights.
        
        # Heuristic:
        # score = rem_work + 0.5 * scarcity * rem_work + 0.01 * min_duration
        # The 0.01 * min_duration ensures that if rem_work is identical, shorter ops are preferred.
        # The 0.5 * scarcity * rem_work boosts constrained ops that also have significant work.
        
        score = (
            1.0 * rem_work +
            0.5 * scarcity * rem_work +
            0.01 * min_duration
        )
        scores[i] = score
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END

"""Open-Shop Scheduling dispatch rule for open-shop event-driven simulation."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE_MARGIN = 10  # seconds before the 120s kill; be conservative


def _elapsed() -> float:
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. Calculate a "priority score" for each ready operation.
    2. The score is based on the ratio of the job's remaining work to the 
       minimum processing time of that operation on any available machine.
       Score = remaining_work / min_duration
    3. This prioritizes operations that:
       - Belong to jobs with significant remaining work (critical path).
       - Can be processed quickly on some machine (reduces machine occupancy).
    4. For the selected operation, assign it to the machine with the minimum 
       duration to ensure it finishes as soon as possible.
    
    Return: (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, best_d = _best_machine_for_op(op)
        return (0, best_m)

    scores = []
    best_machines = []
    
    for op in ready_operations:
        rw = op["remaining_work"]
        durations = op["machine_options"]
        
        if not durations:
            scores.append(0)
            best_machines.append(None)
            continue
            
        durs = list(durations.values())
        min_dur = min(durs)
        
        # Avoid division by zero
        if min_dur <= 0:
            min_dur = 1e-9
            
        # Score: High remaining work, Low duration is good.
        # Using ratio emphasizes efficiency.
        score = rw / min_dur
        
        scores.append(score)
        best_m, _ = _best_machine_for_op(op)
        best_machines.append(best_m)

    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    return (best_idx, best_machine)


def _best_machine_for_op(op):
    """
    For a given operation, select the machine that minimizes its duration.
    Returns (machine, duration).
    """
    durations = op["machine_options"]
    if not durations:
        return None, 0
    best_machine = min(durations, key=lambda m: durations[m])
    return best_machine, durations[best_machine]
# EVOLVE-BLOCK-END

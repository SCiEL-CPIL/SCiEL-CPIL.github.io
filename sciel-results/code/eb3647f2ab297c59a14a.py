"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    Returns (index, machine) to allow the simulator to place the operation
    on the fastest available machine, and uses a composite score that balances:
      - remaining work (urgency)
      - number of remaining ops (tail length)
      - flexibility (number of machine options)
      - shortest processing time on the best machine
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        # Pick the machine with the shortest duration
        best_machine = min(op["machine_options"].items(), key=lambda x: x[1])[0]
        return (0, best_machine)

    scores = []
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        durations = list(mo.values())
        min_dur = min(durations)
        max_dur = max(durations)
        n_opts = len(mo)
        
        # Remaining work and ops
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        
        # Flexibility: more options is better
        flexibility = math.log(n_opts + 1)
        
        # Urgency: more remaining work is worse (should be scheduled sooner)
        urgency = rw
        
        # Tail: more remaining ops means longer tail
        tail = ro
        
        # Speed: shorter min duration is better
        speed_bonus = -min_dur
        
        # Composite score: we want to maximize this
        # Weighted combination
        score = (
            1.0 * urgency +
            0.5 * tail +
            0.3 * flexibility +
            0.2 * speed_bonus
        )
        scores.append(score)
    
    scores = np.array(scores)
    best_idx = int(np.argmax(scores))
    
    # Pick the machine with the shortest duration for the chosen operation
    op = ready_operations[best_idx]
    best_machine = min(op["machine_options"].items(), key=lambda x: x[1])[0]
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END

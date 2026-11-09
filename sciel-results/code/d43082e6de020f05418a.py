import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Calculate min duration and best machine for each op
    min_durs = []
    for op in ready_operations:
        opts = op["machine_options"]
        if not opts:
            min_durs.append(0.0)
        else:
            min_durs.append(min(opts.values()))
    min_durs = np.array(min_durs)

    # SPT: Shortest Processing Time
    # Pick the operation with the shortest duration to clear the machine quickly.
    # This is a strong heuristic for open shop to reduce makespan.
    best_idx = int(np.argmin(min_durs))
    
    # Tie breaker: if durations are equal, pick the one with most remaining work (MWRF)
    # or most remaining ops to balance load.
    # Let's check if there are ties in min_durs.
    min_val = min_durs[best_idx]
    ties = np.where(min_durs == min_val)[0]
    if len(ties) > 1:
        # Among ties, pick the one with highest remaining_work
        works = np.array([op["remaining_work"] for op in ready_operations])
        tie_works = works[ties]
        best_tie_idx = ties[np.argmax(tie_works)]
        return int(best_tie_idx)
        
    return best_idx

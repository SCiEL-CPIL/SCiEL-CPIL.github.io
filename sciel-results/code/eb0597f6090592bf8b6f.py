def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    best_idx = 0
    best_min_dur = float('inf')
    best_work = -1
    
    for i, op in enumerate(ready_operations):
        if op["machine_options"]:
            min_dur = min(op["machine_options"].values())
        else:
            min_dur = 0
            
        work = op["remaining_work"]
        
        # Primary: Shortest Processing Time
        # Secondary: Most Work Remaining
        if min_dur < best_min_dur or (min_dur == best_min_dur and work > best_work):
            best_min_dur = min_dur
            best_work = work
            best_idx = i
            
    return best_idx

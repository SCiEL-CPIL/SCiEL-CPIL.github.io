<<<<<<< SEARCH
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 108.0  # Safety margin for 120s limit
    
    # 1. Initialization - try multiple strategies
    rows = _greedy_init(n_rows, k)
    all_used = _compute_all_used(rows)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Try greedy with target to see if we can do better
    greedy_target_max = best_val
    for _ in range(3):
        rows2 = _greedy_init_target(n_rows, k, greedy_target_max)
        val2 = _global_max(rows2)
        if val2 < best_val:
            best_rows = [r[:] for r in rows2]
            best_val = val2
            greedy_target_max = best_val
    
    # Maintain row maxes for efficient global max computation
    row_maxes = [max(r) for r in rows]
=======
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Initialization - try multiple strategies
    rows = _greedy_init(n_rows, k)
    all_used = _compute_all_used(rows)
    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)
    
    # Try greedy with target to see if we can do better
    greedy_target_max = best_val
    for _ in range(3):
        rows2 = _greedy_init_target(n_rows, k, greedy_target_max)
        val2 = _global_max(rows2)
        if val2 < best_val:
            best_rows = [r[:] for r in rows2]
            best_val = val2
            greedy_target_max = best_val
    
    # Maintain row maxes for efficient global max computation
    row_maxes = [max(r) for r in rows]
>>>>>>> REPLACE


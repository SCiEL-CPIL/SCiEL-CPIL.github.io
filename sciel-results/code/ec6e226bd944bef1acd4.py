<<<<<<< SEARCH
def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 150s kill
    
    best_p = None
    best_cost = np.inf
    
    # Multi-start portfolio
    # Start 1: Rank-matching
    # Start 2: Random
    # Start 3: Heavy-flow greedy
    
    initializations = []
    
    # 1. Rank-matching
    p1 = _greedy_rank_init(a, b, seed=42)
    initializations.append(('rank', p1))
    
    # 2. Random
    p2 = _random_init(n, seed=123)
    initializations.append(('random', p2))
    
    # 3. Heavy-flow greedy (slower init but potentially better)
    if time.time() - start_time < time_limit - 30:
        p3 = _heavy_flow_greedy_init(a, b, seed=77)
        initializations.append(('heavy', p3))
    
    # Allocate time to each start
    num_starts = len(initializations)
    time_per_start = (time_limit - 10) / num_starts  # Leave 10s buffer
    
    for idx, (name, p_init) in enumerate(initializations):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 15:
            break
        
        alloc_time = min(time_per_start, remaining - 5)
        if alloc_time < 10:
            break
        
        p_best, cost_best = _pairwise_facility_exchange_search(
            a, b, p_init.copy(), alloc_time, start_time, seed=1000 + idx
        )
        
        if cost_best < best_cost:
            best_cost = cost_best
            best_p = p_best
    
    if best_p is None:
        best_p = _greedy_rank_init(a, b, seed=0)
    
    return best_p.tolist()
=======
def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    best_p = None
    best_cost = np.inf
    
    # Multi-start portfolio
    initializations = []
    
    # 1. Rank-matching
    p1 = _greedy_rank_init(a, b, seed=42)
    initializations.append(('rank', p1))
    
    # 2. Random
    p2 = _random_init(n, seed=123)
    initializations.append(('random', p2))
    
    # 3. Heavy-flow greedy (slower init but potentially better)
    if time.time() - start_time < time_limit - 40:
        p3 = _heavy_flow_greedy_init(a, b, seed=77)
        initializations.append(('heavy', p3))
    
    # Allocate time to each start
    num_starts = len(initializations)
    time_per_start = (time_limit - 15) / num_starts  # Leave 15s buffer
    
    for idx, (name, p_init) in enumerate(initializations):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 20:
            break
        
        alloc_time = min(time_per_start, remaining - 8)
        if alloc_time < 12:
            break
        
        p_best, cost_best = _pairwise_facility_exchange_search(
            a, b, p_init.copy(), alloc_time, start_time, seed=1000 + idx
        )
        
        if cost_best < best_cost:
            best_cost = cost_best
            best_p = p_best
    
    if best_p is None:
        best_p = _greedy_rank_init(a, b, seed=0)
    
    return best_p.tolist()
>>>>>>> REPLACE

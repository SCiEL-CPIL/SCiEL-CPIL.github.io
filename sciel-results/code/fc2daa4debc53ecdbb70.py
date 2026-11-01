<<<<<<< SEARCH
def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115
    
    if k != 3:
        q = k
        n, edges = _luw_construct(q, g)
        return n, edges
    
    # Moore bound for k=3, girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    best_result = None
    best_n = float('inf')
    
    # Strategy 1: Try voltage lifts with both Petersen and K3,3 bases
    # K3,3 lifts produce graphs of size 6m, Petersen lifts produce 10m
    # For the same n, K3,3 allows smaller m values
    
    # For Petersen: n = 10m, so m = n/10
    # For K3,3: n = 6m, so m = n/6
    
    # Start from the Moore bound
    m_min_petersen = max(1, (moore + 9) // 10)
    m_min_k33 = max(1, (moore + 5) // 6)
    
    # Try K3,3 lifts first (smaller m values for same n)
    for m in range(m_min_k33, m_min_k33 + 300):
        if _time.time() > deadline - 10:
            break
        n = 6 * m
        if n < moore:
            continue
        if n >= best_n:
            break
        
        # Try deterministic primitive root voltage
        result = _deterministic_voltage_lift_k33(m, g)
        if result is not None:
            n_lift, edges = result
            if n_lift < best_n:
                best_n = n_lift
                best_result = (n_lift, edges)
            break
        
        # Try random voltage assignments
        for attempt in range(30):
            if _time.time() > deadline - 10:
                break
            result = _random_voltage_lift_k33(m, g, seed=attempt * 7919 + m * 13)
            if result is not None:
                n_lift, edges = result
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                break
    
    # Try Petersen lifts
    for m in range(m_min_petersen, m_min_petersen + 200):
        if _time.time() > deadline - 10:
            break
        n = 10 * m
        if n < moore:
            continue
        if n >= best_n:
            break
        
        result = _deterministic_voltage_lift_petersen(m, g)
        if result is not None:
            n_lift, edges = result
            if n_lift < best_n:
                best_n = n_lift
                best_result = (n_lift, edges)
            break
        
        for attempt in range(20):
            if _time.time() > deadline - 10:
                break
            result = _random_voltage_lift_petersen(m, g, seed=attempt * 7919 + m * 13)
            if result is not None:
                n_lift, edges = result
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                break
    
    # Strategy 2: Try LUW construction as a strong baseline
    if _time.time() < deadline - 15:
        n_luw, edges_luw = _luw_construct(3, g)
        if n_luw < best_n:
            best_n = n_luw
            best_result = (n_luw, edges_luw)
    
    # Strategy 3: Random 3-regular graphs near the Moore bound
    if _time.time() < deadline - 15 and best_result is None:
        n_try = moore
        if n_try % 2 != 0:
            n_try += 1
        while n_try < 5000 and _time.time() < deadline - 10:
            for attempt in range(30):
                if _time.time() > deadline - 10:
                    break
                result = _random_3reg(n_try, g, seed=attempt * 104729 + n_try)
                if result is not None:
                    n_r, edges_r = result
                    if n_r < best_n:
                        best_n = n_r
                        best_result = (n_r, edges_r)
                    break
            n_try += 32
    
    # Strategy 4: More extensive random search
    if _time.time() < deadline - 10 and best_result is None:
        for n_try in range(moore, moore + 200, 2):
            if _time.time() > deadline - 10:
                break
            for attempt in range(100):
                if _time.time() > deadline - 5:
                    break
                result = _random_3reg(n_try, g, seed=attempt * 31 + n_try * 17)
                if result is not None:
                    best_result = result
                    break
            if best_result is not None:
                break
    
    # Fallback: LUW construction
    if best_result is None:
        n_luw, edges_luw = _luw_construct(3, g)
        best_result = (n_luw, edges_luw)
    
    return best_result
# EVOLVE-BLOCK-END
=======
def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 140
    
    if k != 3:
        q = k
        n, edges = _luw_construct(q, g)
        return n, edges
    
    # Moore bound for k=3, girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    best_result = None
    best_n = float('inf')
    
    # Strategy 1: LUW construction as a strong baseline (fast, deterministic)
    n_luw, edges_luw = _luw_construct(3, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)
    
    # Strategy 2: Greedy cage construction - try to find smaller graphs
    # The greedy approach searches near the Moore bound
    greedy_result = _greedy_cage_build(g, best_n - 1, seed=42, deadline=deadline - 10)
    if greedy_result is not None:
        n_greedy, edges_greedy = greedy_result
        if n_greedy < best_n:
            best_n = n_greedy
            best_result = (n_greedy, edges_greedy)
    
    # Strategy 3: More extensive greedy search with different seeds and sizes
    if _time.time() < deadline - 15:
        # Try multiple sizes with more attempts
        n_target = moore
        if n_target % 2 != 0:
            n_target += 1
        
        for offset in range(0, min(best_n - n_target + 1, 200), 2):
            if _time.time() > deadline - 10:
                break
            n_try = n_target + offset
            if n_try >= best_n:
                break
            
            # Try with multiple seeds
            for seed in range(0, 50, 1):
                if _time.time() > deadline - 10:
                    break
                result = _greedy_single_attempt(n_try, g, seed=seed * 1000003 + n_try, deadline=deadline - 10)
                if result is not None:
                    n_found, edges_found = result
                    if n_found < best_n:
                        best_n = n_found
                        best_result = (n_found, edges_found)
                    break  # Found one at this size, move to smaller size
    
    # Strategy 4: If we have time, try even more random constructions
    if _time.time() < deadline - 10 and best_n > moore + 50:
        n_try = moore
        if n_try % 2 != 0:
            n_try += 1
        while n_try < best_n - 10 and _time.time() < deadline - 5:
            for attempt in range(100):
                if _time.time() > deadline - 5:
                    break
                result = _greedy_single_attempt(n_try, g, seed=attempt * 999983 + n_try * 7, deadline=deadline - 5)
                if result is not None:
                    n_r, edges_r = result
                    if n_r < best_n:
                        best_n = n_r
                        best_result = (n_r, edges_r)
                    break
            n_try += 2
    
    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

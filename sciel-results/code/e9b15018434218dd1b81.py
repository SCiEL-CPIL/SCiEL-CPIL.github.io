<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    baseline_cycle = list(range(_N))
    baseline_score = _score_cycle(baseline_cycle)
    best_cycle = baseline_cycle
    best_score = baseline_score
    
    # Binary search for minimum K
    lo = 24
    hi = best_score
    
    while lo < hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        cycle = _check_feasibility(mid, time_limit=remaining - 1, start_time=start_time)
        if cycle is not None:
            if len(set(cycle)) == _N and len(cycle) == _N:
                s = _score_cycle(cycle)
                if s < best_score:
                    best_cycle = cycle
                    best_score = s
                hi = mid
            else:
                lo = mid + 1
        else:
            lo = mid + 1
    
    # Final verification
    if time.time() < deadline - 2:
        remaining = deadline - time.time()
        if remaining > 1:
            cycle = _check_feasibility(lo, time_limit=remaining - 0.5, start_time=start_time)
            if cycle is not None and len(set(cycle)) == _N:
                s = _score_cycle(cycle)
                if s < best_score:
                    best_cycle = cycle
                    best_score = s
    
    return best_cycle
=======
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    baseline_cycle = list(range(_N))
    baseline_score = _score_cycle(baseline_cycle)
    best_cycle = baseline_cycle
    best_score = baseline_score
    
    # Binary search for minimum K
    lo = 24
    hi = best_score
    
    while lo < hi and time.time() < deadline - 10:
        mid = (lo + hi) // 2
        remaining = deadline - time.time()
        if remaining < 15:
            break
        
        cycle = _check_feasibility(mid, time_limit=min(remaining - 5, 60), start_time=start_time)
        if cycle is not None:
            if len(set(cycle)) == _N and len(cycle) == _N:
                s = _score_cycle(cycle)
                if s < best_score:
                    best_cycle = cycle
                    best_score = s
                hi = mid
            else:
                lo = mid + 1
        else:
            lo = mid + 1
    
    # Final verification at lo
    if time.time() < deadline - 5:
        remaining = deadline - time.time()
        if remaining > 10:
            cycle = _check_feasibility(lo, time_limit=min(remaining - 3, 60), start_time=start_time)
            if cycle is not None and len(set(cycle)) == _N:
                s = _score_cycle(cycle)
                if s < best_score:
                    best_cycle = cycle
                    best_score = s
    
    return best_cycle
>>>>>>> REPLACE

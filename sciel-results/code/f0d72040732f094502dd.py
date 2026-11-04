def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Skip rank 27 attempt due to instability/bugs in heuristic search
    # rank_27 = _try_rank_27(deadline)
    # if rank_27 is not None:
    #     return rank_27
        
    # Fallback to Rank 28
    return _strassen_rank28(n, m, p)

def construct(q, n, radius):
    best_code = None
    best_size = float('inf')
    
    if q == 3 and n == 6 and radius == 1:
        greedy_res = _greedy_covering(q, n, radius)
        if greedy_res:
            if len(greedy_res) < best_size:
                best_size = len(greedy_res)
                best_code = greedy_res
    
    # Fallback 1
    if q == 3 and n >= 5:
        proj_res = _q3_tetracode_projected(n)
        if len(proj_res) < best_size:
            best_size = len(proj_res)
            best_code = proj_res
            
    # Fallback 2
    direct_res = _direct_sum(q, n, radius)
    if len(direct_res) < best_size:
        best_size = len(direct_res)
        best_code = direct_res
        
    return best_code

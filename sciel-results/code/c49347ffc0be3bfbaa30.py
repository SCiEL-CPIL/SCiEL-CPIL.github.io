<<<<<<< SEARCH
def _rank_matching_init(a, b):
    """
    Simple rank-matching initialization based on row sums.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p
=======
def _rank_matching_init(a, b):
    """
    Simple rank-matching initialization based on row sums.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _greedy_init(a, b, seed=42):
    """
    Greedy initialization: iteratively place the facility with highest remaining flow
    to the location that minimizes incremental cost.
    """
    n = len(a)
    rng = np.random.default_rng(seed)
    
    p = np.full(n, -1, dtype=np.int64)
    used_locs = np.zeros(n, dtype=bool)
    
    # Sort facilities by total flow (descending)
    facility_order = np.argsort(-a.sum(axis=1), kind='stable')
    
    for fac in facility_order:
        if fac < 0 or fac >= n:
            continue
        if p[fac] >= 0:
            continue
            
        # Find best unused location for this facility
        best_loc = -1
        best_cost = np.inf
        
        for loc in range(n):
            if used_locs[loc]:
                continue
            
            # Compute incremental cost of placing fac at loc
            inc_cost = 0
            for other_fac in range(n):
                if p[other_fac] < 0:
                    continue
                other_loc = p[other_fac]
                inc_cost += a[fac, other_fac] * b[loc, other_loc] + a[other_fac, fac] * b[other_loc, loc]
            
            if inc_cost < best_cost:
                best_cost = inc_cost
                best_loc = loc
        
        if best_loc >= 0:
            p[fac] = best_loc
            used_locs[best_loc] = True
    
    # Fill any remaining unassigned
    remaining_facs = [i for i in range(n) if p[i] < 0]
    remaining_locs = [j for j in range(n) if not used_locs[j]]
    for fac, loc in zip(remaining_facs, remaining_locs):
        p[fac] = loc
    
    return p
>>>>>>> REPLACE


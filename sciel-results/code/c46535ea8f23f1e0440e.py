"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _rank_matching_start(a, b, rng):
    """Rank-matching heuristic: heavy flow rows to central distance rows."""
    n = len(a)
    # Row sums of flow matrix
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    # Sort facilities by decreasing flow
    heavy_order = np.argsort(-flow_sums, kind="stable")
    # Sort locations by increasing distance (central first)
    central_order = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    return p


def _hungarian_start(a, b, rng):
    """
    Formulate as a relaxed assignment:
    We want to assign facility i to location p[i].
    
    A common relaxation: minimize sum_i c[i, p[i]] where c[i, j] approximates
    the contribution of placing facility i at location j.
    
    A good surrogate: c[i, j] = sum_k a[i, k] * b[j, p[k]] -- but p is unknown.
    
    Instead, use a two-stage approach:
    1. Compute "attractiveness" of each location for each facility based on
       flow vector of facility and distance row of location.
    2. Use Hungarian to find best assignment.
    
    For the cost matrix, use: C[i, j] = dot(a[i, :], b[j, :]) as a proxy.
    This captures the interaction between facility i's flow profile and
    location j's distance profile.
    """
    n = len(a)
    # C[i, j] = sum_k a[i,k] * b[j,k] -- this is the "alignment" cost
    # We want to MINIMIZE the total QAP cost, and high alignment means
    # facility i's flows align with location j's distances, which is bad
    # if we want heavy flows on short distances. So we want to MINIMIZE
    # sum_i C[i, p[i]]? No -- actually we want heavy flows to go to short
    # distances, meaning for facility i with large flow to k, we want p[k]
    # to be close to p[i]. This is complex.
    
    # Alternative: use row-sum based cost
    # C[i, j] = a.sum(axis=1)[i] * b.sum(axis=1)[j]
    # This pushes high-flow facilities to low-distance locations.
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    C = np.outer(flow_sums, dist_sums)
    
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _local_search_2opt(a, b, p, max_iter=50000, time_limit=None, start_time=None):
    """
    2-opt local search: swap pairs of facility locations and accept if cost decreases.
    """
    n = len(a)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    
    if start_time is None:
        start_time = time.time()
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Try all pairs (i, j) with i < j
        # To speed up, we can sample or use a systematic sweep
        # For n=150, 150*149/2 = 11175 pairs per sweep
        for i in range(n):
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                if time_limit is not None and time.time() - start_time > time_limit:
                    break
                
                # Compute delta cost of swapping p[i] and p[j]
                # New positions: p[i] <-> p[j]
                loc_i = p[i]
                loc_j = p[j]
                
                # The cost change from swapping locations of facilities i and j
                # delta = sum over all k != i,j of:
                #   a[i,k]*(b[loc_j,p[k]] - b[loc_i,p[k]]) + a[k,i]*(b[p[k],loc_j] - b[p[k],loc_i])
                #   + a[j,k]*(b[loc_i,p[k]] - b[loc_j,p[k]]) + a[k,j]*(b[p[k],loc_i] - b[p[k],loc_j])
                # + a[i,j]*(b[loc_j,loc_i] - b[loc_i,loc_j])
                # + a[j,i]*(b[loc_i,loc_j] - b[loc_j,loc_i])
                # 
                # Since b is symmetric (distances), b[loc_j,loc_i] = b[loc_i,loc_j], so the
                # cross terms a[i,j] and a[j,i] cancel.
                # 
                # delta = sum_{k != i,j} [
                #   a[i,k]*(b[loc_j,p[k]] - b[loc_i,p[k]]) + a[k,i]*(b[p[k],loc_j] - b[p[k],loc_i])
                #   + a[j,k]*(b[loc_i,p[k]] - b[loc_j,p[k]]) + a[k,j]*(b[p[k],loc_i] - b[p[k],loc_j])
                # ]
                #
                # = sum_{k != i,j} [
                #   (a[i,k] - a[j,k])*(b[loc_j,p[k]] - b[loc_i,p[k]])
                #   + (a[k,i] - a[k,j])*(b[p[k],loc_j] - b[p[k],loc_i])
                # ]
                
                # Vectorized computation
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                
                # b[loc_j, :] - b[loc_i, :]
                diff_row = b[loc_j, :] - b[loc_i, :]
                # b[:, loc_j] - b[:, loc_i]
                diff_col = b[:, loc_j] - b[:, loc_i]
                
                # (a[i, mask] - a[j, mask]) * diff_row[mask]
                term1 = np.dot(a[i, mask] - a[j, mask], diff_row[mask])
                # (a[mask, i] - a[mask, j]) * diff_col[mask]
                term2 = np.dot(a[mask, i] - a[mask, j], diff_col[mask])
                
                delta = term1 + term2
                
                if delta < 0:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
    
    return p, current_cost


def _random_perturbation(p, rng, num_swaps=5):
    """Apply random swaps to create diversity."""
    p = p.copy()
    n = len(p)
    for _ in range(num_swaps):
        i, j = rng.choice(n, size=2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    # Generate multiple diverse starting points
    starts = []
    
    # 1. Rank-matching start
    p1 = _rank_matching_start(a, b, rng)
    starts.append(p1)
    
    # 2. Hungarian-based start
    p2 = _hungarian_start(a, b, rng)
    starts.append(p2)
    
    # 3. Random start
    p3 = rng.permutation(n)
    starts.append(p3)
    
    # 4. Reverse rank-matching (light flows to central, heavy to periphery -- test both)
    heavy_order = np.argsort(-a.sum(axis=1), kind="stable")
    periphery_order = np.argsort(-b.sum(axis=1), kind="stable")
    p4 = np.empty(n, dtype=np.int64)
    p4[heavy_order] = periphery_order
    starts.append(p4)
    
    # 5. Another random
    p5 = rng.permutation(n)
    starts.append(p5)
    
    # Local search from each start, keeping the best
    best_p = None
    best_cost = np.inf
    
    for idx, p_init in enumerate(starts):
        if time.time() - start_time > deadline:
            break
        
        # Allocate remaining time among remaining starts
        remaining_starts = len(starts) - idx
        time_per_start = (deadline - (time.time() - start_time)) / max(remaining_starts, 1)
        time_limit_for_this = max(time_per_start - 2.0, 5.0)  # At least 5s per start
        
        p_local, cost_local = _local_search_2opt(
            a, b, p_init,
            max_iter=100000,
            time_limit=time_limit_for_this,
            start_time=start_time
        )
        
        if cost_local < best_cost:
            best_cost = cost_local
            best_p = p_local.copy()
    
    # Final polish: additional local search iterations on the best solution
    if time.time() - start_time < deadline - 5.0 and best_p is not None:
        remaining_time = deadline - (time.time() - start_time) - 3.0
        if remaining_time > 1.0:
            best_p, best_cost = _local_search_2opt(
                a, b, best_p,
                max_iter=200000,
                time_limit=remaining_time,
                start_time=start_time
            )
    
    if best_p is None:
        best_p = p1
    
    return best_p.tolist()
# EVOLVE-BLOCK-END

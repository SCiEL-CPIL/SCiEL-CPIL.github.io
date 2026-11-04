"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def compute_cost(a, b, p):
    """Exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    # Use integer arithmetic to be exact
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when we swap the locations of facilities i and j.
    p is the current assignment: facility k is at location p[k].
    Swapping means: facility i goes to p[j], facility j goes to p[i].
    """
    li = p[i]
    lj = p[j]
    
    # Current contribution involving i or j
    # After swap:
    #   term for (i, j): a[i,j] * b[lj, li]  (was a[i,j]*b[li,lj])
    #   term for (j, i): a[j,i] * b[li, lj]  (was a[j,i]*b[lj,li])
    #   term for (i, k) for k != i,j: a[i,k] * b[lj, p[k]]  (was a[i,k]*b[li, p[k]])
    #   term for (k, i) for k != i,j: a[k,i] * b[p[k], lj]  (was a[k,i]*b[p[k], li])
    #   term for (j, k) for k != i,j: a[j,k] * b[li, p[k]]  (was a[j,k]*b[lj, p[k]])
    #   term for (k, j) for k != i,j: a[k,j] * b[p[k], li]  (was a[k,j]*b[p[k], lj])
    
    # Delta for cross terms (i,j) and (j,i):
    delta_ij = a[i, j] * (b[lj, li] - b[li, lj]) + a[j, i] * (b[li, lj] - b[lj, li])
    delta_ij = delta_ij  # = (a[i,j] - a[j,i]) * (b[lj,li] - b[li,lj])
    
    # Delta for i with all others k != i,j:
    # sum_{k!=i,j} a[i,k] * (b[lj, p[k]] - b[li, p[k]])
    others = np.arange(len(p))
    others = others[(others != i) & (others != j)]
    if len(others) > 0:
        delta_i_others = np.sum(a[i, others] * (b[lj, p[others]] - b[li, p[others]]))
        delta_j_others = np.sum(a[j, others] * (b[li, p[others]] - b[lj, p[others]]))
        delta_oi = np.sum(a[others, i] * (b[p[others], lj] - b[p[others], li]))
        delta_oj = np.sum(a[others, j] * (b[p[others], li] - b[p[others], lj]))
    else:
        delta_i_others = 0
        delta_j_others = 0
        delta_oi = 0
        delta_oj = 0
    
    return int(delta_ij + delta_i_others + delta_j_others + delta_oi + delta_oj)


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute for speed
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)
    
    # Initial assignment: rank-matching (heavy flows to short distances)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.tolist()
    
    current_cost = compute_cost(a, b, p)
    best_cost = current_cost
    best_p = p[:]
    
    # 2-opt local search with simulated annealing-style acceptance
    temperature = max(current_cost * 0.001, 10.0)
    cool_rate = 0.995
    min_temp = 0.01
    
    def _swap_delta_fast(p_arr, i, j):
        """Fast swap delta using numpy."""
        li = p_arr[i]
        lj = p_arr[j]
        
        # Cross terms
        delta = (a[i, j] - a[j, i]) * (b[lj, li] - b[li, lj])
        
        others_mask = np.ones(n, dtype=bool)
        others_mask[i] = False
        others_mask[j] = False
        others_idx = np.nonzero(others_mask)[0]
        
        if len(others_idx) > 0:
            pk_others = p_arr[others_idx]
            # a[i,k]*(b[lj,pk] - b[li,pk]) + a[k,i]*(b[pk,lj] - b[pk,li])
            # + a[j,k]*(b[li,pk] - b[lj,pk]) + a[k,j]*(b[pk,li] - b[pk,lj])
            # = (a[i,k]-a[j,k])*(b[lj,pk]-b[li,pk]) + (a[k,i]-a[k,j])*(b[pk,lj]-b[pk,li])
            ai_jk = a[i, others_idx] - a[j, others_idx]
            aki_kj = a[others_idx, i] - a[others_idx, j]
            blj_bli = b[lj, pk_others] - b[li, pk_others]
            bpk_blj = b[pk_others, lj] - b[pk_others, li]
            delta += np.sum(ai_jk * blj_bli + aki_kj * bpk_blj)
        
        return int(delta)
    
    p_arr = np.array(p, dtype=np.int64)
    
    # Local search loop
    steps_since_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Pick random pair to swap
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        d = _swap_delta_fast(p_arr, i, j)
        
        # Accept if improvement or with probability exp(-d/T)
        if d < 0 or (temperature > min_temp and np.random.random() < np.exp(-d / temperature)):
            # Perform swap
            p_arr[i], p_arr[j] = p_arr[j], p_arr[i]
            current_cost += d
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p_arr.tolist()
                steps_since_improve = 0
            else:
                steps_since_improve += 1
            
            # Cool down
            temperature *= cool_rate
            if temperature < min_temp:
                temperature = min_temp
        else:
            steps_since_improve += 1
        
        if steps_since_improve > max_no_improve:
            # Reset: restart from best or reheat
            if time.time() > deadline - 30:
                break
            p_arr = np.array(best_p, dtype=np.int64)
            current_cost = best_cost
            temperature = max(best_cost * 0.002, 10.0)
            steps_since_improve = 0
    
    return best_p
# EVOLVE-BLOCK-END

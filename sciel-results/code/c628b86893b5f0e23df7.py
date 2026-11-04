"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the exact QAP cost sum_{i,j} a[i,j] * b[p[i],p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.dot(a.ravel(), bp.ravel()))


def _compute_swap_deltas(a, b, p, i):
    """
    Compute deltas for swapping facility i with each facility j != i.
    Returns array of length n-1 (deltas for j=0..n-1 excluding j=i).
    
    Uses vectorized operations for efficiency.
    """
    n = len(p)
    pi = p[i]
    
    # For each j != i, delta = change in cost when swapping p[i] and p[j]
    # Let's derive:
    # Old terms involving i or j:
    #   a[i,i]*b[pi,pi] + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] + a[j,j]*b[pj,pj]
    #   + sum_{k!=i,j} a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi]
    #   + sum_{k!=i,j} a[j,k]*b[pj,p[k]] + a[k,j]*b[p[k],pj]
    # New terms:
    #   a[i,i]*b[pj,pj] + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] + a[j,j]*b[pi,pi]
    #   + sum_{k!=i,j} a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj]
    #   + sum_{k!=i,j} a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi]
    
    # Let pj = p[j]
    # delta_j = a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
    #          + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    #          + sum_{k!=i,j} a[i,k]*(b[pj,p[k]]-b[pi,p[k]])
    #          + sum_{k!=i,j} a[k,i]*(b[p[k],pj]-b[p[k],pi])
    #          + sum_{k!=i,j} a[j,k]*(b[pi,p[k]]-b[pj,p[k]])
    #          + sum_{k!=i,j} a[k,j]*(b[p[k],pi]-b[p[k],pj])
    
    # Group terms:
    # = (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
    #   + (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
    #   + sum_{k!=i,j} [a[i,k]-a[j,k]]*(b[pj,p[k]]-b[pi,p[k]])
    #   + sum_{k!=i,j} [a[k,i]-a[k,j]]*(b[p[k],pj]-b[p[k],pi])
    
    # Let's compute for all j at once using vectorization.
    # For a fixed i, we want delta for each j.
    
    # Precompute:
    # row_diff[i,k] = a[i,k] - a[j,k] -- but j varies, so this doesn't work directly
    
    # Alternative: compute the full delta matrix for all pairs (i,j) at once?
    # That would be O(n^3) which is too much.
    
    # Let's just compute for a fixed i, all j.
    # For each j:
    #   term1 = (a[i,i] - a[j,j]) * (b[p[j],p[j]] - b[pi,pi])
    #   term2 = (a[i,j] - a[j,i]) * (b[p[j],pi] - b[pi,p[j]])
    #   term3 = sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[p[j],p[k]] - b[pi,p[k]])
    #   term4 = sum_{k!=i,j} (a[k,i]-a[k,j]) * (b[p[k],p[j]] - b[p[k],pi])
    
    # term3 and term4 are the expensive parts.
    
    # Let's compute them using matrix operations.
    # b[p[j], p[k]] for all k = b[p[j], p] (row p[j] of b, indexed by p)
    # b[pi, p[k]] for all k = b[pi, p] (row pi of b, indexed by p)
    
    # For a given j, term3 = sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[p[j],p[k]] - b[pi,p[k]])
    # Let diff_a = a[i,:] - a[j,:]  (length n)
    # Let diff_b = b[p[j], p] - b[pi, p]  (length n)
    # term3 = sum_{k} diff_a[k] * diff_b[k] - diff_a[i]*diff_b[i] - diff_a[j]*diff_b[j]
    #        = dot(diff_a, diff_b) - diff_a[i]*diff_b[i] - diff_a[j]*diff_b[j]
    
    # Similarly for term4:
    # term4 = sum_{k!=i,j} (a[k,i]-a[k,j]) * (b[p[k],p[j]] - b[p[k],pi])
    # Let diff_a_col = a[:,i] - a[:,j]  (length n)
    # Let diff_b_col = b[p, p[j]] - b[p, pi]  (length n)
    # term4 = sum_{k} diff_a_col[k] * diff_b_col[k] - diff_a_col[i]*diff_b_col[i] - diff_a_col[j]*diff_b_col[j]
    
    # This is O(n) per j, so O(n^2) total for all j. That's acceptable.
    
    deltas = np.empty(n, dtype=np.int64)
    deltas[i] = 0  # Won't use
    
    b_pi = b[pi]  # Row pi of b
    b_p = b[p]  # b[p[k], p[l]] for all k,l -- this is the permuted b
    
    for j in range(n):
        if j == i:
            continue
        pj = p[j]
        
        # term1
        term1 = (a[i,i] - a[j,j]) * (b[pj,pj] - b[pi,pi])
        
        # term2
        term2 = (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj])
        
        # term3
        diff_a = a[i,:] - a[j,:]
        diff_b = b[pj, p] - b[pi, p]
        term3 = np.dot(diff_a, diff_b) - diff_a[i]*diff_b[i] - diff_a[j]*diff_b[j]
        
        # term4
        diff_a_col = a[:,i] - a[:,j]
        diff_b_col = b[p, pj] - b[p, pi]
        term4 = np.dot(diff_a_col, diff_b_col) - diff_a_col[i]*diff_b_col[i] - diff_a_col[j]*diff_b_col[j]
        
        deltas[j] = int(term1 + term2 + term3 + term4)
    
    return deltas


def _tabu_search(a, b, p, max_iters, tabu_tenure, deadline):
    """
    Tabu search with efficient swap delta evaluation.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    tabu_until = {}  # (min(i,j), max(i,j)) -> iteration when tabu expires
    iteration = 0
    
    while iteration < max_iters:
        if time.time() > deadline:
            break
        
        # Find best non-tabu move (or aspiration)
        best_delta = np.inf
        best_pair = None
        best_is_tabu = False
        
        for i in range(n):
            deltas = _compute_swap_deltas(a, b, p, i)
            for j in range(i + 1, n):
                delta = deltas[j]
                
                key = (i, j)
                is_tabu = key in tabu_until and tabu_until[key] > iteration
                
                # Aspiration: accept tabu if it improves best
                if is_tabu and current_cost + delta < best_cost:
                    is_tabu = False
                
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (i, j)
                    best_is_tabu = is_tabu
        
        if best_pair is None:
            break
        
        i, j = best_pair
        
        # If best move is tabu and not aspiration, skip
        if best_is_tabu:
            break
        
        # Make the move
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Add to tabu
        tabu_until[(i, j)] = iteration + tabu_tenure
        
        # Clean up
        if len(tabu_until) > n * n // 2:
            tabu_until = {k: v for k, v in tabu_until.items() if v > iteration}
        
        iteration += 1
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind='stable')
    central = np.argsort(b.sum(axis=1), kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Tabu search
    remaining = deadline - time.time()
    max_iters = int(remaining * 20)  # Conservative estimate
    max_iters = max(max_iters, 50)
    
    tabu_tenure = max(5, n // 10)
    
    best_p, best_cost = _tabu_search(a, b, p, max_iters, tabu_tenure, deadline)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END

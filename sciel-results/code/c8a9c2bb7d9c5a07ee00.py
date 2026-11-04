"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact QAP cost for a given permutation p."""
    # p[i] is the location assigned to facility i
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # This is equivalent to sum_{i,j} a[i,j] * B[p[i], p[j]] where B = b
    # We can compute this efficiently:
    # Let P be the permutation matrix such that P[i, p[i]] = 1
    # Then cost = sum_{i,j} a[i,j] * (P^T B P)[i,j] ... actually simpler:
    # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i a[i,:] dot b[p[i], p[?]] ... 
    # More directly: create permuted b
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))

def _swap_cost_delta(a, b, p, i, j):
    """
    Compute the change in QAP cost when swapping positions of facilities i and j
    in permutation p. That is, p[i] and p[j] are swapped.
    
    Let p_old be the current permutation.
    Let p_new be p with p[i] and p[j] swapped.
    
    cost_old = sum_{k,l} a[k,l] * b[p[k], p[l]]
    cost_new = sum_{k,l} a[k,l] * b[p_new[k], p_new[l]]
    
    The difference only involves terms where k or l is i or j.
    
    Let u = p[i], v = p[j]
    
    In p_old:
      term for (i,i): a[i,i]*b[u,u]
      term for (i,j): a[i,j]*b[u,v]
      term for (j,i): a[j,i]*b[v,u]
      term for (j,j): a[j,j]*b[v,v]
      terms for (i,k) k!=i,j: a[i,k]*b[u, p[k]]
      terms for (j,k) k!=i,j: a[j,k]*b[v, p[k]]
      terms for (k,i) k!=i,j: a[k,i]*b[p[k], u]
      terms for (k,j) k!=i,j: a[k,j]*b[p[k], v]
    
    In p_new, p_new[i]=v, p_new[j]=u, others same:
      term for (i,i): a[i,i]*b[v,v]
      term for (i,j): a[i,j]*b[v,u]
      term for (j,i): a[j,i]*b[u,v]
      term for (j,j): a[j,j]*b[u,u]
      terms for (i,k) k!=i,j: a[i,k]*b[v, p[k]]
      terms for (j,k) k!=i,j: a[j,k]*b[u, p[k]]
      terms for (k,i) k!=i,j: a[k,i]*b[p[k], v]
      terms for (k,j) k!=i,j: a[k,j]*b[p[k], u]
    
    delta = cost_new - cost_old
    
    delta = a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v])
          + sum_{k!=i,j} [a[i,k]*(b[v,p[k]]-b[u,p[k]]) + a[j,k]*(b[u,p[k]]-b[v,p[k]])]
          + sum_{k!=i,j} [a[k,i]*(b[p[k],v]-b[p[k],u]) + a[k,j]*(b[p[k],u]-b[p[k],v])]
    
    Let's simplify. Note b is symmetric in QAP typically, but let's not assume.
    Actually for tai256c, b is a distance matrix which is symmetric. But let's be general.
    
    Let's compute efficiently using precomputed row/col sums or vector operations.
    """
    u = p[i]
    v = p[j]
    
    # Direct terms involving i and j
    delta = 0
    delta += a[i,i] * (b[v,v] - b[u,u])
    delta += a[i,j] * (b[v,u] - b[u,v])
    delta += a[j,i] * (b[u,v] - b[v,u])
    delta += a[j,j] * (b[u,u] - b[v,v])
    
    # Terms where one index is i or j, the other is k (k != i,j)
    # We need:
    # sum_{k!=i,j} a[i,k]*(b[v,k'] - b[u,k']) where k' = p[k]
    # sum_{k!=i,j} a[j,k]*(b[u,k'] - b[v,k'])
    # sum_{k!=i,j} a[k,i]*(b[k',v] - b[k',u])
    # sum_{k!=i,j} a[k,j]*(b[k',u] - b[k',v])
    
    # Let's use vectorized operations.
    # For the first sum: sum_{k!=i,j} a[i,k] * (b[v, p[k]] - b[u, p[k]])
    # = sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]]) - a[i,i]*(b[v,p[i]]-b[u,p[i]]) - a[i,j]*(b[v,p[j]]-b[u,p[j]])
    # p[i]=u, p[j]=v
    # = sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]]) - a[i,i]*(b[v,u]-b[u,u]) - a[i,j]*(b[v,v]-b[u,v])
    
    # This is getting complex. Let's just compute the full cost difference by computing
    # the relevant submatrix contributions.
    
    # Actually, a simpler approach: compute the cost contribution of all pairs (k,l) where k in {i,j} or l in {i,j}
    # in both old and new permutations.
    
    # Old contribution from pairs involving i or j:
    # We'll compute the full cost change by considering all pairs (k,l) where k or l is i or j.
    
    # Let's just compute it directly with numpy for clarity, it's O(n) per swap
    # which is acceptable for n=256.
    
    # Create a mask of indices that are not i or j
    mask = np.ones(len(a), dtype=bool)
    mask[i] = False
    mask[j] = False
    others = np.where(mask)[0]
    
    # Old cost terms involving i or j:
    old_ij = 0
    # (i,i), (i,j), (j,i), (j,j)
    old_ij += a[i,i]*b[u,u] + a[i,j]*b[u,v] + a[j,i]*b[v,u] + a[j,j]*b[v,v]
    # (i,k) for k in others
    if len(others) > 0:
        old_ij += np.sum(a[i, others] * b[u, p[others]])
        old_ij += np.sum(a[j, others] * b[v, p[others]])
        old_ij += np.sum(a[others, i] * b[p[others], u])
        old_ij += np.sum(a[others, j] * b[p[others], v])
    
    # New cost terms involving i or j (after swap, p_new[i]=v, p_new[j]=u):
    new_ij = 0
    new_ij += a[i,i]*b[v,v] + a[i,j]*b[v,u] + a[j,i]*b[u,v] + a[j,j]*b[u,u]
    if len(others) > 0:
        new_ij += np.sum(a[i, others] * b[v, p[others]])
        new_ij += np.sum(a[j, others] * b[u, p[others]])
        new_ij += np.sum(a[others, i] * b[p[others], v])
        new_ij += np.sum(a[others, j] * b[p[others], u])
    
    return new_ij - old_ij

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    
    # 2-opt local search
    # Precompute some helper arrays for speed
    # For faster swap evaluation, we can precompute row/col sums
    
    # Let's use a simpler but effective 2-opt
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                if time.time() > deadline:
                    break
                delta = _swap_cost_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    improved = True
                    if best_cost < _qap_cost(a, b, best_p):
                        best_p = p.copy()
        
        if time.time() > deadline:
            break
    
    # Verify best cost
    final_cost = _qap_cost(a, b, best_p)
    if final_cost < best_cost:
        best_cost = final_cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END

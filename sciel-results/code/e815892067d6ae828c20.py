"""Improved QAP solver using Iterated Local Search with O(n) swap deltas."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    n = len(p)
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))

def _rank_matching_init(a, b):
    """Initial permutation based on rank matching."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _calc_delta(a, b, p, i, j):
    """
    Calculate the change in cost if we swap positions i and j in permutation p.
    O(n) complexity.
    """
    vi = p[i]
    vj = p[j]
    
    # We need to compute the difference between the new cost and old cost.
    # Old terms involving i or j:
    # 1. A[i,i] B[vi,vi]
    # 2. A[j,j] B[vj,vj]
    # 3. A[i,j] B[vi,vj]
    # 4. A[j,i] B[vj,vi]
    # 5. Sum_{k!=i,j} A[k,i] B[vk,vi]
    # 6. Sum_{k!=i,j} A[k,j] B[vk,vj]
    # 7. Sum_{l!=i,j} A[i,l] B[vi,vl]
    # 8. Sum_{l!=i,j} A[j,l] B[vj,vl]
    
    # New terms:
    # 1. A[i,i] B[vj,vj]
    # 2. A[j,j] B[vi,vi]
    # 3. A[i,j] B[vj,vi]
    # 4. A[j,i] B[vi,vj]
    # 5. Sum_{k!=i,j} A[k,i] B[vk,vj]
    # 6. Sum_{k!=i,j} A[k,j] B[vk,vi]
    # 7. Sum_{l!=i,j} A[i,l] B[vj,vl]
    # 8. Sum_{l!=i,j} A[j,l] B[vi,vl]
    
    # Delta = New - Old
    
    # Diagonal terms
    delta_diag = a[i, i] * (b[vj, vj] - b[vi, vi]) + a[j, j] * (b[vi, vi] - b[vj, vj])
    
    # Cross terms
    delta_cross = a[i, j] * (b[vj, vi] - b[vi, vj]) + a[j, i] * (b[vi, vj] - b[vj, vi])
    
    # Terms with k (rows in A, cols in B)
    # Sum_{k} A[k,i] (B[vk, vj] - B[vk, vi]) - (k=i, k=j terms)
    # Note: B is symmetric in QAPLIB, so B[vk, vj] = B[vj, vk].
    # But let's stick to indices.
    
    # Vectorized sum for all k
    sum_k_i = np.dot(a[:, i], b[:, vj]) - np.dot(a[:, i], b[:, vi])
    # Subtract k=i and k=j
    sum_k_i -= a[i, i] * (b[i, vj] - b[i, vi])
    sum_k_i -= a[j, i] * (b[j, vj] - b[j, vi])
    
    sum_k_j = np.dot(a[:, j], b[:, vi]) - np.dot(a[:, j], b[:, vj])
    sum_k_j -= a[i, j] * (b[i, vi] - b[i, vj])
    sum_k_j -= a[j, j] * (b[j, vi] - b[j, vj])
    
    # Terms with l (cols in A, cols in B)
    # Sum_{l} A[i,l] (B[vj, vl] - B[vi, vl]) - (l=i, l=j terms)
    
    sum_l_i = np.dot(a[i, :], b[vj, :]) - np.dot(a[i, :], b[vi, :])
    sum_l_i -= a[i, i] * (b[vj, i] - b[vi, i])
    sum_l_i -= a[i, j] * (b[vj, j] - b[vi, j])
    
    sum_l_j = np.dot(a[j, :], b[vi, :]) - np.dot(a[j, :], b[vj, :])
    sum_l_j -= a[j, i] * (b[vi, i] - b[vj, i])
    sum_l_j -= a[j, j] * (b[vi, j] - b[vj, j])
    
    total_delta = delta_diag + delta_cross + sum_k_i + sum_k_j + sum_l_i + sum_l_j
    return int(total_delta)

def _local_search(a, b, p, time_limit, start_time):
    """
    Perform 2-opt local search using O(n) delta calculation.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    improved = True
    
    while improved:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                return p, current_cost
            for j in range(i + 1, n):
                delta = _calc_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
    return p, current_cost

def solve(a, b):
    """
    Solve the QAP instance using Iterated Local Search.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Convert to int64 if needed
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    best_p = None
    best_cost = np.inf
    rng = np.random.RandomState(42)
    
    # Initial solution
    p0 = _rank_matching_init(a, b)
    
    # Iterated Local Search
    while time.time() < deadline - 10.0:  # Leave 10s for final polish
        # Local Search
        p_ls, cost_ls = _local_search(a, b, p0, time_limit=5.0, start_time=time.time())
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        # Perturb the local optimum to escape
        p0 = p_ls.copy()
        n_perturb = max(2, int(n * 0.1))
        for _ in range(n_perturb):
            i, j = rng.choice(n, 2, replace=False)
            p0[i], p0[j] = p0[j], p0[i]
        
        # Optional: Random restart if stuck for too long? 
        # For now, just perturb.
        
    # Final Polish with remaining time
    remaining_time = deadline - time.time() - 2.0
    if remaining_time > 0 and best_p is not None:
        p_final, cost_final = _local_search(a, b, best_p, time_limit=remaining_time, start_time=time.time())
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final
    
    if best_p is None:
        best_p = p0
    
    return best_p.tolist()
# EVOLVE-BLOCK-END

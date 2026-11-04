"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Calculate the exact QAP cost for a given permutation p."""
    # p is a list or array of length n, p[i] is the location of facility i
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Vectorized:
    # b_perm[i, j] = b[p[i], p[j]]
    # cost = sum(a * b_perm)
    b_perm = b[np.array(p)[:, None], np.array(p)[None, :]]
    return np.sum(a * b_perm)

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_cost = np.inf
    best_p = None
    
    # Helper to run local search from a given start
    def local_search(p_init):
        p = np.array(p_init, dtype=np.int64)
        current_cost = _qap_cost(a, b, p)
        
        while True:
            if time.time() > deadline:
                return p, current_cost
            
            # Compute B_p = B[p[:, None], p[None, :]]
            # This is the distance matrix in facility indices
            B_p = b[p[:, None], p[None, :]]
            
            # Compute M = A @ B_p^T
            # M[u, v] = sum_k A[u, k] * B_p[v, k]
            M = a @ B_p.T
            
            # Compute Delta matrix D
            # Delta(u, v) = 2 * (M[u, v] + M[v, u] - M[u, u] - M[v, v] + 2 * A[u, v] * B_p[u, v])
            # Note: A and B_p are symmetric in this context? 
            # A is symmetric. B is symmetric, so B_p is symmetric.
            # So M is not necessarily symmetric, but M[v, u] is just M.T[u, v].
            
            diag_M = np.diag(M)
            # D[u, v] for u != v
            # We want to find min D[u, v]
            
            # Construct D
            # Term 1: M + M.T
            # Term 2: - diag_M[:, None] - diag_M[None, :]
            # Term 3: 2 * (A * B_p)
            
            D = M + M.T - diag_M[:, None] - diag_M[None, :] + 2.0 * (a * B_p)
            
            # Mask diagonal
            np.fill_diagonal(D, np.inf)
            
            # Find best swap
            min_idx = np.unravel_index(np.argmin(D), D.shape)
            u, v = min_idx
            
            delta = D[u, v]
            
            if delta < -1e-9: # Significant improvement
                p[u], p[v] = p[v], p[u]
                current_cost += delta
            else:
                # Local optimum reached
                break
        
        return p, current_cost

    # Strategy: Multiple restarts with different initializations
    # 1. Greedy Rank Matching (from seed)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_greedy = np.empty(n, dtype=np.int64)
    p_greedy[heavy] = central
    p, cost = local_search(p_greedy)
    if cost < best_cost:
        best_cost = cost
        best_p = p.copy()
        
    # 2. Random Restarts
    # Determine how many restarts we can afford
    # Estimate time per restart
    # We'll just loop until deadline
    while time.time() < deadline:
        p_rand = np.random.permutation(n)
        p, cost = local_search(p_rand)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
    return best_p.tolist()
# EVOLVE-BLOCK-END

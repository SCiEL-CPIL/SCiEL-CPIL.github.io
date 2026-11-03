"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                max_err = max(max_err, abs(s - 1.0))
    
    return max_err < 1e-9

def _compute_error(U, V, W, n, m, p, rank):
    """Compute max error for a decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                max_err = max(max_err, abs(s - 1.0))
    return max_err

def _local_optimize(U, V, W, n, m, p, rank, max_iters=50, time_budget=2.0):
    """Vectorized coordinate descent optimization to reduce error."""
    start_time = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Precompute index mappings for vectorized operations
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Optimize U: for each (i,j), U[i*m+j, :] = (1 - U[i*m+j,:] @ (V*W_k)) / (V*W_k)
        # where V*W_k[c] = sum_k V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                # Compute coeff vector: for each c, sum over k of V[j*p+k, c] * W[k*n+i, c]
                # V[j*p:(j+1)*p, :] has shape (p, rank)
                # W[:, i] has shape (p,) -- wait, W is (p*n, rank), W[k*n+i, c]
                # So we need W at positions [0*n+i, 1*n+i, ..., (p-1)*n+i]
                w_idx = np.array([k * n + i for k in range(p)])
                v_slice = V[j * p:(j + 1) * p, :]  # (p, rank)
                w_slice = W[w_idx, :]  # (p, rank)
                coeff = np.sum(v_slice * w_slice, axis=0)  # (rank,)
                
                # Compute rest: U[idx, :] @ (V_slice * W_slice) - U[idx, :] * coeff
                # Actually: sum_c2 U[idx,c2] * V[j*p+k,c2] * W[k*n+i,c2] summed over k
                # = sum_k (V_slice[:, c2] * W_slice[:, c2]) dot U[idx, c2] for each c2
                # = U[idx, :] @ (sum_k V_slice * W_slice) = U[idx, :] @ coeff
                # But that's the total including c itself. We need rest = total - U[idx,c]*coeff[c]
                # So rest for all c: U[idx,:] @ coeff - U[idx,:] * coeff = 0? No.
                # rest_c = sum_{c2 != c} U[idx,c2] * coeff[c2] = sum_c2 U[idx,c2]*coeff[c2] - U[idx,c]*coeff[c]
                total = np.dot(U[idx, :], coeff)
                rest = total - U[idx, :] * coeff  # element-wise: rest[c] = total - U[idx,c]*coeff[c]
                
                # U[idx, c] = (1 - rest[c]) / coeff[c]
                mask = np.abs(coeff) > 1e-12
                U[idx, mask] = (1.0 - rest[mask]) / coeff[mask]
                U[idx, ~mask] = 0.0
        
        # Optimize V: for each (j,k), V[j*p+k, c] = (1 - rest) / coeff
        # coeff[c] = sum_i U[i*m+j, c] * W[k*n+i, c]
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                u_idx = np.array([i * m + j for i in range(n)])
                w_idx = np.array([k * n + i for i in range(n)])
                u_slice = U[u_idx, :]  # (n, rank)
                w_slice = W[w_idx, :]  # (n, rank)
                coeff = np.sum(u_slice * w_slice, axis=0)  # (rank,)
                
                total = np.dot(V[idx, :], coeff)
                rest = total - V[idx, :] * coeff
                
                mask = np.abs(coeff) > 1e-12
                V[idx, mask] = (1.0 - rest[mask]) / coeff[mask]
                V[idx, ~mask] = 0.0
        
        # Optimize W: for each (k,i), W[k*n+i, c] = (1 - rest) / coeff
        # coeff[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                u_idx = np.array([i * m + j for j in range(m)])
                v_idx = np.array([j * p + k for j in range(m)])
                u_slice = U[u_idx, :]  # (m, rank)
                v_slice = V[v_idx, :]  # (m, rank)
                coeff = np.sum(u_slice * v_slice, axis=0)  # (rank,)
                
                total = np.dot(W[idx, :], coeff)
                rest = total - W[idx, :] * coeff
                
                mask = np.abs(coeff) > 1e-12
                W[idx, mask] = (1.0 - rest[mask]) / coeff[mask]
                W[idx, ~mask] = 0.0
    
    return U, V, W

def _structured_init(n, m, p, target_rank, trial):
    """Create structured initializations based on schoolbook subsets."""
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    if target_rank >= n * m * p:
        # Full schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < target_rank:
                        U[i*m + j, col] = 1.0
                        V[j*p + k, col] = 1.0
                        W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Strategy: Use schoolbook terms for as many columns as possible,
    # then add small random perturbations to enable optimization
    all_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_schoolbook = min(target_rank, len(all_terms))
    
    if trial == 0:
        # Deterministic: first num_schoolbook terms
        selected = all_terms[:num_schoolbook]
    else:
        # Random subset
        selected = random.sample(all_terms, num_schoolbook)
    
    for c, (i, j, k) in enumerate(selected):
        U[i*m + j, c] = 1.0
        V[j*p + k, c] = 1.0
        W[k*n + i, c] = 1.0
    
    # Add random perturbation to all columns to break symmetry
    # Use small perturbations that allow the optimizer to move
    scale = 0.05
    U[:, :] += np.random.randn(n * m, target_rank) * scale
    V[:, :] += np.random.randn(m * p, target_rank) * scale
    W[:, :] += np.random.randn(p * n, target_rank) * scale
    
    return U, V, W

def _try_rank_search(n, m, p, max_rank, time_limit=10.0):
    """Try to find a decomposition with rank <= max_rank using structured search.
    Limited time budget to ensure fast fallback."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Try ranks from low to high, but focus on the most promising range
    # For 3x5x5, known lower bounds suggest rank ~ 25-35 might be achievable
    # Start from a reasonable minimum and go up
    min_rank = max(1, n * m * p // 3)  # Rough heuristic
    max_search = min(max_rank, 60)
    
    # Try ranks in ascending order
    ranks_to_try = list(range(min_rank, max_search + 1))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 2.0:
            break
        
        # Adaptive number of trials based on remaining time
        remaining = deadline - time.time()
        num_tries = max(2, int(remaining / max(target_rank * 0.1, 1.0)))
        num_tries = min(num_tries, 20)
        
        for trial in range(num_tries):
            if time.time() > deadline - 1.0:
                break
            
            # Use structured initialization
            U, V, W = _structured_init(n, m, p, target_rank, trial)
            
            # Local optimization with strict time limit
            time_per_opt = min(0.8, remaining / (num_tries * 2))
            U, V, W = _local_optimize(U, V, W, n, m, p, target_rank, 
                                       max_iters=30, time_budget=time_per_opt)
            
            # Check error
            err = _compute_error(U, V, W, n, m, p, target_rank)
            
            if err < 1e-9:
                U_list = [[float(x) for x in U[row]] for row in range(n*m)]
                V_list = [[float(x) for x in V[row]] for row in range(m*p)]
                W_list = [[float(x) for x in W[row]] for row in range(p*n)]
                
                # Try to round to half-integers
                U_rounded = [[round(2*x)/2 for x in row] for row in U_list]
                V_rounded = [[round(2*x)/2 for x in row] for row in V_list]
                W_rounded = [[round(2*x)/2 for x in row] for row in W_list]
                
                if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, target_rank):
                    return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                else:
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, get the schoolbook decomposition as a safe fallback
    schoolbook = _schoolbook_decomposition(n, m, p)
    schoolbook_rank = n * m * p
    
    # Try to find a better decomposition with a strict time limit
    # Use most of the available time (150s limit, keep 30s safety margin)
    time_limit = 110.0
    result = _try_rank_search(n, m, p, schoolbook_rank - 1, time_limit=time_limit)
    
    if result is not None:
        return result
    else:
        return schoolbook
# EVOLVE-BLOCK-END

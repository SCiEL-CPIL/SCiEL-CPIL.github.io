"""Matrix multiplication 4,5,5 decomposition search.
Attempts to find a decomposition with rank < 100 via algebraic/numerical search,
falls back to schoolbook (rank 100) if no better solution is found within budget.
"""

import time
import math
import random
import numpy as np

DIMENSIONS = [4, 5, 5]


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W form a valid rank-r decomposition."""
    rank = len(V[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor from the decomposition:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And compare with the matmul tensor:
    # T_matmul[i,j,k] = (A @ B)[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] should be 1 if the "standard" matmul tensor, which is:
    # T[i,j,k] = 1 for all (i,j,k) in the matmul identity
    # Actually the tensor for matmul is T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k}... 
    # No. The matmul tensor is defined so that sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k].
    # That means T[i,j,k] = 1 for all i,j,k (it's just the index selection).
    # Wait, that's not right either. The tensor T has components T[i,j,k] such that
    # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]. Since C[i,k] = sum_j A[i,j] B[j,k],
    # we need T[i,j,k] = 1 for all i,j,k.
    
    # So we check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(rank):
                    s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Vectorized verification."""
    rank = len(V[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape U, V, W into 3D tensors for faster computation
    # U_reshaped[i, j, c] = U[i*m + j, c]
    # V_reshaped[j, k, c] = V[j*p + k, c]  
    # W_reshaped[k, i, c] = W[k*n + i, c]
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum
    # Contract over c
    UV = np.einsum('ijc,jkc->ijkc', U3, V3)  # shape (n, m, p, rank)
    T = np.einsum('ijkc,kic->ijk', UV, W3)   # shape (n, m, p)
    
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Leave 10s safety margin
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    if time.time() > deadline:
        return best_result
    
    # Try to find a lower-rank decomposition using random search with gradient-free optimization
    # Strategy: random initialization + local search (pattern search) on the 3 factor matrices
    
    rng = random.Random(42)
    np.random.seed(42)
    
    # Upper bound to try: we'll attempt ranks from (n*m*p) - 10 down to some lower bound
    # For 4x5x5, known lower bounds suggest rank could be around 70-90
    # Let's try a few candidate ranks with random search
    
    candidate_ranks = []
    for r in range(best_rank - 1, best_rank - 30, -1):
        candidate_ranks.append(r)
    
    for target_rank in candidate_ranks[:8]:  # Try up to 8 different ranks
        if time.time() > deadline - 5:
            break
        
        best_in_round = None
        best_in_round_rank = target_rank + 1
        
        # Multiple random restarts for this target rank
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small perturbations around a sparse structure
            # Initialize with a bias toward having some 1s and -1s
            U = np.array([[0.0] * target_rank for _ in range(n * m)], dtype=float)
            V = np.array([[0.0] * target_rank for _ in range(m * p)], dtype=float)
            W = np.array([[0.0] * target_rank for _ in range(p * n)], dtype=float)
            
            # Sparsify: set each entry to 1, -1, or 0 with some probability
            for i in range(n * m):
                for c in range(target_rank):
                    r = rng.random()
                    if r < 0.3:
                        U[i, c] = 1.0
                    elif r < 0.5:
                        U[i, c] = -1.0
                    else:
                        U[i, c] = 0.0
            
            for j in range(m * p):
                for c in range(target_rank):
                    r = rng.random()
                    if r < 0.3:
                        V[j, c] = 1.0
                    elif r < 0.5:
                        V[j, c] = -1.0
                    else:
                        V[j, c] = 0.0
            
            for k in range(p * n):
                for c in range(target_rank):
                    r = rng.random()
                    if r < 0.3:
                        W[k, c] = 1.0
                    elif r < 0.5:
                        W[k, c] = -1.0
                    else:
                        W[k, c] = 0.0
            
            # Check initial error
            if time.time() > deadline - 5:
                break
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        for c in range(target_rank):
                            s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                        err += (s - 1.0) ** 2
                        if time.time() > deadline - 3:
                            break
                    if time.time() > deadline - 3:
                        break
                if time.time() > deadline - 3:
                    break
            if time.time() > deadline - 3:
                break
            
            # Local search: coordinate descent style updates
            alpha = 0.1
            n_iter = 2000
            for it in range(n_iter):
                if time.time() > deadline - 3:
                    break
                
                # Pick a random element to update
                # Choose which matrix and which entry
                mat_choice = rng.random()
                if mat_choice < 0.33:
                    idx_i = rng.randint(0, n * m - 1)
                    idx_c = rng.randint(0, target_rank - 1)
                    old_val = U[idx_i, idx_c]
                    # Try adding a random perturbation
                    delta = rng.choice([-1, -0.5, 0.5, 1, 2, -2]) * alpha
                    new_val = old_val + delta
                    U[idx_i, idx_c] = new_val
                    matrix = 'U'
                    row, col = idx_i, idx_c
                    old_v, new_v = old_val, new_val
                elif mat_choice < 0.66:
                    idx_j = rng.randint(0, m * p - 1)
                    idx_c = rng.randint(0, target_rank - 1)
                    old_val = V[idx_j, idx_c]
                    delta = rng.choice([-1, -0.5, 0.5, 1, 2, -2]) * alpha
                    new_val = old_val + delta
                    V[idx_j, idx_c] = new_val
                    matrix = 'V'
                    row, col = idx_j, idx_c
                    old_v, new_v = old_val, new_val
                else:
                    idx_k = rng.randint(0, p * n - 1)
                    idx_c = rng.randint(0, target_rank - 1)
                    old_val = W[idx_k, idx_c]
                    delta = rng.choice([-1, -0.5, 0.5, 1, 2, -2]) * alpha
                    new_val = old_val + delta
                    W[idx_k, idx_c] = new_val
                    matrix = 'W'
                    row, col = idx_k, idx_c
                    old_v, new_v = old_val, new_val
                
                # Evaluate error (full recomputation is expensive; sample for speed)
                if it % 50 == 0 or it < 10:
                    err = 0.0
                    # Sample a subset for speed
                    sample_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
                    rng.shuffle(sample_indices)
                    sample_size = min(20, len(sample_indices))
                    for (i, j, k) in sample_indices[:sample_size]:
                        s = 0.0
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        for c in range(target_rank):
                            s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                        err += (s - 1.0) ** 2
                    avg_err = err / sample_size
                else:
                    avg_err = None
                
                # Accept if improvement or random acceptance
                # For simplicity, use a simple threshold
                if avg_err is not None and avg_err > 2.0:
                    # Revert
                    if matrix == 'U':
                        U[row, col] = old_v
                    elif matrix == 'V':
                        V[row, col] = old_v
                    else:
                        W[row, col] = old_v
            
            # Final verification with full check
            if time.time() > deadline - 2:
                break
            
            if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-4):
                # Try to clean up to integer/half-integer
                U_int = np.round(U * 2) / 2.0
                V_int = np.round(V * 2) / 2.0
                W_int = np.round(W * 2) / 2.0
                
                if _verify_fast(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p, tol=1e-6):
                    best_in_round = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    best_in_round_rank = target_rank
                    break
                elif _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
                    best_in_round = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_in_round_rank = target_rank
                    break
        
        if best_in_round is not None and best_in_round_rank < best_rank:
            best_rank = best_in_round_rank
            best_result = best_in_round
    
    # Convert to the required format
    return best_result


# Keep the EVOLVE-BLOCK markers as expected
# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END

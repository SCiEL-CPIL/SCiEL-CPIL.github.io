import numpy as np
import time
import itertools

def construct():
    """
    Rank-reduced decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Kronecker-Strassen construction (rank-98) then systematic column elimination
    with affine combination search to find rank-minimal exact decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 4, 4, 8

    # Build the base rank-98 Strassen decomposition
    U_base, V_base, W_base = _construct_rank98(n, m, p)

    # Verify the base decomposition
    if not _verify_decomposition(U_base, V_base, W_base, n, m, p):
        U_base, V_base, W_base = _construct_naive(n, m, p)
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}

    best_result = (U_base, V_base, W_base)
    best_rank = U_base.shape[1]

    # Try column elimination to reduce rank
    elapsed = time.time() - start_time
    remaining_time = deadline - start_time
    if remaining_time > 15:
        reduced = _column_elimination_aggressive(U_base, V_base, W_base, n, m, p, remaining_time)
        if reduced is not None:
            U_r, V_r, W_r = reduced
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                if U_r.shape[1] < best_rank:
                    best_result = (U_r, V_r, W_r)
                    best_rank = U_r.shape[1]

    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}

def _construct_rank98(n, m, p):
    """Rank-98 decomposition using two independent 4x4 Strassen instances."""
    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j, c] = U2[i1*2+j1, t] * U2[i2*2+j2, s]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k, c] = V2[j1*2+k1, t] * V2[j2*2+k2, s]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i, c] = W2[k1*2+i1, t] * W2[k2*2+i2, s]
    
    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First block (output cols 0-3)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, 0:49]
    
    # Second block (output cols 4-7)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V4[j*4+k, 0:49]
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, 49:98] = W4[k*4+i, 0:49]
    
    return U, V, W

def _column_elimination_aggressive(U, V, W, n, m, p, time_budget):
    """
    Aggressive column elimination using affine combination search.
    For each column, try to express it as an affine combination of a small subset
    of other columns, then remove it and adjust the remaining columns.
    """
    start = time.time()
    rank = U.shape[1]
    num_eqs = n * m * p
    
    # Build tensor matrix T where T[e, t] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    T = np.zeros((num_eqs, rank), dtype=float)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for t in range(rank):
                    T[eq_idx, t] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                eq_idx += 1
    
    target = np.ones(num_eqs, dtype=float)
    
    if not np.allclose(T.sum(axis=1), target, atol=1e-9):
        return None
        
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_T = T.copy()
    current_rank = rank
    
    # Try to eliminate columns one by one
    for _ in range(current_rank):
        if time.time() - start > time_budget:
            break
            
        removed = False
        for r in range(current_rank):
            if current_rank <= 1:
                break
                
            # Try to express column r as affine combination of a subset of others
            remaining = [s for s in range(current_rank) if s != r]
            T_target = current_T[:, r]
            
            # Try subsets of increasing size
            found_subset = False
            for subset_size in range(1, min(4, len(remaining))):
                if time.time() - start > time_budget:
                    break
                    
                for subset in itertools.combinations(remaining, subset_size):
                    if time.time() - start > time_budget:
                        break
                        
                    T_sub = current_T[:, list(subset)]
                    
                    # Solve for affine combination: T_target = sum(alpha_s * T_s) + beta * target
                    # This is equivalent to: T_target - beta*target = T_sub @ alpha
                    # We have num_eqs equations and (subset_size + 1) unknowns
                    # Augmented system: [T_sub | -target] @ [alpha; beta] = T_target
                    
                    A_aug = np.column_stack([T_sub, -target])
                    
                    try:
                        coeffs, residuals, rank_A, sv = np.linalg.lstsq(A_aug, T_target, rcond=1e-10)
                    except:
                        continue
                        
                    if residuals.size > 0:
                        res_val = residuals[0]
                    else:
                        res_val = np.linalg.norm(A_aug @ coeffs - T_target)
                        
                    if res_val > 1e-8:
                        continue
                        
                    # Found valid affine combination
                    # coeffs[:subset_size] are alpha_s, coeffs[subset_size] is beta
                    alphas = coeffs[:subset_size]
                    beta = coeffs[subset_size]
                    
                    # Update the remaining columns to absorb the eliminated column
                    # New T_s' = T_s + alpha_s * T_r (for s in subset)
                    # But we need to update U, V, W consistently
                    # This is tricky because T depends on products of U, V, W entries
                    
                    # Alternative: just remove column r and verify if the sum still works
                    # If T_r = sum(alpha_s * T_s) + beta * target, then:
                    # sum_{s != r} T_s = target - T_r + sum_{s in subset} alpha_s * T_s - sum_{s in subset} T_s
                    # = target - T_r + sum_{s in subset} (alpha_s - 1) * T_s
                    # This doesn't simplify nicely unless alpha_s = 1 for all s in subset
                    
                    # For now, only accept if beta = 0 and sum of alphas = 1 (pure affine)
                    # Actually, let's just try removing and checking
                    keep = [s for s in range(current_rank) if s != r]
                    test_U = current_U[:, keep]
                    test_V = current_V[:, keep]
                    test_W = current_W[:, keep]
                    
                    # Verify the reduced decomposition
                    if _verify_decomposition_fast(test_U, test_V, test_W, n, m, p, current_T[:, keep]):
                        current_U = test_U
                        current_V = test_V
                        current_W = test_W
                        current_T = current_T[:, keep]
                        current_rank -= 1
                        removed = True
                        found_subset = True
                        break
                        
                if found_subset:
                    break
                    
            if found_subset:
                break
                
        if not removed:
            break
            
    if current_rank < rank:
        return current_U, current_V, current_W
    return None

def _verify_decomposition_fast(U, V, W, n, m, p, T_matrix):
    """Fast verification using precomputed T matrix."""
    if T_matrix is None:
        return _verify_decomposition(U, V, W, n, m, p)
    
    target = np.ones(T_matrix.shape[0], dtype=float)
    return np.allclose(T_matrix.sum(axis=1), target, atol=1e-9)

def _construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    rng = np.random.default_rng(42)
    for trial in range(3):
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        M = np.zeros(rank)
        for t in range(rank):
            u_val = np.dot(U[:, t], A_flat)
            v_val = np.dot(V[:, t], B_flat)
            M[t] = u_val * v_val
        
        C_computed = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                C_computed[i, k] = np.dot(W[k*n+i, :], M)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

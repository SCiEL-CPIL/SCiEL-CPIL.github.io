"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _tensor_from_matrices(U, V, W):
    """Compute the tensor from factor matrices."""
    n_a, rank = U.shape
    n_b = V.shape[0]
    n_c = W.shape[0]
    T = np.zeros((n_a, n_b, n_c))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    T_recon = _tensor_from_matrices(U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _try_rank16_from_rank17(U17, V17, W17, T, deadline):
    """Try to find a rank-16 decomposition by dropping one term and optimizing."""
    n_a, _ = U17.shape
    n_b = V17.shape[0]
    n_c = W17.shape[0]
    
    rng = np.random.default_rng(42)
    
    for r0 in range(17):
        if time.time() > deadline - 20:
            break
        
        # Keep all terms except r0
        keep = [r for r in range(17) if r != r0]
        U0 = U17[:, keep].copy()
        V0 = V17[:, keep].copy()
        W0 = W17[:, keep].copy()
        
        # Residual to absorb
        T_current = _tensor_from_matrices(U0, V0, W0)
        residual = T - T_current
        
        # Try iterative optimization: adjust the 16 terms to absorb the residual
        # Use a simple gradient-free approach with random restarts
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        
        for restart in range(30):
            if time.time() > deadline - 15:
                break
            
            U_try = U0.copy()
            V_try = V0.copy()
            W_try = W0.copy()
            
            # Add small random perturbations
            if restart > 0:
                scale = 0.5 / (1 + restart * 0.3)
                U_try += rng.normal(0, scale, U0.shape)
                V_try += rng.normal(0, scale, V0.shape)
                W_try += rng.normal(0, scale, W0.shape)
            
            # Iterative refinement: for each iteration, try to reduce the error
            # by adjusting one term at a time
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                T_try = _tensor_from_matrices(U_try, V_try, W_try)
                err = np.sum((T_try - T)**2)
                
                if err < 1e-12:
                    best_err = err
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    break
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                
                # Pick a random term and coordinate to adjust
                r_idx = rng.integers(0, 16)
                factor = rng.integers(0, 3)  # 0=U, 1=V, 2=W
                if factor == 0:
                    coord = rng.integers(0, n_a)
                    delta = rng.normal(0, 0.1)
                    U_try[coord, r_idx] += delta
                elif factor == 1:
                    coord = rng.integers(0, n_b)
                    delta = rng.normal(0, 0.1)
                    V_try[coord, r_idx] += delta
                else:
                    coord = rng.integers(0, n_c)
                    delta = rng.normal(0, 0.1)
                    W_try[coord, r_idx] += delta
            
            if best_err < 1e-12:
                break
        
        if best_err < 1e-9:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def _try_rank16_hosvd(T, deadline):
    """Try HOSVD-based approach to find rank-16 decomposition."""
    n_a, n_b, n_c = T.shape
    
    # Flatten and try SVD-based approach
    # This is a heuristic - HOSVD gives approximate decomposition
    # We'll use it to initialize a local search
    
    # For each mode, compute the best rank-16 approximation
    # Mode-1 unfolding: T1 (n_a x n_b*n_c)
    T1 = T.reshape(n_a, n_b * n_c)
    U1, S1, Vt1 = np.linalg.svd(T1, full_matrices=False)
    
    # Mode-2 unfolding: T2 (n_b x n_a*n_c)
    T2 = T.transpose(1, 0, 2).reshape(n_b, n_a * n_c)
    U2, S2, Vt2 = np.linalg.svd(T2, full_matrices=False)
    
    # Mode-3 unfolding: T3 (n_c x n_a*n_b)
    T3 = T.transpose(2, 0, 1).reshape(n_c, n_a * n_b)
    U3, S3, Vt3 = np.linalg.svd(T3, full_matrices=False)
    
    # Use the top 16 singular vectors as initial factors
    # This is very approximate - we'll refine with local search
    U_init = U1[:, :16]
    V_init = U2[:, :16]
    W_init = U3[:, :16]
    
    # Refine with local search
    rng = np.random.default_rng(123)
    best_err = np.inf
    best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
    
    for it in range(500):
        if time.time() > deadline - 10:
            break
        
        T_try = _tensor_from_matrices(best_U, best_V, best_W)
        err = np.sum((T_try - T)**2)
        
        if err < best_err:
            best_err = err
        
        if err < 1e-12:
            break
        
        # Random perturbation
        scale = 0.1 / (1 + it * 0.05)
        best_U += rng.normal(0, scale, best_U.shape)
        best_V += rng.normal(0, scale, best_V.shape)
        best_W += rng.normal(0, scale, best_W.shape)
    
    if best_err < 1e-9:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    T = _build_tensor(n, m, p)
    
    # Build the rank-17 solution
    rank = 17
    U17 = np.zeros((n_a, rank))
    V17 = np.zeros((n_b, rank))
    W17 = np.zeros((n_c, rank))
    
    terms = [
        # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
        ([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U17[a, r] = val
        for b, val in zip(v_idx, v_val):
            V17[b, r] = val
        for c, val in zip(w_idx, w_val):
            W17[c, r] = val
    
    # Verify rank-17 solution
    if not _verify_decomposition(U17, V17, W17, T):
        # Fallback to trivial rank-18
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to find rank-16
    result_16 = _try_rank16_from_rank17(U17, V17, W17, T, deadline)
    if result_16 is not None:
        # Verify the rank-16 solution
        U16 = np.array(result_16["U"])
        V16 = np.array(result_16["V"])
        W16 = np.array(result_16["W"])
        if _verify_decomposition(U16, V16, W16, T):
            return result_16
    
    # Try HOSVD-based approach
    result_16_hosvd = _try_rank16_hosvd(T, deadline)
    if result_16_hosvd is not None:
        U16 = np.array(result_16_hosvd["U"])
        V16 = np.array(result_16_hosvd["V"])
        W16 = np.array(result_16_hosvd["W"])
        if _verify_decomposition(U16, V16, W16, T):
            return result_16_hosvd
    
    # Fall back to rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END

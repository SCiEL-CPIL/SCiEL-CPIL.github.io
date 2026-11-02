"""
Program to find rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
Uses Strassen-based rank-28 construction as baseline.
Implements a refined search for rank 27 using Alternating Least Squares (ALS).
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_28():
    """
    Constructs the standard rank-28 Strassen decomposition for 2x4 * 4x4.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms for 2x2 matrices:
    # M1 = (x0+x3)(y0+y3)
    # M2 = (x2+x3)y0
    # M3 = x0(y1-y3)
    # M4 = x3(y2-y0)
    # M5 = (x0+x1)y3
    # M6 = (x2-x0)(y0+y1)
    # M7 = (x1-x3)(y2+y3)
    
    # Local indices: 0:00, 1:01, 2:10, 3:11
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    def map_u_a(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        global_c = c if side == 'L' else 2 + c
        return r * m + global_c
        
    def map_v_b(block, local_idx):
        r_off, c_off = {
            'TL': (0, 0), 'TR': (0, 2), 'BL': (2, 0), 'BR': (2, 2)
        }[block]
        r = local_idx // 2
        c = local_idx % 2
        global_r = r_off + r
        global_c = c_off + c
        return global_r * p + global_c
        
    def map_w_c(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        global_c = c if side == 'L' else 2 + c
        return global_c * n + r

    def apply_strassen(a_side, b_block, c_side, start_col):
        for t in strassen_terms:
            u_local, v_local, w_local = t
            col = start_col
            
            for idx, coeff in u_local:
                U[map_u_a(a_side, idx), col] += coeff
            for idx, coeff in v_local:
                V[map_v_b(b_block, idx), col] += coeff
            for idx, coeff in w_local:
                W[map_w_c(c_side, idx), col] += coeff
            
            start_col += 1
        
        return start_col

    col = apply_strassen('L', 'TL', 'L', 0)
    col = apply_strassen('R', 'BL', 'L', col)
    col = apply_strassen('L', 'TR', 'R', col)
    col = apply_strassen('R', 'BR', 'R', col)
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=4, p=4, trials=5, seed=42):
    """Verify exact decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        A = rng.random((n, m))
        B = rng.random((m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(rank):
                    for j in range(m):
                        acc += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                C_computed[i, k] = acc
                
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _attempt_rank_27_search(start_time, deadline=145.0):
    """
    Attempts to find a rank-27 decomposition using ALS.
    """
    n, m, p = 2, 4, 4
    target_rank = 27
    
    U28, V28, W28 = _build_strassen_28()
    
    # Try dropping each of the 28 columns
    candidates = list(range(28))
    
    for drop_idx in candidates:
        if time.time() > deadline - 5:
            break
            
        # Initialize with rank 27 (drop column drop_idx)
        U = np.delete(U28, drop_idx, axis=1).copy()
        V = np.delete(V28, drop_idx, axis=1).copy()
        W = np.delete(W28, drop_idx, axis=1).copy()
        
        # ALS Iterations
        max_iter = 200
        converged = False
        
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
                
            # Update U
            # For each i, j: solve U[i,j,:] @ M[i,j] = 1
            # M[i,j] is 27 x p (k dimension)
            # M[i,j, r, k] = V[j*p+k, r] * W[k*n+i, r]
            for i in range(n):
                for j in range(m):
                    M = np.zeros((target_rank, p))
                    for r in range(target_rank):
                        for k in range(p):
                            M[r, k] = V[j * p + k, r] * W[k * n + i, r]
                    # Solve U[i,j,:] @ M = ones(p)
                    # U[i,j,:] is 1 x 27
                    # M is 27 x p
                    # Use least squares
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
                        
            # Update V
            # For each j, k: solve V[j,k,:] @ M[j,k] = 1
            # M[j,k] is 27 x n (i dimension)
            # M[j,k, r, i] = U[i*m+j, r] * W[k*n+i, r]
            for j in range(m):
                for k in range(p):
                    M = np.zeros((target_rank, n))
                    for r in range(target_rank):
                        for i in range(n):
                            M[r, i] = U[i * m + j, r] * W[k * n + i, r]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
                        
            # Update W
            # For each k, i: solve W[k,i,:] @ M[k,i] = 1
            # M[k,i] is 27 x m (j dimension)
            # M[k,i, r, j] = U[i*m+j, r] * V[j*p+k, r]
            for k in range(p):
                for i in range(n):
                    M = np.zeros((target_rank, m))
                    for r in range(target_rank):
                        for j in range(m):
                            M[r, j] = U[i * m + j, r] * V[j * p + k, r]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        residual += (val - 1.0)**2
                        
            if residual < 1e-10:
                converged = True
                break
                
        if converged:
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    U28, V28, W28 = _build_strassen_28()
    best_U, best_V, best_W = U28.tolist(), V28.tolist(), W28.tolist()
    best_rank = 28
    
    if time.time() - start_time < 5:
        result_27 = _attempt_rank_27_search(start_time, deadline)
        if result_27 is not None:
            best_U, best_V, best_W = result_27
            best_rank = 27
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END

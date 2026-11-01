"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T with shape (n*p, n*m, m*p).
    T[output, A, B] = 1 if indices are compatible, else 0.
    output index: i*p + k (row i, col k of C)
    A index: i*m + j (row i, col j of A)
    B index: j*p + k (row j, col k of B)
    Compatible if: A row == output row, A col == B row, B col == output col
    """
    T = np.zeros((n * p, n * m, m * p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                out_idx = i * p + k
                a_idx = i * m + j
                b_idx = j * p + k
                T[out_idx, a_idx, b_idx] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Verify that sum_c U[o,c]*V[a,c]*W[b,c] == T[o,a,b] for all o,a,b."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Contract: result[o,a,b] = sum_c U[o,c] * V[a,c] * W[b,c]
    # Use einsum: 'oc,ac,bc->oab'
    result = np.einsum('oc,ac,bc->oab', U, V, W)
    return np.allclose(result, T, atol=tol)

def _round_to_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close, else return x."""
    rounded = np.round(2 * x) / 2.0
    if np.allclose(x, rounded, atol=tol):
        return rounded
    return x

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build the tensor
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Residual function for scipy
    def residual_func(params, r):
        U = params[:n*p*r].reshape(n*p, r)
        V = params[n*p*r:n*p*r+n*m*r].reshape(n*m, r)
        W = params[n*p*r+n*m*r:].reshape(m*p, r)
        result = np.einsum('oc,ac,bc->oab', U, V, W)
        return result.flatten() - T_flat

    # Try ranks from 20 up to 35
    # We use a smooth surrogate (sum of squares) and local optimization
    for r in range(20, rank_school):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            
            # Initialize with small random values
            np.random.seed(42 + attempt + r)
            init_params = np.random.randn(n*p*r + n*m*r + m*p*r) * 0.1
            
            try:
                res = opt.least_squares(residual_func, init_params, args=(r,), method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if res.cost < 1e-8:
                # Extract U, V, W
                U = res.x[:n*p*r].reshape(n*p, r)
                V = res.x[n*p*r:n*p*r+n*m*r].reshape(n*m, r)
                W = res.x[n*p*r+n*m*r:].reshape(m*p, r)
                
                # Round to half-integers
                U_r = _round_to_half_int(U)
                V_r = _round_to_half_int(V)
                W_r = _round_to_half_int(W)
                
                if _verify(U_r, V_r, W_r, T):
                    found = True
                    if r < best_rank:
                        best_rank = r
                        best_U = U_r.copy()
                        best_V = V_r.copy()
                        best_W = W_r.copy()
                    break
        
        if found:
            # If we found a solution at rank r, we can stop or continue to find lower?
            # The loop goes from 20 up, so we want the lowest r.
            # If we found one at 20, we can break.
            # But we might have missed a lower one if we started at 20?
            # The loop starts at 20. If 20 works, break.
            # If 21 works, we continue to 22? No, we want minimal rank.
            # So if we find a solution at r, it is the minimal one we found so far?
            # No, we iterate r increasing. So the first r we find is the minimal.
            # Thus we can break.
            break
    
    # Convert to lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

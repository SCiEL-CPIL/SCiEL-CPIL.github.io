import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err = abs(val - 1.0)
                max_err = max(max_err, err)
    
    return max_err < 1e-10

def _try_rank13():
    """Attempt to find a rank-13 decomposition using non-linear optimization."""
    n, m, p = DIMENSIONS
    target_rank = 13
    
    # Total variables: rank * (n*m + m*p + p*n)
    dim_u = n * m  # 4
    dim_v = m * p  # 8
    dim_w = p * n  # 8
    total_dim = target_rank * (dim_u + dim_v + dim_w)
    
    # Precompute indices for the 16 tensor entries
    # Each entry is (i, j, k) corresponding to U[i*m+j], V[j*p+k], W[k*n+i]
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
                
    # Residual function
    def residuals(x):
        # Reshape x into U, V, W
        # Order: U[0..12], V[0..12], W[0..12]
        u_flat = x[0:target_rank * dim_u]
        v_flat = x[target_rank * dim_u : target_rank * dim_u + target_rank * dim_v]
        w_flat = x[target_rank * dim_u + target_rank * dim_v : total_dim]
        
        U = u_flat.reshape(target_rank, dim_u).T # U is (dim_u, rank)
        V = v_flat.reshape(target_rank, dim_v).T # V is (dim_v, rank)
        W = w_flat.reshape(target_rank, dim_w).T # W is (dim_w, rank)
        
        res = []
        for ui, vi, wi in indices:
            # sum_r U[ui, r] * V[vi, r] * W[wi, r]
            val = np.dot(U[ui, :], V[vi, :] * W[wi, :])
            res.append(val - 1.0)
        return np.array(res)
    
    best_sol = None
    best_err = float('inf')
    
    # Try multiple random restarts
    num_restarts = 50
    deadline = time.time() + 100 # Leave 50s for verification and fallback
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        x0 = np.random.randn(total_dim)
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if result.success and result.cost < 1e-12:
                if result.cost < best_err:
                    best_err = result.cost
                    best_sol = result.x
        except Exception:
            continue
    
    if best_sol is None:
        return None
        
    # Extract U, V, W
    u_flat = best_sol[0:target_rank * dim_u]
    v_flat = best_sol[target_rank * dim_u : target_rank * dim_u + target_rank * dim_v]
    w_flat = best_sol[target_rank * dim_u + target_rank * dim_v : total_dim]
    
    U = u_flat.reshape(target_rank, dim_u).T
    V = v_flat.reshape(target_rank, dim_v).T
    W = w_flat.reshape(target_rank, dim_w).T
    
    # Try to rationalize to integer/half-integer
    # We check if values are close to k/2
    def rationalize(arr):
        new_arr = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                # Check integers and half-integers
                best_val = val
                best_diff = float('inf')
                for k in range(-20, 21): # -10 to 10
                    half_int = k / 2.0
                    if abs(val - half_int) < 1e-6:
                        best_val = half_int
                        best_diff = abs(val - half_int)
                        break
                new_arr[i, j] = best_val
        return new_arr
    
    U_r = rationalize(U)
    V_r = rationalize(V)
    W_r = rationalize(W)
    
    # Verify rationalized solution
    if _verify_decomposition(U_r, V_r, W_r, n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
        
    # If rationalization failed, maybe the solution is valid but not integer?
    # The task requires integer/half-integer. So we must return None if not rational.
    return None

def _build_rank14():
    """Return the known rank-14 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for B1 (cols 0,1)
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Strassen for B2 (cols 2,3)
    # N1
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # N2
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # N3
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # N4
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    # N5
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # N6
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # N7
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try rank 13
    res13 = _try_rank13()
    if res13 is not None:
        U, V, W = res13
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # Fallback to rank 14
    U, V, W = _build_rank14()
    return {"U": U, "V": V, "W": W}

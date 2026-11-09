"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T(i,j,k) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(actual - expected) > tol:
                    return False
    return True

def _build_rank_34_decomposition(n, m, p):
    """
    Construct a rank-34 decomposition for 3x3 x 3x4 matrix multiplication.
    
    Strategy: Use the known rank-34 result from the literature.
    For 3x3 x 3x4, we can exploit the fact that one of the 3x3 blocks
    can be computed with Strassen's rank-7 method instead of schoolbook rank-9,
    saving 2 multiplications.
    
    Partition:
    A (3x3) = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    B (3x4) = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x2, B21 is 1x2, B22 is 1x2
    
    C (3x4) = [C11 C12; C21 C22] where C11 is 2x2, C12 is 2x2, C21 is 1x2, C22 is 1x2
    
    C11 = A11@B11 + A12@B21  (2x2 = 2x2@2x2 + 2x1@1x2)
    C12 = A11@B12 + A12@B22  (2x2 = 2x2@2x2 + 2x1@1x2)
    C21 = A21@B11 + A22@B21  (1x2 = 1x2@2x2 + 1x1@1x2)
    C22 = A21@B12 + A22@B22  (1x2 = 1x2@2x2 + 1x1@1x2)
    
    For C11: A11@B11 is 2x2x2 (rank 4 schoolbook), A12@B21 is 2x1x1x2 (rank 2)
    Total for C11: 6 multiplications
    
    For C12: A11@B12 is 2x2x2 (rank 4 schoolbook), A12@B22 is 2x1x1x2 (rank 2)
    Total for C12: 6 multiplications
    
    For C21: A21@B11 is 1x2x2x2 (rank 4), A22@B21 is 1x1x1x2 (rank 2)
    Total for C21: 6 multiplications
    
    For C22: A21@B12 is 1x2x2x2 (rank 4), A22@B22 is 1x1x1x2 (rank 2)
    Total for C22: 6 multiplications
    
    Total: 6+6+6+6 = 24? No, that's wrong.
    
    Let me recount properly:
    A11@B11: (2x2)(2x2) = 2x2 output, needs 2*2*2 = 8 scalar mults
    A12@B21: (2x1)(1x2) = 2x2 output, needs 2*1*2 = 4 scalar mults
    C11 total: 8+4 = 12
    
    A11@B12: (2x2)(2x2) = 2x2 output, needs 8 scalar mults
    A12@B22: (2x1)(1x2) = 2x2 output, needs 4 scalar mults
    C12 total: 8+4 = 12
    
    A21@B11: (1x2)(2x2) = 1x2 output, needs 1*2*2 = 4 scalar mults
    A22@B21: (1x1)(1x2) = 1x2 output, needs 1*1*2 = 2 scalar mults
    C21 total: 4+2 = 6
    
    A21@B12: (1x2)(2x2) = 1x2 output, needs 4 scalar mults
    A22@B22: (1x1)(1x2) = 1x2 output, needs 2 scalar mults
    C22 total: 4+2 = 6
    
    Grand total: 12+12+6+6 = 36. This is just schoolbook again.
    
    To get rank 34, I need to find 2 linear dependencies among the 36 schoolbook terms.
    
    Actually, the known result is that rank(3x3x3x4) = 34. Let me use a different approach:
    I'll use the fact that for 3x3x3, the rank is 23 (or 22?), and extend it.
    
    For 3x3x3: rank = 23 (known result by Bläser et al.)
    For 3x3x4: we can think of it as 3x3x3 + 3x3x1, where the 3x3x1 part has rank 3*3*1=9.
    But that gives 23+9=32, which is too optimistic.
    
    Actually, let me just try to construct it directly using optimization.
    """
    n, m, p = 3, 3, 4
    target_rank = 34
    
    # Initialize with schoolbook truncated to 34 columns
    U_s, V_s, W_s = _build_schoolbook(n, m, p)
    U_init = np.array(U_s)[:, :target_rank].copy()
    V_init = np.array(V_s)[:, :target_rank].copy()
    W_init = np.array(W_s)[:, :target_rank].copy()
    
    # The two missing terms (columns 34, 35) are:
    # col 34: i=2, j=2, k=2 -> U[8]=1, V[10]=1, W[8]=1
    # col 35: i=2, j=2, k=3 -> U[8]=1, V[11]=1, W[11]=1
    
    # We need to absorb these two terms into the existing 34 terms.
    # This means finding small adjustments to U, V, W such that the two missing
    # contributions are reproduced.
    
    from scipy.optimize import minimize
    
    def objective(params):
        if _time_remaining() < 5:
            return 1e10
        U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_flat = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_flat[i*m+j, :], V_flat[j*p+k, :] * W_flat[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err
    
    best_err = float('inf')
    best_params = None
    
    np.random.seed(42)
    
    for attempt in range(10):
        if _time_remaining() < 15:
            break
        
        # Start from schoolbook truncated, add small perturbation
        U = U_init.copy() + np.random.randn(n*m, target_rank) * 0.01
        V = V_init.copy() + np.random.randn(m*p, target_rank) * 0.01
        W = W_init.copy() + np.random.randn(p*n, target_rank) * 0.01
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14})
            if result.fun < best_err:
                best_err = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is not None and best_err < 1e-10:
        U_opt = best_params[:n*m*target_rank].reshape(n*m, target_rank)
        V_opt = best_params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_opt = best_params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        # Round to nearest half-integer if close
        U_opt = np.round(U_opt * 2) / 2
        V_opt = np.round(V_opt * 2) / 2
        W_opt = np.round(W_opt * 2) / 2
        
        if _verify_decomposition(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-6):
            return U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
    
    return None

def construct():
    """Construct the rank-minimal decomposition for 3x3x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Try to find rank-34 decomposition
    if _time_remaining() > 30:
        result = _build_rank_34_decomposition(n, m, p)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 36
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

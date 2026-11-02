import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline for safety
    
    n, m, p = 2, 4, 8
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    # First try to find a lower-rank decomposition using optimization
    # Then fall back to Strassen (rank 56) if that fails
    
    best_result = None
    best_rank = 64  # schoolbook rank
    
    # Try ranks from 48 to 55
    for rank in range(48, 56):
        if time.time() > deadline - 10:
            break
        
        result = _try_optimized_decomp(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found a valid lower-rank decomp, use it
    
    if best_result is not None:
        return best_result
    
    # Fall back to Strassen rank 56
    strassen_result = _strassen_decomp(n, m, p)
    if strassen_result is not None:
        return strassen_result
    
    # Final fallback: schoolbook
    return _schoolbook(n, m, p)

def _try_optimized_decomp(n, m, p, rank, deadline):
    """Try to find an exact rank-`rank` decomposition via optimization."""
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    
    # Target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Random initialization
    np.random.seed(42)
    U = np.random.randn(A_dim, rank) * 0.5
    V = np.random.randn(B_dim, rank) * 0.5
    W = np.random.randn(C_dim, rank) * 0.5
    
    def compute_residual(params):
        U_flat = params[:A_dim * rank].reshape(A_dim, rank)
        V_flat = params[A_dim * rank:A_dim * rank + B_dim * rank].reshape(B_dim, rank)
        W_flat = params[A_dim * rank + B_dim * rank:].reshape(C_dim, rank)
        
        # Compute residual: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                    residual += (val - 1.0) ** 2
        return residual
    
    def gradient(params):
        U_flat = params[:A_dim * rank].reshape(A_dim, rank)
        V_flat = params[A_dim * rank:A_dim * rank + B_dim * rank].reshape(B_dim, rank)
        W_flat = params[A_dim * rank + B_dim * rank:].reshape(C_dim, rank)
        
        grad_U = np.zeros_like(U_flat)
        grad_V = np.zeros_like(V_flat)
        grad_W = np.zeros_like(W_flat)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                    err = 2.0 * (val - 1.0)
                    grad_U[i*m+j] += err * (V_flat[j*p+k] * W_flat[k*n+i])
                    grad_V[j*p+k] += err * (U_flat[i*m+j] * W_flat[k*n+i])
                    grad_W[k*n+i] += err * (U_flat[i*m+j] * V_flat[j*p+k])
        
        return np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
    
    params0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    
    # Use L-BFGS-B for optimization
    try:
        res = minimize(compute_residual, params0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                               'maxls': 50, 'maxfun': 5000})
    except Exception:
        return None
    
    if time.time() > deadline - 5:
        return None
    
    if res.fun > 1e-6:
        return None
    
    # Extract and round to half-integers
    U_opt = res.x[:A_dim * rank].reshape(A_dim, rank)
    V_opt = res.x[A_dim * rank:A_dim * rank + B_dim * rank].reshape(B_dim, rank)
    W_opt = res.x[A_dim * rank + B_dim * rank:].reshape(C_dim, rank)
    
    # Round to nearest half-integer
    U_int = np.round(U_opt * 2) / 2
    V_int = np.round(V_opt * 2) / 2
    W_int = np.round(W_opt * 2) / 2
    
    # Verify exactness
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Try rounding to integers too
    U_int2 = np.round(U_opt)
    V_int2 = np.round(V_opt)
    W_int2 = np.round(W_opt)
    if _verify(U_int2.tolist(), V_int2.tolist(), W_int2.tolist(), n, m, p):
        return {"U": U_int2.tolist(), "V": V_int2.tolist(), "W": W_int2.tolist()}
    
    return None

def _strassen_decomp(n, m, p):
    """Strassen-based rank-56 decomposition."""
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    
    rank = 56
    U = np.zeros((A_dim, rank), dtype=int)
    V = np.zeros((B_dim, rank), dtype=int)
    W = np.zeros((C_dim, rank), dtype=int)
    
    col = 0
    
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for j_group in range(2):
        for k_group in range(4):
            for term in strassen_terms:
                A_coeffs, B_coeffs, C_coeffs = term
                
                for (li, lj), coeff in A_coeffs.items():
                    global_i = li
                    global_j = 2 * j_group + lj
                    U[global_i * m + global_j, col] += coeff
                
                for (lj, lk), coeff in B_coeffs.items():
                    global_j = 2 * j_group + lj
                    global_k = 2 * k_group + lk
                    V[global_j * p + global_k, col] += coeff
                
                for (li, lk), coeff in C_coeffs.items():
                    global_i = li
                    global_k = 2 * k_group + lk
                    W[global_k * n + global_i, col] += coeff
                
                col += 1
    
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _verify(U, V, W, n, m, p, tol=1e-6):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Vectorized verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

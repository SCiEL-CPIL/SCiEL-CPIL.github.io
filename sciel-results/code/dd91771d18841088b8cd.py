import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 4
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, c] = 1
                V_school[j * p + k, c] = 1
                W_school[k * n + i, c] = 1
                c += 1
    
    start_time = time.time()
    deadline = 110 # 10s margin for 120s limit
    
    # Try rank 24
    for target_rank in [24, 25, 26]:
        if time.time() > deadline:
            break
        decomp = _find_decomp(n, m, p, target_rank, deadline)
        if decomp is not None:
            return decomp
            
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def _find_decomp(n, m, p, rank, deadline):
    # Tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
        
    T_flat = T.flatten()
    
    def objective(x):
        U = x[:rank * n * m].reshape(rank, n * m)
        V = x[rank * n * m : rank * n * m + rank * m * p].reshape(rank, m * p)
        W = x[rank * n * m + rank * m * p :].reshape(rank, p * n)
        
        # Reconstruct
        # Sum over r of U[:, r] @ V[:, r] ... no, outer product
        # Tensor reconstruction:
        # T_recon[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        # Using einsum
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        error = np.linalg.norm(T_recon.flatten() - T_flat)
        return error ** 2
        
    # Initial guess
    np.random.seed(42)
    best_error = float('inf')
    best_x = None
    
    # Try a few restarts
    for restart in range(10):
        if time.time() > deadline:
            break
        x0 = np.random.randn(rank * (n * m + m * p + p * n))
        
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
        
        if res.fun < best_error:
            best_error = res.fun
            best_x = res.x
            
        if best_error < 1e-8:
            break
            
    if best_error < 1e-6:
        # Try to round to half-integers?
        # Or just return the floats?
        # Task says "integer/half-integer entries".
        # If I return floats, will it be accepted?
        # "exactly decompose". Floats are not exact.
        # I must round.
        
        x = best_x
        U = x[:rank * n * m].reshape(rank, n * m)
        V = x[rank * n * m : rank * n * m + rank * m * p].reshape(rank, m * p)
        W = x[rank * n * m + rank * m * p :].reshape(rank, p * n)
        
        # Round to nearest 0.5
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        # Verify
        T_recon = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
        if np.allclose(T_recon, T, atol=1e-5):
            return {
                "U": U_int.tolist(),
                "V": V_int.tolist(),
                "W": W_int.tolist()
            }
            
    return None

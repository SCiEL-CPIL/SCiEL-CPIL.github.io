"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

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

def _tensor_error(U, V, W, n, m, p):
    """Calculate the error of the tensor decomposition."""
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1
                
    T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    return np.linalg.norm(T_recon - T_target)

def _try_local_search(U, V, W, n, m, p, max_time=10.0):
    """Attempt to reduce rank via local search."""
    start_time = time.time()
    best_U, best_V, best_W = U, V, W
    best_error = _tensor_error(U, V, W, n, m, p)
    
    # Try to remove one term
    rank = U[0].__len__()
    for col in range(rank):
        if time.time() - start_time > max_time:
            break
            
        U_new = [row[:] for row in U]
        V_new = [row[:] for row in V]
        W_new = [row[:] for row in W]
        
        for r in range(n*m): U_new[r][col] = 0
        for r in range(m*p): V_new[r][col] = 0
        for r in range(p*n): W_new[r][col] = 0
        
        # Simple perturbation of other columns
        for _ in range(50):
            U_pert = [row[:] for row in U_new]
            V_pert = [row[:] for row in V_new]
            W_pert = [row[:] for row in W_new]
            
            for c in range(rank):
                if c == col: continue
                if np.random.random() < 0.2:
                    for r in range(n*m): U_pert[r][c] += np.random.normal(0, 0.1)
                    for r in range(m*p): V_pert[r][c] += np.random.normal(0, 0.1)
                    for r in range(p*n): W_pert[r][c] += np.random.normal(0, 0.1)
            
            err = _tensor_error(U_pert, V_pert, W_pert, n, m, p)
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U_pert, V_pert, W_pert
                
        if best_error < 1e-6:
            return best_U, best_V, best_W, best_error

    return best_U, best_V, best_W, best_error

def construct():
    """Construct the rank-minimal decomposition for 3x3x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    U, V, W = _build_schoolbook(n, m, p)
    
    # Attempt search
    U_new, V_new, W_new, err = _try_local_search(U, V, W, n, m, p, max_time=10.0)
    
    if err < 1e-6:
        return {"U": U_new, "V": V_new, "W": W_new}
    else:
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

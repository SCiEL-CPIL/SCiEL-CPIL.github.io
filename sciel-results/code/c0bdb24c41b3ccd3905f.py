# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def _strassen_rank7():
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    U[0, 0] += 1; U[3, 0] += 1; V[0, 0] += 1; V[3, 0] += 1; W[0, 0] += 1; W[3, 0] += 1
    U[2, 1] += 1; U[3, 1] += 1; V[0, 1] += 1; W[2, 1] += 1
    U[0, 2] += 1; V[2, 2] += 1; V[3, 2] -= 1; W[1, 2] += 1; W[3, 2] += 1
    U[3, 3] += 1; V[1, 3] += 1; V[0, 3] -= 1; W[0, 3] += 1; W[2, 3] += 1
    U[0, 4] += 1; U[1, 4] += 1; V[3, 4] += 1; W[0, 4] -= 1; W[1, 4] += 1
    U[2, 5] += 1; U[0, 5] -= 1; V[0, 5] += 1; V[1, 5] += 1; W[2, 6] += 1
    U[1, 6] += 1; U[3, 6] -= 1; V[2, 6] += 1; V[3, 6] += 1; W[0, 6] -= 1; W[3, 6] += 1
    return U, V, W

def _kron_decompose_4x4x4():
    U2, V2, W2 = _strassen_rank7()
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            U4[:, col] = np.kron(U2[:, r], U2[:, s])
            V4[:, col] = np.kron(V2[:, r], V2[:, s])
            W4[:, col] = np.kron(W2[:, r], W2[:, s])
    return U4, V4, W4

def _build_mm_tensor_4x4x4():
    n = m = p = 4
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, tol=1e-9):
    T = _build_mm_tensor_4x4x4()
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol, rtol=0)

def _try_find_lower_rank(target_rank, time_limit=30):
    T = _build_mm_tensor_4x4x4()
    # Flatten T
    T_flat = T.flatten()
    
    # Initial guess: Use the first target_rank columns of the Kronecker decomposition
    U0, V0, W0 = _kron_decompose_4x4x4()
    U_init = U0[:, :target_rank]
    V_init = V0[:, :target_rank]
    W_init = W0[:, :target_rank]
    
    # Reshape to 1D for optimizer
    def objective(x):
        U = x[:16*target_rank].reshape(16, target_rank)
        V = x[16*target_rank:32*target_rank].reshape(16, target_rank)
        W = x[32*target_rank:48*target_rank].reshape(16, target_rank)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_recon - T).flatten()
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    start_time = time.time()
    try:
        result = least_squares(objective, x0, max_nfev=1000)
    except Exception:
        return None
    
    if time.time() - start_time > time_limit:
        return None
        
    if result.cost < 1e-8:
        U = result.x[:16*target_rank].reshape(16, target_rank)
        V = result.x[16*target_rank:32*target_rank].reshape(16, target_rank)
        W = result.x[32*target_rank:48*target_rank].reshape(16, target_rank)
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if _verify(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Try to find a lower rank decomposition
    for rank in [37, 38, 39, 40]:
        if time.time() > deadline - 20:
            break
        result = _try_find_lower_rank(rank, time_limit=10)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
            
    # Fallback to seed result
    U, V, W = _kron_decompose_4x4x4()
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

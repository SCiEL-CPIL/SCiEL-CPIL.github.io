import numpy as np
from scipy.optimize import least_squares
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    size_U = n * m  # 6
    size_V = m * p  # 15
    size_W = p * n  # 10
    
    # Build target tensor
    targets_a = []
    targets_b = []
    targets_c = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                targets_a.append(a)
                targets_b.append(b)
                targets_c.append(c)
                
    targets_a = np.array(targets_a, dtype=int)
    targets_b = np.array(targets_b, dtype=int)
    targets_c = np.array(targets_c, dtype=int)
    
    T_target = np.zeros((size_U, size_V, size_W))
    T_target[targets_a, targets_b, targets_c] = 1.0
    T_target_flat = T_target.flatten()

    def verify_solution(U, V, W):
        T_full = np.einsum('ar,br,cr->abc', U, V, W)
        max_err = np.max(np.abs(T_full - T_target))
        return max_err < 1e-9

    def get_schoolbook():
        R = n * m * p
        U = np.zeros((size_U, R))
        V = np.zeros((size_V, R))
        W = np.zeros((size_W, R))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def try_reduce_rank(R, num_attempts=50):
        U_sb, V_sb, W_sb = get_schoolbook()
        R_sb = U_sb.shape[1]
        
        def residuals_from_flat(x):
            U = x[:size_U*R].reshape(size_U, R)
            V = x[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
            W = x[(size_U+size_V)*R:].reshape(size_W, R)
            T_full = np.einsum('ar,br,cr->abc', U, V, W)
            return (T_full - T_target).flatten()

        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Strategy 1: Schoolbook subset
            if attempt % 2 == 0:
                kept_indices = np.random.choice(R_sb, R, replace=False)
                U_init = np.zeros((size_U, R))
                V_init = np.zeros((size_V, R))
                W_init = np.zeros((size_W, R))
                for new_idx, old_idx in enumerate(kept_indices):
                    U_init[:, new_idx] = U_sb[:, old_idx]
                    V_init[:, new_idx] = V_sb[:, old_idx]
                    W_init[:, new_idx] = W_sb[:, old_idx]
            else:
                # Strategy 2: Random Gaussian
                U_init = np.random.randn(size_U, R)
                V_init = np.random.randn(size_V, R)
                W_init = np.random.randn(size_W, R)
                # Scale to match energy roughly
                U_init /= np.linalg.norm(U_init)
                V_init /= np.linalg.norm(V_init)
                W_init /= np.linalg.norm(W_init)
                U_init *= np.sqrt(R_sb / R)
                V_init *= np.sqrt(R_sb / R)
                W_init *= np.sqrt(R_sb / R)

            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(
                    residuals_from_flat,
                    x0,
                    method='trf',
                    max_nfev=200,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12
                )
                
                if res.cost < 1e-6:
                    x_sol = res.x
                    x_snapped = np.round(x_sol * 2) / 2.0
                    
                    U = x_snapped[:size_U*R].reshape(size_U, R)
                    V = x_snapped[size_U*R:(size_U+size_V)*R].reshape(size_V, R)
                    W = x_snapped[(size_U+size_V)*R:].reshape(size_W, R)
                    
                    if verify_solution(U, V, W):
                        return U.tolist(), V.tolist(), W.tolist()
                        
            except Exception:
                pass
                
        return None

    start_time = time.time()
    deadline = 110
    
    # Try rank 29 first as it's more likely to be found
    result = try_reduce_rank(29, num_attempts=100)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
        
    # Try rank 28 if time permits
    if time.time() < deadline - 20:
        result = try_reduce_rank(28, num_attempts=50)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # Fallback
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

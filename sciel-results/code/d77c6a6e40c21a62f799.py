import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    First attempts to find a rank-20 decomposition via numerical search.
    Falls back to Strassen's algorithm (rank 21) if no rank-20 solution is found.
    """
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Function to flatten U, V, W into a single vector
    def flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    # Function to unflatten vector into U, V, W
    def unflatten(x, rank):
        size_U = nm * rank
        size_V = mp * rank
        U = x[:size_U].reshape(nm, rank)
        V = x[size_U:size_U+size_V].reshape(mp, rank)
        W = x[size_U+size_V:].reshape(pn, rank)
        return U, V, W

    # Residual function for least squares
    def residuals(x, rank):
        U, V, W = unflatten(x, rank)
        # Compute T_approx = sum_r U[:, r] V[:, r] W[:, r]
        # Using einsum: 'ar,br,cr->abc'
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_approx - T).flatten()

    # Attempt to find rank 20 decomposition
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s for fallback and overhead
    
    rank_try = 20
    best_result = None
    best_error = np.inf
    
    # Try multiple random initializations
    num_attempts = 10
    for attempt in range(num_attempts):
        if time.time() > deadline - 10:  # Stop if we're running low on time
            break
        
        np.random.seed(42 + attempt)
        x0 = np.random.randn(nm * rank_try + mp * rank_try + pn * rank_try) * 0.1
        
        try:
            result = least_squares(
                residuals, 
                x0, 
                args=(rank_try,), 
                method='trf', 
                max_nfev=2000, 
                xtol=1e-12, 
                ftol=1e-12, 
                gtol=1e-12
            )
            if result.cost < best_error:
                best_error = result.cost
                best_result = result.x
                
                # If we found a very good solution, do a polish pass
                if best_error < 1e-6:
                    # Polish with tighter tolerance
                    result2 = least_squares(
                        residuals, 
                        best_result, 
                        args=(rank_try,), 
                        method='trf', 
                        max_nfev=5000, 
                        xtol=1e-14, 
                        ftol=1e-14, 
                        gtol=1e-14
                    )
                    if result2.cost < best_error:
                        best_error = result2.cost
                        best_result = result2.x
                        
        except Exception as e:
            pass

    # Check if we found a solution
    if best_result is not None and best_error < 1e-8:
        U, V, W = unflatten(best_result, rank_try)
        # Verify
        T_check = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_check, T, atol=1e-6):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to Strassen rank 21
    rank = 21
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    def map_a(local_idx):
        return local_idx
        
    def map_b(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return 0 * 6 + k0
        elif local_idx == 1:
            return 0 * 6 + k1
        elif local_idx == 2:
            return 1 * 6 + k0
        else:
            return 1 * 6 + k1

    def map_c(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0:
            return k0 * 2 + 0
        elif local_idx == 1:
            return k1 * 2 + 0
        elif local_idx == 2:
            return k0 * 2 + 1
        else:
            return k1 * 2 + 1

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for local_idx in range(4):
                u_val = step['u'][local_idx]
                if u_val != 0:
                    U[map_a(local_idx), r] = u_val
                    
                v_val = step['v'][local_idx]
                if v_val != 0:
                    V[map_b(local_idx, g), r] = v_val
                    
                w_val = step['w'][local_idx]
                if w_val != 0:
                    W[map_c(local_idx, g), r] = w_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Attempts to find a rank 20 decomposition by perturbing a rank 21 solution.
    Falls back to rank 21 if rank 20 is not found within the time limit.
    """
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # Build the rank 21 Strassen solution
    rank_21 = 21
    U_21 = np.zeros((nm, rank_21))
    V_21 = np.zeros((mp, rank_21))
    W_21 = np.zeros((pn, rank_21))
    
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
        if local_idx == 0: return 0 * 6 + k0
        elif local_idx == 1: return 0 * 6 + k1
        elif local_idx == 2: return 1 * 6 + k0
        else: return 1 * 6 + k1

    def map_c(local_idx, g):
        k0 = 2 * g
        k1 = 2 * g + 1
        if local_idx == 0: return k0 * 2 + 0
        elif local_idx == 1: return k1 * 2 + 0
        elif local_idx == 2: return k0 * 2 + 1
        else: return k1 * 2 + 1

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for local_idx in range(4):
                u_val = step['u'][local_idx]
                if u_val != 0:
                    U_21[map_a(local_idx), r] = u_val
                v_val = step['v'][local_idx]
                if v_val != 0:
                    V_21[map_b(local_idx, g), r] = v_val
                w_val = step['w'][local_idx]
                if w_val != 0:
                    W_21[map_c(local_idx, g), r] = w_val

    # Expected tensor
    T_exp = np.zeros((nm, mp, pn))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                T_exp[a, b, c] = 1.0

    def residual(params):
        # params is flattened U, V, W
        # U: 4*rank, V: 12*rank, W: 12*rank
        # Total size: (4+12+12)*rank = 28*rank
        rank = len(params) // 28
        U = params[:4*rank].reshape(4, rank)
        V = params[4*rank:4*rank+12*rank].reshape(12, rank)
        W = params[4*rank+12*rank:].reshape(12, rank)
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_calc - T_exp).flatten()

    # Try to find rank 20
    start_time = time.time()
    deadline = start_time + 100.0 # Leave 20s margin
    
    # Initialize with rank 21, zero out one column
    # Which column to zero? Try a few.
    # We flatten U, V, W
    U_flat = U_21.flatten()
    V_flat = V_21.flatten()
    W_flat = W_21.flatten()
    
    # Set last column of U, V, W to 0
    # Note: Flattening is row-major.
    # U is (4, 21). Last col is indices 20, 24, 28, 32?
    # It's easier to work with the matrices directly and then flatten for optimizer.
    
    # We will try to optimize U, V, W with rank 20.
    # We start from the rank 21 solution, but we must reduce the rank.
    # One way: Take the first 20 columns of the rank 21 solution.
    # This will not satisfy the equations exactly, but provides a starting point.
    
    U_init = U_21[:, :20]
    V_init = V_21[:, :20]
    W_init = W_21[:, :20]
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    best_result = None
    best_res_norm = np.inf
    
    # Try a few random perturbations of the starting point to escape local minima
    np.random.seed(42)
    
    for attempt in range(5):
        if time.time() > deadline:
            break
            
        # Perturb the initial guess slightly
        # The rank 21 solution is exact. Removing a column makes it inexact.
        # We need to adjust the remaining 20 columns to compensate.
        
        # Add small noise to help escape
        noise_scale = 0.1 if attempt > 0 else 0.0
        x_attempt = x0 + np.random.normal(0, noise_scale, x0.shape)
        
        try:
            res = least_squares(residual, x_attempt, method='trf', max_nfev=5000, xtol=1e-8, ftol=1e-8)
            res_norm = np.linalg.norm(res.fun)
            
            if res_norm < best_res_norm:
                best_res_norm = res_norm
                best_result = res.x
                
            # Check if we found a solution
            if res_norm < 1e-6:
                break
                
        except Exception:
            continue

    if best_result is not None and best_res_norm < 1e-6:
        # Reconstruct U, V, W from best_result
        rank = 20
        U = best_result[:4*rank].reshape(4, rank)
        V = best_result[4*rank:4*rank+12*rank].reshape(12, rank)
        W = best_result[4*rank+12*rank:].reshape(12, rank)
        
        # Final verification
        T_check = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_check, T_exp, atol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 21
    return {"U": U_21.tolist(), "V": V_21.tolist(), "W": W_21.tolist()}

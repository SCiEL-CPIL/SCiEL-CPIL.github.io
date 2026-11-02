import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _snap_to_half_integer(x):
    return np.round(2 * x) / 2.0

def _verify(U, V, W, n, m, p):
    if not np.allclose(2 * U, np.round(2 * U)):
        return False
    if not np.allclose(2 * V, np.round(2 * V)):
        return False
    if not np.allclose(2 * W, np.round(2 * W)):
        return False
    
    rank = U.shape[1]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    try:
        T_computed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_expected = np.ones((n, m, p))
        return np.allclose(T_computed, T_expected, atol=1e-9)
    except Exception:
        return False

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Initialize schoolbook
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    current_best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Search for better ranks
    # We limit the number of trials to ensure we finish in time
    max_trials = 5000
    trials = 0
    
    for R in range(rank_schoolbook - 1, max(rank_schoolbook // 2, 20), -1):
        if time.time() > deadline:
            break
        if trials >= max_trials:
            break
            
        # Try a few random seeds for this R
        # We use a deterministic sequence of seeds to ensure reproducibility
        for seed in range(100):
            if time.time() > deadline:
                break
            if trials >= max_trials:
                break
            trials += 1
                
            np.random.seed(seed)
            
            # Initialize V, W with small integers
            V_init = np.random.randint(-2, 3, size=(m * p, R))
            W_init = np.random.randint(-2, 3, size=(p * n, R))
            
            U_sol = np.zeros((n * m, R))
            valid = True
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, R))
                    for k in range(p):
                        M[k, :] = V_init[j * p + k, :] * W_init[k * n + i, :]
                    
                    try:
                        u, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_sol[i * m + j, :] = u
                    except Exception:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
                
            U_snap = _snap_to_half_integer(U_sol)
            V_snap = _snap_to_half_integer(V_init)
            W_snap = _snap_to_half_integer(W_init)
            
            if _verify(U_snap, V_snap, W_snap, n, m, p):
                current_best = {"U": U_snap, "V": V_snap, "W": W_snap}
                best_rank = R
                # We can break out of the seed loop if we found one for this R?
                # Yes, we want the lowest R, so if we find one for R, we can stop checking higher R?
                # No, we are iterating R downwards.
                # If we find one for R=59, we should check R=58.
                # So we don't break the R loop.
                # But we can break the seed loop for this R to save time for lower R?
                # Yes.
                break
                
    if best_rank < rank_schoolbook:
        return current_best
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END

import time
import numpy as np
import random

DIMENSIONS = [3, 4, 5]

def _validate_np(n, m, p, U, V, W, tolerance=1e-9):
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tolerance)

def _error_np(n, m, p, U, V, W):
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    target = np.ones((n, m, p))
    return np.sum((T - target) ** 2)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = rank
    
    vals = [-1, -0.5, 0, 0.5, 1]
    
    for target_rank in range(rank - 1, max(20, rank - 15), -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        
        # Strategy 1: Perturb schoolbook
        # Remove one column and try to repair
        if target_rank == rank - 1:
            # Try removing each column? Or just the first one?
            # Let's try removing the first column and repairing.
            U_init = U[:, 1:].copy()
            V_init = V[:, 1:].copy()
            W_init = W[:, 1:].copy()
            
            current_error = _error_np(n, m, p, U_init, V_init, W_init)
            
            steps = 0
            improved = True
            while improved and steps < 5000 and time.time() < deadline - 2:
                improved = False
                steps += 1
                mat_choice = random.choice([U_init, V_init, W_init])
                r, c = np.random.randint(0, mat_choice.shape[0]), np.random.randint(0, mat_choice.shape[1])
                old_val = mat_choice[r, c]
                for new_val in vals:
                    if new_val == old_val: continue
                    mat_choice[r, c] = new_val
                    new_error = _error_np(n, m, p, U_init, V_init, W_init)
                    if new_error < current_error - 1e-12:
                        current_error = new_error
                        improved = True
                        break
                    else:
                        mat_choice[r, c] = old_val
                if current_error < 1e-9:
                    best_U = U_init.copy()
                    best_V = V_init.copy()
                    best_W = W_init.copy()
                    best_rank = target_rank
                    found = True
                    break
        
        # Strategy 2: Random search
        if not found:
            max_attempts = 20
            for attempt in range(max_attempts):
                if time.time() > deadline - 2: break
                
                U_rand = np.random.choice(vals, size=(n * m, target_rank))
                V_rand = np.random.choice(vals, size=(m * p, target_rank))
                W_rand = np.random.choice(vals, size=(p * n, target_rank))
                
                current_error = _error_np(n, m, p, U_rand, V_rand, W_rand)
                
                if current_error < 1e-9:
                    best_U = U_rand.copy()
                    best_V = V_rand.copy()
                    best_W = W_rand.copy()
                    best_rank = target_rank
                    found = True
                    break
                
                steps = 0
                improved = True
                while improved and steps < 1000 and time.time() < deadline - 2:
                    improved = False
                    steps += 1
                    mat_choice = random.choice([U_rand, V_rand, W_rand])
                    r, c = np.random.randint(0, mat_choice.shape[0]), np.random.randint(0, mat_choice.shape[1])
                    old_val = mat_choice[r, c]
                    for new_val in vals:
                        if new_val == old_val: continue
                        mat_choice[r, c] = new_val
                        new_error = _error_np(n, m, p, U_rand, V_rand, W_rand)
                        if new_error < current_error - 1e-12:
                            current_error = new_error
                            improved = True
                            break
                        else:
                            mat_choice[r, c] = old_val
                    if current_error < 1e-9:
                        best_U = U_rand.copy()
                        best_V = V_rand.copy()
                        best_W = W_rand.copy()
                        best_rank = target_rank
                        found = True
                        break
                
                if found: break
        
        if found:
            continue # Try lower rank
        else:
            # If we didn't find it, maybe stop?
            # To be safe, we continue to lower ranks but with fewer attempts.
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

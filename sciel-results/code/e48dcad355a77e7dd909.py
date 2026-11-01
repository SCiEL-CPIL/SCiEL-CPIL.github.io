import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank_seed = n * m * p  # 24
    
    # Seed solution (Rank 24)
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1.0
                V_seed[j * p + k, col] = 1.0
                W_seed[k * n + i, col] = 1.0
                col += 1
    
    # Target: 24 equations
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    equations = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                equations.append((u_idx, v_idx, w_idx))
    
    def check_exact(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        for u_idx, v_idx, w_idx in equations:
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            if abs(val - 1.0) > tol:
                return False
        return True
    
    def compute_error(U, V, W):
        """Compute sum of squared errors."""
        err = 0.0
        for u_idx, v_idx, w_idx in equations:
            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
            err += (val - 1.0) ** 2
        return err
    
    def round_to_half(M):
        return np.round(M * 2) / 2
    
    def try_rank_discrete(r, max_iters=5000):
        """Try to find a solution with discrete entries {-1, -0.5, 0, 0.5, 1}."""
        if time.time() > deadline:
            return None
        
        candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        num_vars = r * (n * m + m * p + p * n)
        
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Random initialization with discrete values
            U = np.array([[random.choice(candidates) for _ in range(r)] for _ in range(n * m)])
            V = np.array([[random.choice(candidates) for _ in range(r)] for _ in range(m * p)])
            W = np.array([[random.choice(candidates) for _ in range(r)] for _ in range(p * n)])
            
            err = compute_error(U, V, W)
            if err < 1e-9:
                return U, V, W
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Local search: try flipping individual entries
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Pick a random variable to flip
                var_type = random.randint(0, 2)  # 0=U, 1=V, 2=W
                if var_type == 0:
                    row = random.randint(0, n * m - 1)
                    col = random.randint(0, r - 1)
                    old_val = U[row, col]
                    new_val = random.choice(candidates)
                    if new_val == old_val:
                        continue
                    U[row, col] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < err:
                        err = new_err
                        if err < 1e-9:
                            return U, V, W
                    else:
                        U[row, col] = old_val
                elif var_type == 1:
                    row = random.randint(0, m * p - 1)
                    col = random.randint(0, r - 1)
                    old_val = V[row, col]
                    new_val = random.choice(candidates)
                    if new_val == old_val:
                        continue
                    V[row, col] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < err:
                        err = new_err
                        if err < 1e-9:
                            return U, V, W
                    else:
                        V[row, col] = old_val
                else:
                    row = random.randint(0, p * n - 1)
                    col = random.randint(0, r - 1)
                    old_val = W[row, col]
                    new_val = random.choice(candidates)
                    if new_val == old_val:
                        continue
                    W[row, col] = new_val
                    new_err = compute_error(U, V, W)
                    if new_err < err:
                        err = new_err
                        if err < 1e-9:
                            return U, V, W
                    else:
                        W[row, col] = old_val
        
        # Check best solution
        if best_U is not None and compute_error(best_U, best_V, best_W) < 1e-9:
            return best_U, best_V, best_W
        
        return None
    
    # Try ranks from 20 to 23
    for r in range(20, 24):
        if time.time() > deadline:
            break
        
        sol = try_rank_discrete(r)
        if sol is not None:
            U, V, W = sol
            # Final verification
            if check_exact(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If no lower rank found, return seed
    return {"U": U_seed.tolist(), "V": V_seed.tolist(), "W": W_seed.tolist()}

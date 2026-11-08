"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    # Check exact start time
    deadline = time.time() + 130  # Reserve 20s margin for 150s total
    
    # Attempt numerical search for lower rank if sufficient time
    # Prioritize ranks that are more likely to be optimal or close to it.
    # Ranks 26-30 are good candidates for improvement over 34.
    # We allocate time dynamically.
    ranks_to_try = [26, 27, 28, 29, 30, 31, 32, 33]
    
    # Distribute time among attempts.
    # Give more time to lower ranks as they're harder
    time_per_rank = {26: 15.0, 27: 12.0, 28: 10.0, 29: 8.0, 30: 7.0, 31: 6.0, 32: 5.0, 33: 4.0}
    
    for target_rank in ranks_to_try:
        if time.time() >= deadline - 10:  # Keep 10s for final checks/fallback
            break
            
        tl = time_per_rank.get(target_rank, 5.0)
        # Don't exceed remaining time
        tl = min(tl, deadline - time.time() - 10)
        if tl < 3:
            break
            
        res = attempt_lower_rank(target_rank, time_limit=tl)
        if res is not None and valid_res(res, 3, 3, 4):
            return res
    
    # Fallback: return exactly rank 34 decomposition
    return construct_r34()

def attempt_lower_rank(target_rank, time_limit=10.0):
    n, m, p = 3, 3, 4
    R = target_rank
    start_time = time.time()
    
    # Build the target tensor indices
    indices_a = []
    indices_b = []
    indices_c = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                indices_a.append(a)
                indices_b.append(b)
                indices_c.append(c)
                
    indices_a = np.array(indices_a)
    indices_b = np.array(indices_b)
    indices_c = np.array(indices_c)
    
    # Objective function
    def objective(params):
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        vals = np.sum(U[indices_a, :] * V[indices_b, :] * W[indices_c, :], axis=1)
        error = np.sum((vals - 1.0)**2)
        return error
    
    # Use a structured initialization based on Kronecker product idea
    # 3x3x3 matmul has known rank-23 (Strassen-like), 1x3x4 has rank 3
    # But we'll try direct optimization with better initialization
    
    from scipy.optimize import minimize
    
    best_error = float('inf')
    
    # Try multiple strategies for initialization
    num_restarts = min(8, max(2, int(time_limit / 1.5)))
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 1:
            break
            
        np.random.seed(42 + restart * 137)
        
        # Strategy 1: Sparse random initialization
        if restart % 3 == 0:
            # Create sparse initialization
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            # Randomly assign non-zero entries
            for r in range(R):
                # Each factor gets 2-3 non-zero entries
                u_idx = np.random.choice(n*m, size=np.random.randint(2, 4), replace=False)
                v_idx = np.random.choice(m*p, size=np.random.randint(2, 4), replace=False)
                w_idx = np.random.choice(p*n, size=np.random.randint(2, 4), replace=False)
                
                U[u_idx, r] = np.random.choice([-1, 1], size=len(u_idx))
                V[v_idx, r] = np.random.choice([-1, 1], size=len(v_idx))
                W[w_idx, r] = np.random.choice([-1, 1], size=len(w_idx))
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Strategy 2: Dense random with scaling
        elif restart % 3 == 1:
            params = np.random.randn((n*m + m*p + p*n) * R) * 0.5
        
        # Strategy 3: Block structure
        else:
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            
            for r in range(R):
                # Assign blocks
                i_block = r % n
                k_block = (r // n) % m
                j_block = (r // (n*m)) % p
                
                a_idx = i_block * m + k_block
                b_idx = k_block * p + j_block
                c_idx = j_block * n + i_block
                
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                
                # Add some extra terms
                if r % 2 == 0:
                    extra_a = (a_idx + 1) % (n*m)
                    extra_b = (b_idx + 1) % (m*p)
                    extra_c = (c_idx + 1) % (p*n)
                    U[extra_a, r] = 0.5
                    V[extra_b, r] = 0.5
                    W[extra_c, r] = 0.5
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(objective, params, method='L-BFGS-B', 
                           options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40})
            
            if res.fun < best_error:
                best_error = res.fun
                
                # If we found a very good solution, try to polish/round it
                if res.fun < 1e-8:
                    U = res.x[0:n*m*R].reshape(n*m, R)
                    V = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Try rounding to nearest 0.25
                    for denom in [2, 4, 8]:
                        U_rounded = np.round(U * denom) / denom
                        V_rounded = np.round(V * denom) / denom
                        W_rounded = np.round(W * denom) / denom
                        
                        # Verify exactness
                        vals = np.sum(U_rounded[indices_a, :] * V_rounded[indices_b, :] * W_rounded[indices_c, :], axis=1)
                        if np.allclose(vals, 1.0, atol=1e-9):
                            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        
                    # If no exact rounding works, try to find a local minimum with tighter tolerance
                    res2 = minimize(objective, res.x, method='L-BFGS-B', 
                                   options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12})
                    
                    if res2.fun < 1e-10:
                        U = res2.x[0:n*m*R].reshape(n*m, R)
                        V = res2.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                        W = res2.x[n*m*R + m*p*R:].reshape(p*n, R)
                        
                        # Final rounding attempt
                        U_rounded = np.round(U * 8) / 8
                        V_rounded = np.round(V * 8) / 8
                        W_rounded = np.round(W * 8) / 8
                        
                        vals = np.sum(U_rounded[indices_a, :] * V_rounded[indices_b, :] * W_rounded[indices_c, :], axis=1)
                        if np.allclose(vals, 1.0, atol=1e-9):
                            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        
        except Exception:
            continue
            
    return None

def valid_res(res, n, m, p):
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
        
    R = U.shape[1]
    if V.shape[1] != R or W.shape[1] != R:
        return False
        
    indices_a = []
    indices_b = []
    indices_c = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                indices_a.append(i * m + k)
                indices_b.append(k * p + j)
                indices_c.append(j * n + i)
                
    indices_a = np.array(indices_a)
    indices_b = np.array(indices_b)
    indices_c = np.array(indices_c)
    
    vals = np.sum(U[indices_a, :] * V[indices_b, :] * W[indices_c, :], axis=1)
    return np.allclose(vals, 1.0, atol=1e-9)

def construct_r34():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    # Verify this decomposition is valid
    
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    U[2][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    
    U[5][col] = 1
    V[9][col] = 1
    W[4][col] = 1
    col += 1
    
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    U[7][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    U[7][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    U[8][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    U[8][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1
    
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[10][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1
    
    U[2][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    U[2][col] = 1
    V[11][col] = 1
    W[9][col] = 1
    col += 1
    
    U[5][col] = 1
    V[10][col] = 1
    W[7][col] = 1
    col += 1
    
    U[5][col] = 1
    V[11][col] = 1
    W[10][col] = 1
    col += 1
    
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    
    U[7][col] = 1
    V[6][col] = 1
    W[8][col] = 1
    col += 1
    
    U[7][col] = 1
    V[7][col] = 1
    W[11][col] = 1
    col += 1
    
    U[8][col] = 1
    V[10][col] = 1
    W[8][col] = 1
    col += 1
    
    U[8][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

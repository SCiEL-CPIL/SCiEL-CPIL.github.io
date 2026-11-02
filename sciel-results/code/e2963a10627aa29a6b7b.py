"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _find_m223_rank10(timeout=15):
    """
    Attempts to find a rank-10 decomposition for M_{2,2,3} with half-integer entries.
    Uses a robust initialization strategy: start from a known rank-11 solution,
    zero out one dimension, and refine. Also tries random starts.
    """
    n, m, p = 2, 2, 3
    r = 10
    
    def unpack(params):
        u = params[0:4*r].reshape(n*m, r)
        v = params[4*r:4*r+6*r].reshape(m*p, r)
        w = params[4*r+6*r:4*r+12*r].reshape(p*n, r)
        return u, v, w
    
    def objective(params):
        u, v, w = unpack(params)
        residuals = np.zeros(12)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # val = sum_r u[i,j,r] * v[j,k,r] * w[k,i,r]
                    residuals[idx] = np.sum(u[i*m+j] * v[j*p+k] * w[k*n+i]) - 1.0
                    idx += 1
        return np.sum(residuals**2)

    def half_integer_penalty(params):
        u, v, w = unpack(params)
        all_vals = np.concatenate([u.flatten(), v.flatten(), w.flatten()])
        dists = np.abs(all_vals * 2 - np.round(all_vals * 2)) / 2
        return np.sum(dists**2)

    def combined_objective(params):
        return objective(params) + 0.01 * half_integer_penalty(params)

    best_sol = None
    start_time = time.time()
    
    # Construct rank 11 solution for M_{2,2,3} to use as initialization
    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    
    # Strassen for k=0,1
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    # Schoolbook for k=2
    U_school = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    V_school = np.array([
        [1, 0, 1, 0], # j=0, k=2
        [0, 1, 0, 1]  # j=1, k=2
    ])
    W_school = np.array([
        [1, 1, 0, 0], # i=0, k=2
        [0, 0, 1, 1]  # i=1, k=2
    ])
    
    for i in range(4):
        U11[i, :7] = U_strassen[i]
        U11[i, 7:] = U_school[i]
        
    for j in range(2):
        for local_k in range(2):
            V11[j*2+local_k, :7] = V_strassen[j*2+local_k]
        V11[j*3+2, 7:] = V_school[j]
        
    for local_k in range(2):
        for i in range(2):
            W11[local_k*2+i, :7] = W_strassen[local_k*2+i]
        for i in range(2):
            W11[2*2+i, 7:] = W_school[i]
            
    # Try a few strategies
    strategies = []
    
    # Strategy 1: Random starts
    for seed in range(8):
        strategies.append(('random', seed))
        
    # Strategy 2: Project rank 11 to rank 10 by zeroing out a column
    # This is a heuristic; we don't know which column to zero, so we try a few
    for zero_col in range(11):
        strategies.append(('project', zero_col))
        
    # Strategy 3: Start from rank 11, set one column to small random values
    for seed in range(4):
        strategies.append(('perturb', seed))

    for strat_type, strat_param in strategies:
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(1000 + strat_param)
        
        if strat_type == 'random':
            x0 = np.random.randn(12*r) * 0.5
        elif strat_type == 'project':
            # Take rank 11 solution, zero out column strat_param, then drop one more column?
            # No, we need rank 10.
            # Let's just take the first 10 columns of the rank 11 solution?
            # This is equivalent to assuming the 11th column is redundant.
            U_init = U11[:, :10]
            V_init = V11[:, :10]
            W_init = W11[:, :10]
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            # Add small noise to avoid saddle points
            x0 += np.random.randn(12*r) * 0.01
        elif strat_type == 'perturb':
            U_init = U11[:, :10]
            V_init = V11[:, :10]
            W_init = W11[:, :10]
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            # Perturb one specific column
            col_to_perturb = strat_param % 10
            x0[4*r + col_to_perturb] += np.random.randn() * 0.1
            x0[4*r + 6*r + col_to_perturb] += np.random.randn() * 0.1
            x0[4*r + 12*r + col_to_perturb] += np.random.randn() * 0.1
            
        try:
            # Use L-BFGS-B for better convergence
            res = minimize(combined_objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            
            if res.fun < 1e-5:
                u, v, w = unpack(res.x)
                u_rounded = np.round(u * 2) / 2
                v_rounded = np.round(v * 2) / 2
                w_rounded = np.round(w * 2) / 2
                
                # Verify
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(u_rounded[i*m+j] * v_rounded[j*p+k] * w_rounded[k*n+i])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-9:
                    return u_rounded, v_rounded, w_rounded
                    
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Rank 53 solution
    rank_fallback = 53
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    U_school = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_school = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_school = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U_fb[i * m + j][c] = U_strassen[i * 2 + j][r]
            for j in range(2):
                V_fb[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                V_fb[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W_fb[kk * n + i][c] = W_strassen[k * 2 + i][r]
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U_fb[i * m + j][c] = U_school[i * 2 + j][r]
        for j in range(2):
            V_fb[j * p + k_last][c] = V_school[j][r]
        for i in range(2):
            W_fb[k_last * n + i][c] = W_school[i][r]
            
    # Try to find rank 52 solution
    if time.time() < deadline - 20:
        sol = _find_m223_rank10(timeout=15)
        if sol is not None:
            u_m223, v_m223, w_m223 = sol
            rank_new = 52
            U_new = [[0] * rank_new for _ in range(n * m)]
            V_new = [[0] * rank_new for _ in range(m * p)]
            W_new = [[0] * rank_new for _ in range(p * n)]
            
            col_offset = 0
            for block in range(6):
                k0 = 2 * block
                k1 = 2 * block + 1
                for r in range(7):
                    c = col_offset + r
                    for i in range(2):
                        for j in range(2):
                            U_new[i * m + j][c] = U_strassen[i * 2 + j][r]
                    for j in range(2):
                        V_new[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                        V_new[j * p + k1][c] = V_strassen[j * 2 + 1][r]
                    for k in range(2):
                        kk = k0 + k
                        for i in range(2):
                            W_new[kk * n + i][c] = W_strassen[k * 2 + i][r]
                col_offset += 7
            
            for r in range(10):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U_new[i * m + j][c] = float(u_m223[i * 2 + j, r])
                for j in range(2):
                    for local_k in range(3):
                        global_k = 12 + local_k
                        V_new[j * p + global_k][c] = float(v_m223[j * 3 + local_k, r])
                for local_k in range(3):
                    global_k = 12 + local_k
                    for i in range(2):
                        W_new[global_k * n + i][c] = float(w_m223[local_k * 2 + i, r])
            
            return {"U": U_new, "V": V_new, "W": W_new}
            
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END

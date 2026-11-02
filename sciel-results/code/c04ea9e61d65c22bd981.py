import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find lower rank
    # We'll try ranks 83 down to 70
    for r in range(83, 69, -1):
        if time.time() > deadline:
            break
            
        trials = 100 if r > 75 else 50
        
        for t in range(trials):
            if time.time() > deadline:
                break
            
            # Random U, V with small values
            vals = [-1, -0.5, 0, 0.5, 1]
            U = np.random.choice(vals, size=(n * m, r))
            V = np.random.choice(vals, size=(m * p, r))
            
            num_vars = p * n * r
            num_eqs = n * m * p
            
            A_sys = np.zeros((num_eqs, num_vars))
            b_vec = np.zeros(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        b_vec[eq_idx] = 1.0
                        for c in range(r):
                            var_idx = (k * n + i) * r + c
                            A_sys[eq_idx, var_idx] = U[i * m + j, c] * V[j * p + k, c]
                        eq_idx += 1
            
            W_sol, residuals, rank, s = np.linalg.lstsq(A_sys, b_vec, rcond=None)
            
            if residuals and residuals[0] < 1e-9:
                W_check = np.round(W_sol * 2) / 2
                residual_check = np.linalg.norm(A_sys @ W_check - b_vec)
                if residual_check < 1e-9:
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W_check.reshape(p * n, r).tolist()
                    
                    best_rank = r
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break

    return best_result

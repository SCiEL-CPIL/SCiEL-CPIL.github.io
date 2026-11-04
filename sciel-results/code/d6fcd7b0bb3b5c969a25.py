import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Build rank 42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        c = start_col
        # M1
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True

    if not verify_decomp(U, V, W, rank):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    def try_rank(target_rank):
        if time.time() > deadline - 20:
            return None
        
        # Initialize with a subset of the 42-rank solution
        # Randomly select target_rank columns from the 42
        # Or just take the first target_rank
        U_init = np.array(U[:target_rank], dtype=float)
        V_init = np.array(V[:target_rank], dtype=float)
        W_init = np.array(W[:target_rank], dtype=float)
        
        def objective(params):
            U_flat, V_flat, W_flat = np.split(params, [target_rank * n * m, target_rank * (n * m + m * p)])
            U_curr = U_flat.reshape(n * m, target_rank)
            V_curr = V_flat.reshape(m * p, target_rank)
            W_curr = W_flat.reshape(p * n, target_rank)
            
            residuals = np.zeros(n * m * p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U_curr[i*m + j]
                        v_val = V_curr[j*p + k]
                        w_val = W_curr[k*n + i]
                        s = np.dot(u_val, v_val * w_val)
                        residuals[idx] = s - 1.0
                        idx += 1
            return residuals

        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        best_result = None
        # Try a few random perturbations to escape local minima
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            x0_perturbed = x0 + np.random.normal(0, 0.5, x0.shape)
            try:
                result = least_squares(objective, x0_perturbed, method='lm', max_nfev=1000)
                if best_result is None or np.linalg.norm(result.fun) < np.linalg.norm(best_result.fun):
                    best_result = result
            except Exception:
                pass
                
        if best_result is not None and np.linalg.norm(best_result.fun) < 1e-6:
            U_opt, V_opt, W_opt = np.split(best_result.x, [target_rank * n * m, target_rank * (n * m + m * p)])
            U_opt = U_opt.reshape(n * m, target_rank)
            V_opt = V_opt.reshape(m * p, target_rank)
            W_opt = W_opt.reshape(p * n, target_rank)
            
            def to_fraction_list(arr):
                result = []
                for row in arr:
                    result.append([])
                    for val in row:
                        frac = round(val * 2) / 2
                        if abs(frac - round(frac)) < 1e-8:
                            frac = int(round(frac))
                        result[-1].append(frac)
                return result
            
            U_final = to_fraction_list(U_opt)
            V_final = to_fraction_list(V_opt)
            W_final = to_fraction_list(W_opt)
            
            if verify_decomp(U_final, V_final, W_final, target_rank):
                return U_final, V_final, W_final
        return None

    # Try rank 40
    res = try_rank(40)
    if res:
        U, V, W = res
        return {"U": U, "V": V, "W": W}
        
    # Try rank 38
    res = try_rank(38)
    if res:
        U, V, W = res
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

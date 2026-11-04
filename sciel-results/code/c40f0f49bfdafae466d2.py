import time
import numpy as np
from itertools import combinations
from scipy.linalg import lstsq

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Build rank-42 Strassen decomposition
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
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback
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
    
    best_rank = rank
    best_U, best_V, best_W = U_np, V_np, W_np
    
    # Try to find rank 40, 39, 38 by removing columns and solving for W
    # Fix U and V, solve for W
    
    def try_prune(num_remove):
        if time.time() > deadline - 10:
            return None
            
        for remove_set in combinations(range(rank), num_remove):
            if time.time() > deadline - 5:
                return None
                
            active = [r for r in range(rank) if r not in remove_set]
            R = len(active)
            
            # Construct linear system for W
            # Eq (i,j,k): sum_{r in active} U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # Unknowns: W[k,i,r] for k in 0..5, i in 0..1, r in active
            # Total unknowns: 12 * R
            
            A_mat = np.zeros((48, 12 * R))
            b_vec = np.ones(48)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for col_idx, r in enumerate(active):
                            u_val = U_np[i*m+j, r]
                            v_val = V_np[j*p+k, r]
                            if u_val != 0 and v_val != 0:
                                # W index: row k*n+i, col r
                                # In our flattened unknown vector, index is (k*n+i) * R + col_idx
                                unk_idx = (k * n + i) * R + col_idx
                                A_mat[eq_idx, unk_idx] = u_val * v_val
                        idx += 1
            
            # Solve A_mat @ W_flat = b_vec
            # Use least squares
            try:
                W_flat, residuals, rank_sol, s = lstsq(A_mat, b_vec)
            except:
                continue
                
            # Check if solution is valid and "simple"
            # Residuals should be close to 0
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
                
            # Reshape W_flat to W_test
            W_test = np.zeros((p * n, R))
            for k in range(p):
                for i in range(n):
                    row_idx = k * n + i
                    W_test[row_idx, :] = W_flat[row_idx * R : (row_idx + 1) * R]
            
            # Check if W_test entries are close to integers or half-integers
            # Allow tolerance 1e-4
            simple = True
            for val in W_test:
                if val != 0:
                    # Check if close to k/2
                    half_int = np.round(val * 2) / 2
                    if abs(val - half_int) > 1e-4:
                        simple = False
                        break
            
            if simple:
                # Construct full U, V, W with zeros for removed columns
                U_test = np.zeros((n * m, R))
                V_test = np.zeros((m * p, R))
                for col_idx, r in enumerate(active):
                    U_test[:, col_idx] = U_np[:, r]
                    V_test[:, col_idx] = V_np[:, r]
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0
                            for r in range(R):
                                total += U_test[i*m+j, r] * V_test[j*p+k, r] * W_test[k*n+i, r]
                            if abs(total - 1) > 1e-6:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    return R, U_test, V_test, W_test
                    
        return None

    # Try rank 40
    res = try_prune(2)
    if res:
        R, U_t, V_t, W_t = res
        best_rank = R
        best_U, best_V, best_W = U_t, V_t, W_t
        # Try rank 39?
        if time.time() < deadline - 20:
             res = try_prune(3) # Relative to original? No, relative to current best?
             # try_prune uses original rank. We need to adapt.
             # Let's just stick to 40 for now if found, or try 39 from original.
             if res:
                 R, U_t, V_t, W_t = res
                 best_rank = R
                 best_U, best_V, best_W = U_t, V_t, W_t
                 
    if best_rank < rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}

import time
import numpy as np
import sys

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    def check_valid(U, V, W, tol=1e-9):
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        i_idx = np.repeat(np.arange(n), m*p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n*m)
        
        u_vals = U_arr[i_idx*m + j_idx]
        v_vals = V_arr[j_idx*p + k_idx]
        w_vals = W_arr[k_idx*n + i_idx]
        
        tensor_vals = np.sum(u_vals * v_vals * w_vals, axis=1)
        
        if np.any(np.abs(tensor_vals - 1.0) > tol):
            return False
        return True

    def build_strassen_2x2_2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def kronecker_decompose_4x4x4():
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if val != 0:
                                    U4[i*4+j][r] += val
                                  
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V4[j*4+k][r] += val
                                  
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W4[k*4+i][r] += val
        
        return U4, V4, W4

    def build_2x2x3_rank_11():
        U = [[0]*11 for _ in range(4)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(6)]
        
        U2, V2, W2 = build_strassen_2x2_2x2()
        for r in range(7):
            for idx in range(4):
                U[idx][r] = U2[idx][r]
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                V[j*3 + k][r] = V2[idx][r]
            for idx in range(4):
                W[idx][r] = W2[idx][r]
        
        for i in range(2):
            for j in range(2):
                r = 7 + i*2 + j
                U[i*2 + j][r] = 1
                V[j*3 + 2][r] = 1
                W[2*2 + i][r] = 1
        
        return U, V, W

    def build_4x4x7_decomposition(U4, V4, W4, r4):
        U223, V223, W223 = build_2x2x3_rank_11()
        r223 = 11
        
        r_total = r4 + 4 * r223
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        col_offset = r4
        for row_group in range(2):
            for col_group in range(2):
                base_r = col_offset + (row_group * 2 + col_group) * r223
                
                for r_local in range(r223):
                    r_global = base_r + r_local
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            i = 2 * row_group + i_local
                            j = 2 * col_group + j_local
                            U[i*4 + j][r_global] = U223[i_local*2 + j_local][r_local]
                    
                    for j_local in range(2):
                        for k_local in range(3):
                            j = 2 * col_group + j_local
                            k = 4 + k_local
                            V[j*7 + k][r_global] = V223[j_local*3 + k_local][r_local]
                    
                    for k_local in range(3):
                        for i_local in range(2):
                            k = 4 + k_local
                            i = 2 * row_group + i_local
                            W[k*4 + i][r_global] = W223[k_local*2 + i_local][r_local]
        
        return U, V, W

    U4_49, V4_49, W4_49 = kronecker_decompose_4x4x4()
    
    # Try to find rank 48 for 4x4x4
    U4_48, V4_48, W4_48 = None, None, None
    
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            # Remove one column from 49
            U_init = np.array(U4_49)[:, :48]
            V_init = np.array(V4_49)[:, :48]
            W_init = np.array(W4_49)[:, :48]
            
            # Vectorized Gradient Descent
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            lr = 0.005
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            no_improve_count = 0
            
            # Precompute indices for vectorization
            i_idx = np.repeat(np.arange(4), 4*4)
            j_idx = np.tile(np.repeat(np.arange(4), 4), 4)
            k_idx = np.tile(np.arange(4), 4*4)
            
            U_rows = U[i_idx*4 + j_idx] # Shape (64, 48)
            V_rows = V[j_idx*4 + k_idx] # Shape (64, 48)
            W_rows = W[k_idx*4 + i_idx] # Shape (64, 48)
            
            for step in range(20000):
                if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN - 2:
                    break
                
                # Compute sums: s[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Using broadcasting on the pre-extracted rows
                s = np.sum(U_rows * V_rows * W_rows, axis=1)
                
                # Error: sum (s - 1)^2
                err_vec = s - 1.0
                err = np.sum(err_vec**2)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    no_improve_count = 0
                    lr *= 1.02
                else:
                    no_improve_count += 1
                    lr *= 0.95
                    if no_improve_count > 20:
                        break
                
                # Gradients
                # d_err / d_U[i,j,r] = 2 * (s[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
                # We need to accumulate these into grad_U[i*4+j, r]
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # Vectorized gradient accumulation
                # grad_U_rows = 2 * err_vec[:, None] * V_rows * W_rows
                grad_U_rows = 2.0 * err_vec[:, None] * V_rows * W_rows
                grad_V_rows = 2.0 * err_vec[:, None] * U_rows * W_rows
                grad_W_rows = 2.0 * err_vec[:, None] * U_rows * V_rows
                
                # Scatter back to U, V, W
                # np.add.at is slow, but we can use bincount or just loop if small
                # Since 64 is small, we can use a simple loop for scattering or just use np.add.at
                # Actually, U_rows are just views/copies? No, they are copies from advanced indexing.
                # We need to add grad_U_rows back to U[i*4+j, :]
                
                # Efficient scatter:
                # For each row t in 0..63:
                # i = i_idx[t], j = j_idx[t]
                # U[i*4+j, :] += grad_U_rows[t, :]
                
                # Using np.add.at is the safest way for repeated indices, but here indices are unique for (i,j) pairs?
                # i_idx, j_idx generate all 16 (i,j) pairs 4 times (for each k).
                # So U[i*4+j, :] is updated 4 times.
                
                # Let's just use a loop for scattering, it's fast enough for 64 items
                for t in range(64):
                    u_idx = i_idx[t]*4 + j_idx[t]
                    v_idx = j_idx[t]*4 + k_idx[t]
                    w_idx = k_idx[t]*4 + i_idx[t]
                    
                    grad_U[u_idx, :] += grad_U_rows[t, :]
                    grad_V[v_idx, :] += grad_V_rows[t, :]
                    grad_W[w_idx, :] += grad_W_rows[t, :]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Update rows for next iteration
                U_rows = U[i_idx*4 + j_idx]
                V_rows = V[j_idx*4 + k_idx]
                W_rows = W[k_idx*4 + i_idx]
                
                if best_err < 1e-10:
                    break
            
            if best_err < 1e-9:
                # Polish: try to snap to integers or half-integers if close
                # And verify
                U4_48 = [[best_U[i][r] for r in range(48)] for i in range(16)]
                V4_48 = [[best_V[i][r] for r in range(48)] for i in range(16)]
                W4_48 = [[best_W[i][r] for r in range(48)] for i in range(16)]
                
                # Verify with tight tolerance
                if check_valid(U4_48, V4_48, W4_48, tol=1e-8):
                    print(f"Found rank 48 for 4x4x4 with err {best_err}")
                else:
                    U4_48, V4_48, W4_48 = None, None, None
        except Exception as e:
            print(e)
            U4_48, V4_48, W4_48 = None, None, None

    if U4_48 is not None:
        U_final, V_final, W_final = build_4x4x7_decomposition(U4_48, V4_48, W4_48, 48)
        rank = 48 + 44
    else:
        U_final, V_final, W_final = build_4x4x7_decomposition(U4_49, V4_49, W4_49, 49)
        rank = 49 + 44

    return {"U": U_final, "V": V_final, "W": W_final}

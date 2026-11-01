import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # Start with the known rank 39 solution
    # 4 Strassen blocks (8 cols) + 1 block of 3 cols (rank 11)
    # We will try to find a rank 10 decomposition for the last 3 cols.
    
    # Define the target tensor for 2x2 by 2x3
    # A: 2x2, B: 2x3, C: 2x3
    # Indices:
    # A: 0,1,2,3 (row-major: 00, 01, 10, 11)
    # B: 0,1,2,3,4,5 (row-major: 00, 01, 02, 10, 11, 12)
    # C: 0,1,2,3,4,5 (row-major: 00, 01, 02, 10, 11, 12)
    
    # Target coefficients:
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # We want U (4x10), V (6x10), W (6x10)
    # such that sum_r W[c,r] * (sum_i U[i,r] A[i]) * (sum_j V[j,r] B[j]) = Target[C]
    
    def error_func(params):
        U = params[0:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        
        err = 0.0
        # For each output C[c]
        for c in range(6):
            # Compute the bilinear form for this output
            # L_r = U[:,r] @ A_vec
            # M_r = V[:,r] @ B_vec
            # C_c = sum_r W[c,r] L_r M_r
            # We need to match coefficients of A_i B_j
            # Coeff of A_i B_j in C_c is sum_r W[c,r] U[i,r] V[j,r]
            for i in range(4):
                for j in range(6):
                    coeff = 0.0
                    for r in range(10):
                        coeff += W[c, r] * U[i, r] * V[j, r]
                    
                    # Determine target coeff
                    # c is row-major index of C: c = 3*row + col
                    c_row = c // 3
                    c_col = c % 3
                    # i is row-major index of A: i = 2*row + col
                    a_row = i // 2
                    a_col = i % 2
                    # j is row-major index of B: j = 3*row + col
                    b_row = j // 3
                    b_col = j % 3
                    
                    # Target is 1 if a_row == c_row and b_row == a_col and b_col == c_col
                    # Wait, C[c_row, c_col] = sum_k A[c_row, k] B[k, c_col]
                    # So term is A[c_row, k] B[k, c_col]
                    # i corresponds to A[c_row, a_col]
                    # j corresponds to B[b_row, b_col]
                    # Match if a_col == k and b_row == k and b_col == c_col
                    # So a_col == b_row and b_col == c_col and a_row == c_row
                    if a_row == c_row and a_col == b_row and b_col == c_col:
                        target = 1.0
                    else:
                        target = 0.0
                    
                    err += (coeff - target) ** 2
        return err

    # Initialize with rank 11 solution and drop one rank
    # Rank 11 for 2x2 by 2x3:
    # Strassen for cols 0,1 (rank 7) + Schoolbook for col 2 (rank 4)
    # We will create a 160-dim vector for rank 10.
    # This is tricky.
    
    # Instead, we can try random initialization and minimize.
    # But 160 variables is a lot.
    # Let's try to fix some variables to reduce the search space.
    # Or just run it and hope.
    
    # Time check
    start_time = time.time()
    deadline = start_time + 100.0 # Leave 20s margin
    
    best_params = None
    best_err = float('inf')
    
    # Try a few random restarts
    for _ in range(5):
        if time.time() > deadline:
            break
            
        # Random init
        params = np.random.randn(160)
        # Scale down
        params /= 10
        
        res = minimize(error_func, params, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < best_err:
            best_err = res.fun
            best_params = res.x
            
        if best_err < 1e-4:
            break

    if best_params is not None and best_err < 1e-3:
        # Use the found rank 10 decomposition for the last 3 columns
        U_last = best_params[0:40].reshape(4, 10)
        V_last = best_params[40:100].reshape(6, 10)
        W_last = best_params[100:160].reshape(6, 10)
        
        # Construct full matrices
        # 4 Strassen blocks (cols 0-7) + 1 Rank 10 block (cols 8-10)
        rank = 4 * 7 + 10
        U = np.zeros((4, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        r = 0
        # 4 Strassen blocks
        for block in range(4):
            k0 = block * 2
            k1 = block * 2 + 1
            
            v_e = k0
            v_f = k1
            v_g = 2 + k0 # B[1, k0] is index 3+k0? No.
            # B indices: 0,1,2 (row 0), 3,4,5 (row 1)
            # So B[0, k] is k. B[1, k] is 3+k.
            v_g = 3 + k0
            v_h = 3 + k1
            
            # C indices: 0,1,2 (row 0), 3,4,5 (row 1)
            w_c00 = k0
            w_c10 = 3 + k0
            w_c01 = k1
            w_c11 = 3 + k1
            
            # m1
            U[0, r] = 1; U[3, r] = 1
            V[v_e, r] = 1; V[v_h, r] = 1
            W[w_c00, r] = 1; W[w_c11, r] = 1
            r += 1
            # m2
            U[2, r] = 1; U[3, r] = 1
            V[v_e, r] = 1
            W[w_c10, r] = 1; W[w_c11, r] = -1
            r += 1
            # m3
            U[0, r] = 1
            V[v_f, r] = 1; V[v_h, r] = -1
            W[w_c01, r] = 1; W[w_c11, r] = 1
            r += 1
            # m4
            U[3, r] = 1
            V[v_g, r] = 1; V[v_e, r] = -1
            W[w_c00, r] = 1; W[w_c10, r] = 1
            r += 1
            # m5
            U[0, r] = 1; U[1, r] = 1
            V[v_h, r] = 1
            W[w_c00, r] = -1; W[w_c01, r] = 1
            r += 1
            # m6
            U[2, r] = 1; U[0, r] = -1
            V[v_e, r] = 1; V[v_f, r] = 1
            W[w_c11, r] = 1
            r += 1
            # m7
            U[1, r] = 1; U[3, r] = -1
            V[v_g, r] = 1; V[v_h, r] = 1
            W[w_c00, r] = 1
            r += 1

        # Add Rank 10 block for cols 8,9,10
        # Map local indices 0,1,2 to global 8,9,10
        # U is same (4x10)
        U[:, r:r+10] = U_last
        # V: local 0,1,2 -> global 8,9,10 (row 0)
        #    local 3,4,5 -> global 11,12,13 (row 1)
        V[0:3, r:r+10] = V_last[0:3, :]
        V[3:6, r:r+10] = V_last[3:6, :]
        # W: local 0,1,2 -> global 8,9,10 (row 0)
        #    local 3,4,5 -> global 11,12,13 (row 1)
        W[0:3, r:r+10] = W_last[0:3, :]
        W[3:6, r:r+10] = W_last[3:6, :]
        r += 10
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to rank 39
        rank = 39
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(6)]
        W = [[0] * rank for _ in range(6)]
        
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            
            v_e = k0
            v_f = k1
            v_g = 2 + k0
            v_h = 2 + k1
            
            w_c00 = k0
            w_c10 = 3 + k0
            w_c01 = k1
            w_c11 = 3 + k1
            
            U[0, r] = 1; U[3, r] = 1
            V[v_e, r] = 1; V[v_h, r] = 1
            W[w_c00, r] = 1; W[w_c11, r] = 1
            r += 1
            U[2, r] = 1; U[3, r] = 1
            V[v_e, r] = 1
            W[w_c10, r] = 1; W[w_c11, r] = -1
            r += 1
            U[0, r] = 1
            V[v_f, r] = 1; V[v_h, r] = -1
            W[w_c01, r] = 1; W[w_c11, r] = 1
            r += 1
            U[3, r] = 1
            V[v_g, r] = 1; V[v_e, r] = -1
            W[w_c00, r] = 1; W[w_c10, r] = 1
            r += 1
            U[0, r] = 1; U[1, r] = 1
            V[v_h, r] = 1
            W[w_c00, r] = -1; W[w_c01, r] = 1
            r += 1
            U[2, r] = 1; U[0, r] = -1
            V[v_e, r] = 1; V[v_f, r] = 1
            W[w_c11, r] = 1
            r += 1
            U[1, r] = 1; U[3, r] = -1
            V[v_g, r] = 1; V[v_h, r] = 1
            W[w_c00, r] = 1
            r += 1

        k2 = 10
        v_e2 = k2
        v_g2 = 2 + k2
        w_c02 = k2
        w_c12 = 3 + k2
        
        U[0, r] = 1
        V[v_e2, r] = 1
        W[w_c02, r] = 1
        r += 1
        U[1, r] = 1
        V[v_g2, r] = 1
        W[w_c02, r] = 1
        r += 1
        U[2, r] = 1
        V[v_e2, r] = 1
        W[w_c12, r] = 1
        r += 1
        U[3, r] = 1
        V[v_g2, r] = 1
        W[w_c12, r] = 1
        r += 1

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

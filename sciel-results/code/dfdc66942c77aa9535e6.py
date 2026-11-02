# EVOLVE-BLOCK-START
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Schoolbook
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    best_rank = rank_schoolbook
    
    # Construct a rank 44 decomposition using 22-rank 3x3x3 blocks
    # 22 = 7 (Strassen 2x2) + 15 (Schoolbook for remaining 5 outputs)
    
    rank_3x3 = 22
    rank_total = 2 * rank_3x3 # 44
    
    U_new = np.zeros((n * m, rank_total))
    V_new = np.zeros((m * p, rank_total))
    W_new = np.zeros((p * n, rank_total))
    
    # Helper to add a term to the factors
    def add_term(U_mat, V_mat, W_mat, col, u_idx, v_idx, w_idx, u_val, v_val, w_val):
        U_mat[u_idx, col] = u_val
        V_mat[v_idx, col] = v_val
        W_mat[w_idx, col] = w_val

    # Define the 22 terms for 3x3x3
    # Indices: i (0-2), j (0-2), k (0-2)
    # U index: i*m + j
    # V index: j*p + k (here p=3 for the sub-block)
    # W index: k*n + i (here n=3 for the sub-block)
    
    terms = []
    
    # 7 Strassen terms for top-left 2x2
    # M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1. V: B00=1, B11=1.
    # W: C00=1, C11=1.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to express W contributions.
    # Let's just list the (U_indices, V_indices, W_indices) and values.
    
    # Term 0: M1
    # U: (0,0), (1,1) -> 1
    # V: (0,0), (1,1) -> 1
    # W: (0,0)=1, (1,1)=1
    # Wait, W is for the output.
    # C00 gets +1 from M1. C11 gets +1 from M1.
    # So W indices: 0*3+0=0 (val 1), 1*3+1=4 (val 1).
    
    # This manual construction is error-prone.
    # Let's generate the 22-rank factors programmatically based on the formulas.
    
    U_sub = np.zeros((9, 22))
    V_sub = np.zeros((9, 22)) # j*p+k, p=3
    W_sub = np.zeros((9, 22)) # k*n+i, n=3
    
    # 1. Strassen Terms (cols 0-6)
    # M1
    U_sub[0, 0] = 1; U_sub[4, 0] = 1
    V_sub[0, 0] = 1; V_sub[4, 0] = 1
    W_sub[0, 0] = 1; W_sub[4, 0] = 1
    
    # M2 = (A10+A11)B00
    U_sub[3, 1] = 1; U_sub[4, 1] = 1
    V_sub[0, 1] = 1
    W_sub[3, 1] = 1 # C10 gets +1
    
    # M3 = A00(B10-B11)
    U_sub[0, 2] = 1
    V_sub[3, 2] = 1; V_sub[4, 2] = -1
    W_sub[1, 2] = 1; W_sub[4, 2] = 1 # C01 gets +1, C11 gets +1
    
    # M4 = A11(B01-B00)
    U_sub[4, 3] = 1
    V_sub[1, 3] = 1; V_sub[0, 3] = -1
    W_sub[0, 3] = 1; W_sub[3, 3] = 1 # C00 gets +1, C10 gets +1
    
    # M5 = (A00+A01)B11
    U_sub[0, 4] = 1; U_sub[1, 4] = 1
    V_sub[4, 4] = 1
    W_sub[0, 4] = -1; W_sub[1, 4] = 1 # C00 gets -1, C01 gets +1
    
    # M6 = (A10-A00)(B00+B01)
    U_sub[3, 5] = 1; U_sub[0, 5] = -1
    V_sub[0, 5] = 1; V_sub[1, 5] = 1
    W_sub[4, 5] = 1 # C11 gets +1
    
    # M7 = (A01-A11)(B10+B11)
    U_sub[1, 6] = 1; U_sub[4, 6] = -1
    V_sub[3, 6] = 1; V_sub[4, 6] = 1
    W_sub[0, 6] = 1 # C00 gets +1
    
    # 2. Schoolbook Terms for remaining outputs (cols 7-21)
    # C02 = A00B02 + A01B12 + A02B22
    # C12 = A10B02 + A11B12 + A12B22
    # C20 = A20B00 + A21B10 + A22B20
    # C21 = A20B01 + A21B11 + A22B21
    # C22 = A20B02 + A21B12 + A22B22
    
    col = 7
    # C02 terms
    # A00 B02
    U_sub[0, col] = 1; V_sub[2, col] = 1; W_sub[2, col] = 1; col+=1
    # A01 B12
    U_sub[1, col] = 1; V_sub[5, col] = 1; W_sub[2, col] = 1; col+=1
    # A02 B22
    U_sub[2, col] = 1; V_sub[8, col] = 1; W_sub[2, col] = 1; col+=1
    
    # C12 terms
    # A10 B02
    U_sub[3, col] = 1; V_sub[2, col] = 1; W_sub[5, col] = 1; col+=1
    # A11 B12
    U_sub[4, col] = 1; V_sub[5, col] = 1; W_sub[5, col] = 1; col+=1
    # A12 B22
    U_sub[5, col] = 1; V_sub[8, col] = 1; W_sub[5, col] = 1; col+=1
    
    # C20 terms
    # A20 B00
    U_sub[6, col] = 1; V_sub[0, col] = 1; W_sub[6, col] = 1; col+=1
    # A21 B10
    U_sub[7, col] = 1; V_sub[3, col] = 1; W_sub[6, col] = 1; col+=1
    # A22 B20
    U_sub[8, col] = 1; V_sub[6, col] = 1; W_sub[6, col] = 1; col+=1
    
    # C21 terms
    # A20 B01
    U_sub[6, col] = 1; V_sub[1, col] = 1; W_sub[7, col] = 1; col+=1
    # A21 B11
    U_sub[7, col] = 1; V_sub[4, col] = 1; W_sub[7, col] = 1; col+=1
    # A22 B21
    U_sub[8, col] = 1; V_sub[7, col] = 1; W_sub[7, col] = 1; col+=1
    
    # C22 terms
    # A20 B02
    U_sub[6, col] = 1; V_sub[2, col] = 1; W_sub[8, col] = 1; col+=1
    # A21 B12
    U_sub[7, col] = 1; V_sub[5, col] = 1; W_sub[8, col] = 1; col+=1
    # A22 B22
    U_sub[8, col] = 1; V_sub[8, col] = 1; W_sub[8, col] = 1; col+=1
    
    # Verify sub-decomposition
    # T_ijk = 1
    # sum_r U_sub[i*3+j, r] V_sub[j*3+k, r] W_sub[k*3+i, r]
    
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = 0
                for r in range(22):
                    val += U_sub[i*3+j, r] * V_sub[j*3+k, r] * W_sub[k*3+i, r]
                if abs(val - 1) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        # Construct full 3x3x6 decomposition
        # First block: k=0,1,2 uses U_sub, V_sub, W_sub
        # Second block: k=3,4,5 uses U_sub, V_sub (shifted), W_sub (shifted)
        
        U_new[:, 0:22] = U_sub
        U_new[:, 22:44] = U_sub
        
        for r in range(22):
            for j in range(3):
                for k in range(3):
                    V_new[j*6 + k, r] = V_sub[j*3 + k, r]
                    V_new[j*6 + (k+3), r+22] = V_sub[j*3 + k, r]
                    
            for k in range(3):
                for i in range(3):
                    W_new[k*3 + i, r] = W_sub[k*3 + i, r]
                    W_new[(k+3)*3 + i, r+22] = W_sub[k*3 + i, r]
                    
        # Final verification
        final_valid = True
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    val = 0
                    for r in range(44):
                        val += U_new[i*3+j, r] * V_new[j*6+k, r] * W_new[k*3+i, r]
                    if abs(val - 1) > 1e-9:
                        final_valid = False
                        break
                if not final_valid: break
            if not final_valid: break
            
        if final_valid:
            best_result = {'U': U_new.tolist(), 'V': V_new.tolist(), 'W': W_new.tolist()}
            best_rank = 44
            
    return best_result
# EVOLVE-BLOCK-END

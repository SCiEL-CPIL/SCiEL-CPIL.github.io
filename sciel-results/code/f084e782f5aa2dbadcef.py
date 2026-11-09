"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 4

    # Build rank-28 Strassen decomposition using Kronecker structure
    # The 2x4x4 multiplication can be decomposed into 4 independent 2x2x2 problems
    # Each 2x2x2 uses Strassen (rank 7), total rank = 4*7 = 28
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # For 2x4 * 4x4:
    # A is 2x4: A[:, 0:2] and A[:, 2:4]
    # B is 4x4: B[0:2, :] and B[2:4, :]
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    # Each term is 2x2 * 2x4 or 2x4 * 2x4... 
    # Actually: A(2x4) * B(4x4) = [A1 | A2] * [[B11, B12], [B21, B22]]
    # where A1, A2 are 2x2, B11, B12, B21, B22 are 2x2
    # C = A1*B11 + A2*B21 (first two cols of B) + A1*B12 + A2*B22 (last two cols)
    # = A1*B11 + A2*B21 + A1*B12 + A2*B22
    # That's 4 separate 2x2*2x2 multiplications, each rank 7, total 28.

    rank = 28
    U = np.zeros((n * m, rank))  # (8, 28)
    V = np.zeros((m * p, rank))  # (16, 28)
    W = np.zeros((p * n, rank))  # (8, 28)

    # Strassen factors for 2x2 * 2x2
    # U factors: A combination
    # V factors: B combination  
    # W factors: C combination
    strassen_U = np.array([
        [1, 0, 0, 1, 1, 1, 1],   # A11 row
        [0, 1, 1, 0, 0, 1, 1],   # A21 row
        [1, 0, 0, 1, 1, -1, -1], # A22 row (for M1, M4, M5, M6, M7)
        [0, 1, 1, 0, 0, -1, -1], # A12 row
    ])
    # Let me redo this more carefully.
    # For Strassen 2x2*2x2:
    # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # U coefficients (for A elements):
    # M1: a11 + a22
    # M2: a21 + a22
    # M3: a11
    # M4: a22
    # M5: a11 + a12
    # M6: a21 - a11
    # M7: a12 - a22
    
    # V coefficients (for B elements):
    # M1: b11 + b22
    # M2: b11
    # M3: b21 - b22
    # M4: b12 - b11
    # M5: b22
    # M6: b11 + b12
    # M7: b21 + b22
    
    # W coefficients (for C elements):
    # C11: M1 + M4 - M5 - M7
    # C12: M3 + M5
    # C21: M2 + M4
    # C22: M1 - M2 + M3 + M7
    
    # Indexing: A[i,j] where i=0,1 (rows), j=0,1 (cols within 2x2 block)
    # B[j,k] where j=0,1 (rows within 2x2 block), k=0,1 (cols)
    # C[i,k] where i=0,1 (rows), k=0,1 (cols within 2x2 block)
    
    # U: for each of 7 Strassen terms, coefficient for each A element
    # A elements: a00, a01, a10, a11 (row, col in 2x2)
    U_strassen = np.array([
        [1, 0, 0, 1],  # M1: a00 + a11
        [0, 0, 1, 1],  # M2: a10 + a11
        [1, 0, 0, 0],  # M3: a00
        [0, 0, 0, 1],  # M4: a11
        [1, 1, 0, 0],  # M5: a00 + a01
        [-1, 0, 1, 0], # M6: a10 - a00
        [0, 1, 0, -1], # M7: a01 - a11
    ])
    
    # V: for each of 7 Strassen terms, coefficient for each B element
    # B elements: b00, b01, b10, b11 (row, col in 2x2)
    V_strassen = np.array([
        [1, 0, 0, 1],  # M1: b00 + b11
        [1, 0, 0, 0],  # M2: b00
        [0, 0, 1, -1], # M3: b10 - b11
        [0, 1, -1, 0], # M4: b01 - b00
        [0, 0, 0, 1],  # M5: b11
        [1, 1, 0, 0],  # M6: b00 + b01
        [0, 0, 1, 1],  # M7: b10 + b11
    ])
    
    # W: for each of 4 C elements, coefficient for each of 7 Strassen terms
    # C elements: c00, c01, c10, c11 (row, col in 2x2)
    # W[c, m] = coefficient of M_m in C_c
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, -1],  # C00 = M1 + M4 - M5 - M7
        [0, 0, 1, 0, 1, 0, 0],     # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],     # C10 = M2 + M4
        [1, -1, 1, 0, 0, 0, 1],    # C11 = M1 - M2 + M3 + M7
    ])
    
    # Now build the full 2x4*4x4 decomposition
    # A is 2x4: A[i, j] where i=0,1, j=0..3
    #   j=0,1 -> first 2x2 block (call it A_L)
    #   j=2,3 -> second 2x2 block (call it A_R)
    # B is 4x4: B[j, k] where j=0..3, k=0..3
    #   j=0,1 -> first row block (B_T)
    #   j=2,3 -> second row block (B_B)
    #   k=0,1 -> first col block (B_L)  
    #   k=2,3 -> second col block (B_R)
    # C is 2x4: C[i, k] where i=0,1, k=0..3
    #   k=0,1 -> first 2x2 block (C_L)
    #   k=2,3 -> second 2x2 block (C_R)
    
    # C_L = A_L * B_TL + A_R * B_BL  (where B_TL = B[0:2, 0:2], B_BL = B[2:4, 0:2])
    # C_R = A_L * B_TR + A_R * B_BR  (where B_TR = B[0:2, 2:4], B_BR = B[2:4, 2:4])
    
    # 4 subproblems:
    # Sub 0: A_L * B_TL -> C_L (i in 0,1; j in 0,1; k in 0,1)
    # Sub 1: A_R * B_BL -> C_L (i in 0,1; j in 2,3; k in 0,1)
    # Sub 2: A_L * B_TR -> C_R (i in 0,1; j in 0,1; k in 2,3)
    # Sub 3: A_R * B_BR -> C_R (i in 0,1; j in 2,3; k in 2,3)
    
    for sub in range(4):
        # Determine which A block (L or R) and which B block
        if sub == 0:
            a_block = 0  # A_L: j offset 0
            b_row_offset = 0  # B rows 0,1
            b_col_offset = 0  # B cols 0,1
        elif sub == 1:
            a_block = 1  # A_R: j offset 2
            b_row_offset = 2  # B rows 2,3
            b_col_offset = 0  # B cols 0,1
        elif sub == 2:
            a_block = 0  # A_L: j offset 0
            b_row_offset = 0  # B rows 0,1
            b_col_offset = 2  # B cols 2,3
        else:  # sub == 3
            a_block = 1  # A_R: j offset 2
            b_row_offset = 2  # B rows 2,3
            b_col_offset = 2  # B cols 2,3
        
        c_col_offset = 0 if (sub == 0 or sub == 1) else 2  # C_L or C_R
        
        for f in range(7):
            col = sub * 7 + f
            # U: A[i, a_block*2 + j_local]
            for i in range(2):
                for j_local in range(2):
                    coeff = U_strassen[f, j_local]
                    if coeff != 0:
                        j = a_block * 2 + j_local
                        U[i * m + j, col] += coeff
            
            # V: B[b_row_offset + j_local, b_col_offset + k_local]
            for j_local in range(2):
                for k_local in range(2):
                    coeff = V_strassen[f, k_local]  # V_strassen[f, col_in_2x2]
                    if coeff != 0:
                        j = b_row_offset + j_local
                        k = b_col_offset + k_local
                        V[j * p + k, col] += coeff
            
            # W: C[i, c_col_offset + k_local]
            for k_local in range(2):
                for i in range(2):
                    coeff = W_strassen[i, f]  # W_strassen[c_row, strassen_term]
                    if coeff != 0:
                        k = b_col_offset + k_local
                        W[k * n + i, col] += coeff

    # Verify the decomposition
    def is_valid(U_mat, V_mat, W_mat, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(r):
                        total += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    if not is_valid(U, V, W, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank

    # Try rank reduction: for each column, check if it can be expressed as a linear
    # combination of the others. If so, remove it.
    # The condition is: for all (i,j,k), sum_c' U[i,j,c']*V[j,k,c']*W[k,i,c'] = 1
    # If column c0 can be written as sum_{c != c0} alpha_c * column_c, then
    # we need: U[:,c0]*V[:,c0]*W[:,c0] = sum_{c != c0} alpha_c * U[:,c]*V[:,c]*W[:,c]
    # This is a system of n*m*p = 32 equations in (rank-1) unknowns.
    
    def try_remove_column(U_mat, V_mat, W_mat, r, col_to_remove):
        """Try to express column col_to_remove as linear combination of others."""
        if r <= 1:
            return None
        
        # Build system: for each (i,j,k), 
        # U[i,j,col0]*V[j,k,col0]*W[k,i,col0] = sum_{c != col0} alpha_c * U[i,j,c]*V[j,k,c]*W[k,i,c]
        # This is linear in alpha_c.
        
        A_sys = np.zeros((n * m * p, r - 1))
        b_sys = np.zeros(n * m * p)
        
        other_cols = [c for c in range(r) if c != col_to_remove]
        
        for idx, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
            b_sys[idx] = U_mat[i * m + j, col_to_remove] * V_mat[j * p + k, col_to_remove] * W_mat[k * n + i, col_to_remove]
            for oc_idx, oc in enumerate(other_cols):
                A_sys[idx, oc_idx] = U_mat[i * m + j, oc] * V_mat[j * p + k, oc] * W_mat[k * n + i, oc]
        
        # Solve A_sys * alpha = b_sys
        try:
            alpha, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=1e-10)
        except np.linalg.LinAlgError:
            return None
        
        # Check if solution is valid
        if np.allclose(A_sys @ alpha, b_sys, atol=1e-9):
            # Verify the full decomposition still holds
            U_new = np.delete(U_mat, col_to_remove, axis=1)
            V_new = np.delete(V_mat, col_to_remove, axis=1)
            W_new = np.delete(W_mat, col_to_remove, axis=1)
            
            if is_valid(U_new, V_new, W_new, r - 1):
                return (U_new, V_new, W_new, r - 1)
        
        return None

    # Try removing each column, one at a time, repeatedly
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for col in range(best_rank):
            if time.time() > deadline - 10:
                break
            result = try_remove_column(best_U, best_V, best_W, best_rank, col)
            if result is not None:
                best_U, best_V, best_W, best_rank = result
                improved = True
                break  # Restart from beginning after a removal

    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Target dimensions
    # U: (n*m, rank) -> (4, rank)
    # V: (m*p, rank) -> (6, rank)
    # W: (p*n, rank) -> (6, rank)
    
    # The tensor T has non-zero entries at (i*m+j, j*p+k, k*n+i)
    # We decompose the problem into two blocks:
    # Block 1: j=0, k in {0, 1} (2x2 multiplication part)
    # Block 2: j=1, k in {0, 1, 2} (remaining part)
    # Actually, it's easier to think of A as 2x2 and B as 2x3.
    # C = A B.
    # C has 2x3 entries.
    # We can split B into B1 (2x2) and B2 (2x1).
    # C1 = A B1 (2x2 result)
    # C2 = A B2 (2x1 result)
    
    # Let's use Strassen for C1 (rank 7) and Schoolbook for C2 (rank 4).
    # Total rank = 7 + 4 = 11.
    
    # Strassen factors for 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]]
    # B1 = [[b00, b01], [b10, b11]]
    # Indices for A: 0->a00, 1->a01, 2->a10, 3->a11
    # Indices for B1 (subset of B): 
    #   B indices are j*p+k.
    #   j=0, k=0 -> b00 (idx 0)
    #   j=0, k=1 -> b01 (idx 1)
    #   j=1, k=0 -> b10 (idx 2)
    #   j=1, k=1 -> b11 (idx 3)
    #   j=1, k=2 -> b12 (idx 4)  <-- This is in B2
    #   j=0, k=2 -> b02 (idx 5)  <-- This is in B2? No, B is 2x3.
    #   Let's check B indexing: V has shape (m*p, rank) = (2*3, rank) = (6, rank).
    #   Row index = j*p + k.
    #   j=0: k=0->0, k=1->1, k=2->2. So b00=0, b01=1, b02=2.
    #   j=1: k=0->3, k=1->4, k=2->5. So b10=3, b11=4, b12=5.
    
    # Wait, the standard Strassen applies to 2x2 * 2x2.
    # Here we have 2x2 * 2x3.
    # We can view this as two 2x2 * 2x1 multiplications? Or one 2x2*2x2 and one 2x2*2x1?
    # Let's split B into B_left (cols 0,1) and B_right (col 2).
    # B_left is 2x2. B_right is 2x1.
    # C_left = A * B_left (2x2). Use Strassen (rank 7).
    # C_right = A * B_right (2x1). Use Schoolbook (rank 4).
    
    # Strassen for 2x2 * 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C_left[0,0] = M1 - M4 + M5 - M7
    # C_left[0,1] = M3 + M7
    # C_left[1,0] = M2 + M4
    # C_left[1,1] = M1 - M2 + M3 + M6
    
    # Indices for A (U rows):
    # 0: a00
    # 1: a01
    # 2: a10
    # 3: a11
    
    # Indices for B_left (V rows 0..3):
    # 0: b00
    # 1: b01
    # 2: b02 (Wait, B is 2x3. Rows of V are j*p+k.
    #    j=0, k=0 -> 0 (b00)
    #    j=0, k=1 -> 1 (b01)
    #    j=0, k=2 -> 2 (b02)
    #    j=1, k=0 -> 3 (b10)
    #    j=1, k=1 -> 4 (b11)
    #    j=1, k=2 -> 5 (b12)
    #    
    #    My previous mapping of B_left was:
    #    B_left = [[b00, b01], [b10, b11]]
    #    So B_left rows in V:
    #    b00 is V[0]
    #    b01 is V[1]
    #    b10 is V[3]
    #    b11 is V[4]
    #    
    #    B_right = [[b02], [b12]]
    #    b02 is V[2]
    #    b12 is V[5]
    
    # Let's define the Strassen factors for the 2x2 part (using b00, b01, b10, b11)
    # U_strassen (4x7), V_strassen (6x7), W_strassen (6x7)
    # Note: V_strassen will have zeros in rows 2 and 5 (corresponding to b02, b12)
    # W_strassen will have zeros in rows corresponding to C_right outputs?
    # C indices (W rows): k*n+i.
    # C is 2x3.
    # C[0,0] (i=0, k=0) -> 0*2+0 = 0
    # C[0,1] (i=0, k=1) -> 1*2+0 = 2
    # C[0,2] (i=0, k=2) -> 2*2+0 = 4
    # C[1,0] (i=1, k=0) -> 0*2+1 = 1
    # C[1,1] (i=1, k=1) -> 1*2+1 = 3
    # C[1,2] (i=1, k=2) -> 2*2+1 = 5
    
    # C_left outputs:
    # C[0,0] -> W[0]
    # C[0,1] -> W[2]
    # C[1,0] -> W[1]
    # C[1,1] -> W[3]
    
    # C_right outputs:
    # C[0,2] -> W[4]
    # C[1,2] -> W[5]
    
    # So Strassen part writes to W[0], W[1], W[2], W[3].
    # Schoolbook part writes to W[4], W[5].
    
    U_total = np.zeros((4, 11))
    V_total = np.zeros((6, 11))
    W_total = np.zeros((6, 11))
    
    # --- Strassen Part (Columns 0-6) ---
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Col 0: M1
    U_total[0, 0] = 1  # a00
    U_total[3, 0] = 1  # a11
    V_total[0, 0] = 1  # b00
    V_total[4, 0] = 1  # b11
    W_total[0, 0] = 1  # C00
    W_total[3, 0] = 1  # C11
    # Wait, C00 = M1 - M4 + M5 - M7. C11 = M1 - M2 + M3 + M6.
    # So M1 contributes +1 to C00 and +1 to C11.
    
    # Col 1: M2 = (a10+a11)b00
    U_total[2, 1] = 1  # a10
    U_total[3, 1] = 1  # a11
    V_total[0, 1] = 1  # b00
    # C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
    # M2 contributes +1 to C10 and -1 to C11.
    W_total[1, 1] = 1  # C10
    W_total[3, 1] = -1 # C11
    
    # Col 2: M3 = a00(b01-b11)
    U_total[0, 2] = 1  # a00
    V_total[1, 2] = 1  # b01
    V_total[4, 2] = -1 # b11
    # C01 = M3 + M7. C11 = M1 - M2 + M3 + M6.
    # M3 contributes +1 to C01 and +1 to C11.
    W_total[2, 2] = 1  # C01
    W_total[3, 2] = 1  # C11
    
    # Col 3: M4 = a11(b10-b00)
    U_total[3, 3] = 1  # a11
    V_total[3, 3] = 1  # b10
    V_total[0, 3] = -1 # b00
    # C00 = M1 - M4 + M5 - M7. C10 = M2 + M4.
    # M4 contributes -1 to C00 and +1 to C10.
    W_total[0, 3] = -1 # C00
    W_total[1, 3] = 1  # C10
    
    # Col 4: M5 = (a00+a01)b11
    U_total[0, 4] = 1  # a00
    U_total[1, 4] = 1  # a01
    V_total[4, 4] = 1  # b11
    # C00 = M1 - M4 + M5 - M7.
    # M5 contributes +1 to C00.
    W_total[0, 4] = 1  # C00
    
    # Col 5: M6 = (a10-a00)(b00+b01)
    U_total[2, 5] = 1  # a10
    U_total[0, 5] = -1 # a00
    V_total[0, 5] = 1  # b00
    V_total[1, 5] = 1  # b01
    # C11 = M1 - M2 + M3 + M6.
    # M6 contributes +1 to C11.
    W_total[3, 5] = 1  # C11
    
    # Col 6: M7 = (a01-a11)(b10+b11)
    U_total[1, 6] = 1  # a01
    U_total[3, 6] = -1 # a11
    V_total[3, 6] = 1  # b10
    V_total[4, 6] = 1  # b11
    # C00 = M1 - M4 + M5 - M7. C01 = M3 + M7.
    # M7 contributes -1 to C00 and +1 to C01.
    W_total[0, 6] = -1 # C00
    W_total[2, 6] = 1  # C01
    
    # --- Schoolbook Part for C_right (Columns 7-10) ---
    # C_right = A * B_right
    # B_right = [[b02], [b12]]
    # C[0,2] = a00*b02 + a01*b12
    # C[1,2] = a10*b02 + a11*b12
    
    # Col 7: a00*b02
    U_total[0, 7] = 1
    V_total[2, 7] = 1 # b02 is V[2]
    W_total[4, 7] = 1 # C02 is W[4]
    
    # Col 8: a01*b12
    U_total[1, 8] = 1
    V_total[5, 8] = 1 # b12 is V[5]
    W_total[4, 8] = 1 # C02 is W[4]
    
    # Col 9: a10*b02
    U_total[2, 9] = 1
    V_total[2, 9] = 1 # b02 is V[2]
    W_total[5, 9] = 1 # C12 is W[5]
    
    # Col 10: a11*b12
    U_total[3, 10] = 1
    V_total[5, 10] = 1 # b12 is V[5]
    W_total[5, 10] = 1 # C12 is W[5]
    
    # Verify
    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T_target = get_target_tensor()
    T_calc = np.einsum('ir,jr,kr->ijk', U_total, V_total, W_total)
    
    if not np.allclose(T_calc, T_target, atol=1e-9):
        # Fallback to schoolbook if construction fails (should not happen)
        rank_school = n * m * p
        U_sb = np.zeros((n * m, rank_school))
        V_sb = np.zeros((m * p, rank_school))
        W_sb = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

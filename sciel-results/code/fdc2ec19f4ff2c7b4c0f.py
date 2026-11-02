import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is exact and entries are integer/half-integer."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    # Check entries are integer or half-integer
    for arr in [Ua, Va, Wa]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-9):
            return False
    # Check tensor equation: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Return rank-7 decomposition for 2x2x2 matrix multiplication.
    U (4x7), V (4x7), W (4x7)
    U indexed by i*2+k
    V indexed by k*2+j
    W indexed by j*2+i
    """
    # The 7 multiplications M1..M7
    # M1 = A00 * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A11) * (B00 + B10)
    # M6 = (A01 - A10) * (B10 + B11)
    # M7 = (A00 - A01) * (B01 + B11)
    
    # U matrix: coefficients for A entries
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # V matrix: coefficients for B entries
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # W matrix: coefficients for C entries
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1 # A00
    V[0, 0] = 1 # B00
    V[3, 0] = 1 # B11
    W[0, 0] = 1 # C00
    W[3, 0] = 1 # C11
    
    # M2
    U[2, 1] = 1 # A10
    U[3, 1] = 1 # A11
    V[0, 1] = 1 # B00
    W[2, 1] = 1 # C10
    W[3, 1] = -1 # C11
    
    # M3
    U[0, 2] = 1 # A00
    V[1, 2] = 1 # B01
    V[3, 2] = -1 # B11
    W[1, 2] = 1 # C01
    W[3, 2] = 1 # C11
    
    # M4
    U[3, 3] = 1 # A11
    V[2, 3] = 1 # B10
    V[0, 3] = -1 # B00
    W[0, 3] = 1 # C00
    W[2, 3] = 1 # C10
    
    # M5
    U[0, 4] = 1 # A00
    U[3, 4] = 1 # A11
    V[0, 4] = 1 # B00
    V[2, 4] = 1 # B10
    W[0, 4] = -1 # C00
    W[1, 4] = 1 # C01
    W[2, 4] = 1 # C10
    
    # M6
    U[1, 5] = 1 # A01
    U[2, 5] = -1 # A10
    V[2, 5] = 1 # B10
    V[3, 5] = 1 # B11
    W[3, 5] = 1 # C11
    
    # M7
    U[0, 6] = 1 # A00
    U[1, 6] = -1 # A01
    V[1, 6] = 1 # B01
    V[3, 6] = 1 # B11
    W[0, 6] = 1 # C00
    
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_4x4x4():
    """Construct rank-49 decomposition for 4x4x4 matrix multiplication.
    U (16x49), V (16x49), W (16x49)
    """
    U_2, V_2, W_2 = _strassen_2x2()
    U_2 = np.array(U_2)
    V_2 = np.array(V_2)
    W_2 = np.array(W_2)
    
    # We have 7 high-level multiplications M1..M7.
    # Each Mk is a 2x2x2 multiplication.
    # We need to map the 4x4 A and B to the inputs of these 7 multiplications.
    
    # A is 4x4, indexed by (i,k) where i,k in 0..3.
    # Block A_{ij} is 2x2.
    # Let's define the linear combinations for A and B for each of the 7 terms.
    
    # Term 1: A00 * (B00 + B11)
    # A part: A00 (indices 0,1,2,3 in 4x4? No, A00 is rows 0,1 cols 0,1)
    # B part: B00 + B11 (B00 is rows 0,1 cols 0,1; B11 is rows 2,3 cols 2,3)
    
    # Let's build the global U, V, W.
    # Rank 49.
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # For each of the 7 Strassen steps (s=0..6):
    # We have a 2x2x2 decomposition.
    # Let's say the 2x2x2 decomposition is U_s (4x7), V_s (4x7), W_s (4x7).
    # The inputs to this 2x2x2 are A_s (2x2) and B_s (2x2).
    # A_s is formed from A blocks.
    # B_s is formed from B blocks.
    
    # We need to express A_s[i,k] as sum of A[global_i, global_k] * coeff.
    # And B_s[k,j] as sum of B[global_k, global_j] * coeff.
    
    # Let's define the block indices for A and B.
    # A blocks: A00, A01, A10, A11.
    # A00: rows 0,1; cols 0,1.
    # A01: rows 0,1; cols 2,3.
    # A10: rows 2,3; cols 0,1.
    # A11: rows 2,3; cols 2,3.
    
    # B blocks: B00, B01, B10, B11.
    # B00: rows 0,1; cols 0,1.
    # B01: rows 0,1; cols 2,3.
    # B10: rows 2,3; cols 0,1.
    # B11: rows 2,3; cols 2,3.
    
    # Strassen formulas for 2x2 block matrices:
    # M1 = A00 * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A11) * (B00 + B10)
    # M6 = (A01 - A10) * (B10 + B11)
    # M7 = (A00 - A01) * (B01 + B11)
    
    # For each step s, we define the coefficients for A and B blocks.
    # Let A_blocks be a 4x4 matrix of coefficients for the 4 A blocks.
    # Let B_blocks be a 4x4 matrix of coefficients for the 4 B blocks.
    # Actually, A_s is a 2x2 matrix. Each entry A_s[r,c] is a linear combination of A blocks.
    # But A_s is just one of the A blocks or a sum.
    # In Strassen, A_s and B_s are 2x2 matrices (blocks).
    # So A_s is either A00, A01, etc., or a sum like A00+A11.
    # So A_s is a 2x2 matrix whose entries are linear combinations of the entries of the A blocks.
    
    # Let's map the 2x2 local indices (r,c) to the 4x4 global indices.
    # If A_s is A00, then A_s[0,0] = A[0,0], A_s[0,1] = A[0,1], etc.
    # If A_s is A00 + A11, then A_s[0,0] = A[0,0] + A[2,2], etc.
    
    # Let's define the A and B block selections for each step.
    # Step 0: A=A00, B=B00+B11
    # Step 1: A=A10+A11, B=B00
    # Step 2: A=A00, B=B01-B11
    # Step 3: A=A11, B=B10-B00
    # Step 4: A=A00+A11, B=B00+B10
    # Step 5: A=A01-A10, B=B10+B11
    # Step 6: A=A00-A01, B=B01+B11
    
    # We will iterate over the 7 steps.
    for s in range(7):
        # Determine A and B block coefficients
        # A_coeffs: 4x4 matrix where A_coeffs[bi, bj] is the coefficient of A block (bi, bj)
        # But A_s is a 2x2 matrix.
        # Let's just say A_s is formed by a linear combination of the 4 A blocks.
        # Since the blocks are 2x2, the linear combination is element-wise.
        
        A_coeffs = np.zeros((2, 2)) # for A blocks (row_block, col_block)
        B_coeffs = np.zeros((2, 2)) # for B blocks
        
        if s == 0:
            A_coeffs[0,0] = 1
            B_coeffs[0,0] = 1
            B_coeffs[1,1] = 1
        elif s == 1:
            A_coeffs[1,0] = 1
            A_coeffs[1,1] = 1
            B_coeffs[0,0] = 1
        elif s == 2:
            A_coeffs[0,0] = 1
            B_coeffs[0,1] = 1
            B_coeffs[1,1] = -1
        elif s == 3:
            A_coeffs[1,1] = 1
            B_coeffs[1,0] = 1
            B_coeffs[0,0] = -1
        elif s == 4:
            A_coeffs[0,0] = 1
            A_coeffs[1,1] = 1
            B_coeffs[0,0] = 1
            B_coeffs[1,0] = 1
        elif s == 5:
            A_coeffs[0,1] = 1
            A_coeffs[1,0] = -1
            B_coeffs[1,0] = 1
            B_coeffs[1,1] = 1
        elif s == 6:
            A_coeffs[0,0] = 1
            A_coeffs[0,1] = -1
            B_coeffs[0,1] = 1
            B_coeffs[1,1] = 1
            
        # Now, for each of the 7 local multiplications in this step:
        for l in range(7):
            col = s * 7 + l
            
            # U part:
            # U_2 is 4x7. U_2[i*2+k, l] is the coefficient for A_s[i,k].
            # A_s[i,k] = sum_{bi,bj} A_coeffs[bi,bj] * A_block[bi,bj][i,k]
            # A_block[bi,bj][i,k] corresponds to A[2*bi+i, 2*bj+k]
            
            for i in range(2):
                for k in range(2):
                    u_val = U_2[i*2+k, l]
                    if u_val != 0:
                        for bi in range(2):
                            for bj in range(2):
                                if A_coeffs[bi,bj] != 0:
                                    global_i = 2*bi + i
                                    global_k = 2*bj + k
                                    U[global_i*4 + global_k, col] += u_val * A_coeffs[bi,bj]
                                    
            # V part:
            # V_2 is 4x7. V_2[k*2+j, l] is the coefficient for B_s[k,j].
            # B_s[k,j] = sum_{bi,bj} B_coeffs[bi,bj] * B_block[bi,bj][k,j]
            # B_block[bi,bj][k,j] corresponds to B[2*bi+k, 2*bj+j]
            
            for k in range(2):
                for j in range(2):
                    v_val = V_2[k*2+j, l]
                    if v_val != 0:
                        for bi in range(2):
                            for bj in range(2):
                                if B_coeffs[bi,bj] != 0:
                                    global_k = 2*bi + k
                                    global_j = 2*bj + j
                                    V[global_k*4 + global_j, col] += v_val * B_coeffs[bi,bj]
                                    
            # W part:
            # W_2 is 4x7. W_2[j*2+i, l] is the coefficient for C_s[j,i].
            # C_s[j,i] contributes to C_block[bi,bj][j,i] where bi,bj are determined by the Strassen output formula.
            # Wait, the output C is formed by summing M1..M7.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M6
            
            # So for each step s (which produces M_s), we need to add its contribution to the appropriate C blocks.
            # Let's define the output coefficients for each step.
            # Out_coeffs: 2x2 matrix for C blocks.
            
            Out_coeffs = np.zeros((2, 2))
            if s == 0: # M1
                Out_coeffs[0,0] = 1
                Out_coeffs[1,1] = 1
            elif s == 1: # M2
                Out_coeffs[1,0] = 1
                Out_coeffs[1,1] = -1
            elif s == 2: # M3
                Out_coeffs[0,1] = 1
                Out_coeffs[1,1] = 1
            elif s == 3: # M4
                Out_coeffs[0,0] = 1
                Out_coeffs[1,0] = 1
            elif s == 4: # M5
                Out_coeffs[0,0] = -1
                Out_coeffs[0,1] = 1
                Out_coeffs[1,0] = 1
            elif s == 5: # M6
                Out_coeffs[1,1] = 1
            elif s == 6: # M7
                Out_coeffs[0,0] = 1
                
            for i in range(2):
                for j in range(2):
                    w_val = W_2[j*2+i, l]
                    if w_val != 0:
                        for bi in range(2):
                            for bj in range(2):
                                if Out_coeffs[bi,bj] != 0:
                                    global_i = 2*bi + i
                                    global_j = 2*bj + j
                                    # W is indexed by [j*n + i] ? No, W is [k*n + i] in the main loop?
                                    # The task says W[k*n+i, r].
                                    # In the verification: W[k*n+i, :].
                                    # k is the column of C (which is j in my notation).
                                    # i is the row of C.
                                    # So index is j*4 + i.
                                    W[global_j*4 + global_i, col] += w_val * Out_coeffs[bi,bj]
                                    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 4, 4, 9
    
    # Get 4x4x4 decomposition
    U_444, V_444, W_444 = _strassen_4x4x4()
    
    # Get 4x4x1 decomposition (schoolbook)
    # Rank 16
    U_441 = [[0]*16 for _ in range(16)]
    V_441 = [[0]*16 for _ in range(4)] # 4x1 = 4 entries
    W_441 = [[0]*16 for _ in range(4)] # 1x4 = 4 entries (j=0, i=0..3 -> index i)
    
    col = 0
    for i in range(4):
        for k in range(4):
            U_441[i*4+k][col] = 1
            V_441[k*1+0][col] = 1 # B is 4x1, index k
            W_441[0*4+i][col] = 1 # C is 4x1, index i (j=0)
            col += 1
            
    # Combine
    # Total rank = 49 + 49 + 16 = 114
    total_rank = 114
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)] # 4x9
    W = [[0]*total_rank for _ in range(36)] # 9x4
    
    # Part 1: 4x4x4 for cols 0-3 of B, cols 0-3 of C
    for r in range(49):
        for i in range(16):
            U[i][r] = U_444[i][r]
        for k in range(4):
            for j in range(4):
                V[k*9+j][r] = V_444[k*4+j][r]
        for j in range(4):
            for i in range(4):
                W[j*4+i][r] = W_444[j*4+i][r]
                
    # Part 2: 4x4x4 for cols 4-7 of B, cols 4-7 of C
    for r in range(49):
        for i in range(16):
            U[i][r+49] = U_444[i][r]
        for k in range(4):
            for j in range(4):
                V[k*9+(j+4)][r+49] = V_444[k*4+j][r]
        for j in range(4):
            for i in range(4):
                W[(j+4)*4+i][r+49] = W_444[j*4+i][r]
                
    # Part 3: 4x4x1 for col 8 of B, col 8 of C
    for r in range(16):
        for i in range(16):
            U[i][r+98] = U_441[i][r]
        for k in range(4):
            V[k*9+8][r+98] = V_441[k][r]
        for i in range(4):
            W[8*4+i][r+98] = W_441[i][r]
            
    return {"U": U, "V": V, "W": W}

"""Matrix multiplication 4x4x9 rank-minimal decomposition using Strassen."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build Strassen-based decomposition
    # Split B (4x9) into B1(4x4), B2(4x4), B3(4x1)
    # C = A*B1 + A*B2 + A*B3
    # A*B1 and A*B2 use 4x4x4 Strassen (rank 49 each)
    # A*B3 uses schoolbook 4x4x4x1 (rank 16)
    # Total rank: 49 + 49 + 16 = 114
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper: build Strassen decomposition for 2x2 x 2x2
    # Returns U, V, W for the 2x2x2 tensor
    def strassen_2x2():
        # Indices: a = i*2+j, b = j*2+k, c = k*2+i for i,j,k in {0,1}
        # Strassen multiplications:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a11)(b10+b11)
        
        # U matrix: rows are a-indices (0-3), cols are 7 Strassen terms
        # V matrix: rows are b-indices (0-3), cols are 7 Strassen terms
        # W matrix: rows are c-indices (0-3), cols are 7 Strassen terms
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1 = (a00+a11)(b00+b11)
        # a00 is index 0, a11 is index 3
        # b00 is index 0, b11 is index 3
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # c00 = m1 + m4 - m5 + m7, c01 = m3 + m5, c10 = m2 + m4, c11 = m1 - m2 + m3 + m6
        W[0, 0] = 1; W[2, 0] = 1  # c00 and c11 both use m1
        
        # m2 = (a10+a11)b00
        # a10 is index 2, a11 is index 3, b00 is index 0
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1  # c10 uses m2
        W[3, 1] = -1  # c11 uses -m2
        
        # m3 = a00(b01-b11)
        # a00 is index 0, b01 is index 1, b11 is index 3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1  # c01 uses m3
        W[3, 2] = 1  # c11 uses m3
        
        # m4 = a11(b10-b00)
        # a11 is index 3, b10 is index 2, b00 is index 0
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1  # c00 uses m4
        W[2, 3] = 1  # c10 uses m4
        
        # m5 = (a00+a01)b11
        # a00 is index 0, a01 is index 1, b11 is index 3
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1  # c00 uses -m5
        W[1, 4] = 1  # c01 uses m5
        
        # m6 = (a10-a00)(b00+b01)
        # a10 is index 2, a00 is index 0, b00 is index 0, b01 is index 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # c11 uses m6
        
        # m7 = (a01-a11)(b10+b11)
        # a01 is index 1, a11 is index 3, b10 is index 2, b11 is index 3
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # c00 uses m7
        
        return U, V, W
    
    # Helper: build 4x4 x 4x4 decomposition using Strassen on 2x2 blocks
    def strassen_4x4():
        # A and B are 4x4, viewed as 2x2 blocks of 2x2 matrices
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # C = A*B has blocks C11, C12, C21, C22
        # Using Strassen on the block level:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each Mi is a 2x2 x 2x2 product, rank 7
        # Total rank: 7*7 = 49
        
        U_block = np.zeros((4, 7))  # For block-level U
        V_block = np.zeros((4, 7))  # For block-level V
        W_block = np.zeros((4, 7))  # For block-level W
        
        # Block indices: A11=0, A12=1, A21=2, A22=3 (similarly for B, C)
        # M1 = (A11+A22)(B11+B22)
        U_block[0, 0] = 1; U_block[3, 0] = 1
        V_block[0, 0] = 1; V_block[3, 0] = 1
        W_block[0, 0] = 1; W_block[3, 0] = 1  # C11 and C22 both use M1
        
        # M2 = (A21+A22)B11
        U_block[2, 1] = 1; U_block[3, 1] = 1
        V_block[0, 1] = 1
        W_block[2, 1] = 1  # C21 uses M2
        W_block[3, 1] = -1  # C22 uses -M2
        
        # M3 = A11(B12-B22)
        U_block[0, 2] = 1
        V_block[1, 2] = 1; V_block[3, 2] = -1
        W_block[1, 2] = 1  # C12 uses M3
        W_block[3, 2] = 1  # C22 uses M3
        
        # M4 = A22(B21-B11)
        U_block[3, 3] = 1
        V_block[2, 3] = 1; V_block[0, 3] = -1
        W_block[0, 3] = 1  # C11 uses M4
        W_block[2, 3] = 1  # C21 uses M4
        
        # M5 = (A11+A12)B22
        U_block[0, 4] = 1; U_block[1, 4] = 1
        V_block[3, 4] = 1
        W_block[0, 4] = -1  # C11 uses -M5
        W_block[1, 4] = 1  # C12 uses M5
        
        # M6 = (A21-A11)(B11+B12)
        U_block[2, 5] = 1; U_block[0, 5] = -1
        V_block[0, 5] = 1; V_block[1, 5] = 1
        W_block[3, 5] = 1  # C22 uses M6
        
        # M7 = (A12-A22)(B21+B22)
        U_block[1, 6] = 1; U_block[3, 6] = -1
        V_block[2, 6] = 1; V_block[3, 6] = 1
        W_block[0, 6] = 1  # C11 uses M7
        
        # Now expand each block-level term into 2x2 Strassen
        U_4x4 = np.zeros((16, 49))
        V_4x4 = np.zeros((16, 49))
        W_4x4 = np.zeros((16, 49))
        
        # Block to row index mapping:
        # Block 0 (A11): rows 0-3, Block 1 (A12): rows 4-7
        # Block 2 (A21): rows 8-11, Block 3 (A22): rows 12-15
        # Same for V and W
        
        U2, V2, W2 = strassen_2x2()
        
        for block_idx in range(4):
            for strassen_term in range(7):
                if U_block[block_idx, strassen_term] != 0:
                    coeff_u = U_block[block_idx, strassen_term]
                    # This term contributes to 49 total terms
                    # The column index in the 49-dim space is block_idx*7 + strassen_term
                    col = block_idx * 7 + strassen_term
                    row_offset = block_idx * 4
                    U_4x4[row_offset:row_offset+4, col] = coeff_u * U2[:, strassen_term]
                
                if V_block[block_idx, strassen_term] != 0:
                    coeff_v = V_block[block_idx, strassen_term]
                    col = block_idx * 7 + strassen_term
                    row_offset = block_idx * 4
                    V_4x4[row_offset:row_offset+4, col] = coeff_v * V2[:, strassen_term]
                
                if W_block[block_idx, strassen_term] != 0:
                    coeff_w = W_block[block_idx, strassen_term]
                    col = block_idx * 7 + strassen_term
                    row_offset = block_idx * 4
                    W_4x4[row_offset:row_offset+4, col] = coeff_w * W2[:, strassen_term]
        
        return U_4x4, V_4x4, W_4x4
    
    # Build the full 4x4 x 4x9 decomposition
    # C = A*B where B = [B1 | B2 | B3], B1(4x4), B2(4x4), B3(4x1)
    
    # Part 1: A*B1 (4x4 x 4x4), rank 49
    U1, V1, W1 = strassen_4x4()
    
    # Part 2: A*B2 (4x4 x 4x4), rank 49
    U2, V2, W2 = strassen_4x4()
    
    # Part 3: A*B3 (4x4 x 4x1), rank 16 (schoolbook)
    # For matrix-vector: C[i,k] = sum_j A[i,j]*B[j,k]
    # With k fixed (only k=0), C[i] = sum_j A[i,j]*B[j]
    # U rows: i*4+j (16 rows), V rows: j (4 rows for B3), W rows: i (4 rows)
    U3 = np.zeros((16, 16))
    V3 = np.zeros((4, 16))
    W3 = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U3[i * 4 + j, col] = 1
            V3[j, col] = 1
            W3[i, col] = 1
    
    # Now assemble the full decomposition
    # U: 16 rows, 114 cols (49 + 49 + 16)
    # V: 36 rows (4*9), 114 cols
    # W: 36 rows (9*4), 114 cols
    
    U_full = np.zeros((16, 114))
    V_full = np.zeros((36, 114))
    W_full = np.zeros((36, 114))
    
    # Copy part 1 (A*B1)
    U_full[:, 0:49] = U1
    V_full[0:16, 0:49] = V1  # B1 is columns 0-3 of B, so rows 0-3 in V
    W_full[0:16, 0:49] = W1  # C1 is columns 0-3 of C, so rows 0-3 in W
    
    # Copy part 2 (A*B2)
    U_full[:, 49:98] = U2
    V_full[4:8, 49:98] = V2  # B2 is columns 4-7 of B, so rows 4-7 in V
    W_full[16:24, 49:98] = W2  # C2 is columns 4-7 of C, so rows 16-23 in W
    
    # Copy part 3 (A*B3)
    U_full[:, 98:114] = U3
    V_full[8:12, 98:114] = V3  # B3 is column 8 of B, so row 8 in V
    W_full[28:32, 98:114] = W3  # C3 is column 8 of C, so rows 28-31 in W
    
    # Verify the decomposition
    def check_decomposition(U, V, W, tol=1e-9):
        n, m, p = 4, 4, 9
        U_rows, V_rows, W_rows = n*m, m*p, p*n
        r = U.shape[1]
        
        # Check all triple entries = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check some non-triple entries = 0 (spot check)
        import random
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, U_rows - 1)
            b = random.randint(0, V_rows - 1)
            c = random.randint(0, W_rows - 1)
            # Check if this is a triple
            is_triple = False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if a == i*m + j and b == j*p + k and c == k*n + i:
                            is_triple = True
            if not is_triple:
                val = np.dot(U[a, :] * V[b, :], W[c, :])
                if abs(val) > tol:
                    return False
        return True
    
    if check_decomposition(U_full, V_full, W_full):
        # Round to half-integers if possible
        U_rounded = np.round(U_full * 2) / 2
        V_rounded = np.round(V_full * 2) / 2
        W_rounded = np.round(W_full * 2) / 2
        
        if check_decomposition(U_rounded, V_rounded, W_rounded):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        else:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook
        rank_school = 4 * 4 * 9
        U_school = np.zeros((16, rank_school))
        V_school = np.zeros((36, rank_school))
        W_school = np.zeros((36, rank_school))
        for col in range(rank_school):
            i = col // (4 * 9)
            rem = col % (4 * 9)
            j = rem // 9
            k = rem % 9
            U_school[i * 4 + j, col] = 1
            V_school[j * 9 + k, col] = 1
            W_school[k * 4 + i, col] = 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END

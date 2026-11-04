import numpy as np
import time

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 140.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the decomposition correctly computes C = A @ B.
    The evaluator convention for W is W[k*n+i, r] corresponding to C[i,k].
    Since n=p=5 here, we use idx = i*n + k for C[i,k].
    """
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # L_r = sum_{i,j} U[i*m+j, r] * A[i,j]
    # R_r = sum_{j,k} V[j*p+k, r] * B[j,k]
    L = U_arr @ A_flat
    R = V_arr @ B_flat
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            # The task specifies W[k*n+i, r].
            # For C[i,k], the index into W is k*n + i.
            idx = k * n + i
            C_computed[i, k] = np.sum(W_arr[idx, :] * L * R)
            
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    """
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for A (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _strassen_4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using Strassen's algorithm recursively.
    Rank = 7 * 7 = 49.
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2()
    
    # We need to compose the 2x2 Strassen decomposition with itself to form a 4x4 decomposition.
    # The 4x4 matrix is viewed as 4 blocks of 2x2.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2.
    # C = A @ B.
    
    # The Strassen decomposition for 4x4 uses 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication uses 7 scalar multiplications (Strassen 2x2).
    # Total rank = 7 * 7 = 49.
    
    # Let's define the indices.
    # A_flat index: i*4 + j.
    # Block A00 corresponds to A[0:2, 0:2] -> indices 0,1,4,5.
    # Block A01 corresponds to A[0:2, 2:4] -> indices 2,3,6,7.
    # Block A10 corresponds to A[2:4, 0:2] -> indices 8,9,12,13.
    # Block A11 corresponds to A[2:4, 2:4] -> indices 10,11,14,15.
    
    # Similarly for B and C.
    
    # We will iterate over the 7 "outer" Strassen steps (let's call them S1..S7).
    # For each outer step, we need to form the linear forms for the two 2x2 matrices being multiplied.
    # Then we use the 7 "inner" Strassen steps (T1..T7) to decompose the 2x2 multiplication.
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Outer Strassen steps (indices 0..6)
    # S1: (A00 + A11) * (B00 + B11)
    # S2: (A10 + A11) * B00
    # S3: A00 * (B01 - B11)
    # S4: A11 * (B10 - B00)
    # S5: (A00 + A01) * B11
    # S6: (A10 - A00) * (B00 + B01)
    # S7: (A01 - A11) * (B10 + B11)
    
    # We need to map the block indices to the flat indices.
    # A00: [0,1,4,5]
    # A01: [2,3,6,7]
    # A10: [8,9,12,13]
    # A11: [10,11,14,15]
    
    # B00: [0,1,4,5]
    # B01: [2,3,6,7]
    # B10: [8,9,12,13]
    # B11: [10,11,14,15]
    
    # C00: [0,1,4,5]
    # C01: [2,3,6,7]
    # C10: [8,9,12,13]
    # C11: [10,11,14,15]
    
    # For each outer step s (0..6), and inner step t (0..6):
    # col = s * 7 + t
    
    # The linear form for the first 2x2 matrix in step s is a sum of blocks of A.
    # The linear form for the second 2x2 matrix in step s is a sum of blocks of B.
    # We need to expand these into the flat indices using the 2x2 Strassen coefficients.
    
    # Let's define the block coefficients for each outer step.
    # A_coeffs[s][block] and B_coeffs[s][block]
    # Blocks: 00, 01, 10, 11
    
    A_blocks = {0: [0,1,4,5], 1: [2,3,6,7], 2: [8,9,12,13], 3: [10,11,14,15]}
    B_blocks = {0: [0,1,4,5], 1: [2,3,6,7], 2: [8,9,12,13], 3: [10,11,14,15]}
    C_blocks = {0: [0,1,4,5], 1: [2,3,6,7], 2: [8,9,12,13], 3: [10,11,14,15]}
    
    # Outer step definitions:
    # s=0: A00+A11, B00+B11
    # s=1: A10+A11, B00
    # s=2: A00, B01-B11
    # s=3: A11, B10-B00
    # s=4: A00+A01, B11
    # s=5: A10-A00, B00+B01
    # s=6: A01-A11, B10+B11
    
    A_outer = [
        {0: 1, 3: 1}, # s0
        {2: 1, 3: 1}, # s1
        {0: 1},       # s2
        {3: 1},       # s3
        {0: 1, 1: 1}, # s4
        {2: 1, 0: -1},# s5
        {1: 1, 3: -1} # s6
    ]
    
    B_outer = [
        {0: 1, 3: 1}, # s0
        {0: 1},       # s1
        {1: 1, 3: -1},# s2
        {2: 1, 0: -1},# s3
        {3: 1},       # s4
        {0: 1, 1: 1}, # s5
        {2: 1, 3: 1}  # s6
    ]
    
    # C_outer: How the results M_s contribute to C blocks.
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    C_outer = [
        {0: 1, 1: 1, 2: 1, 3: 1}, # M0 contributes to all? No, M0 is a 2x2 matrix.
        # Wait, M_s is a 2x2 matrix. So it contributes to all 4 blocks of C.
        # The coefficients are the same for all 4 blocks of the resulting 2x2 matrix?
        # No, the Strassen formula for C is:
        # C00 = M00 + M30 - M40 + M60
        # C01 = M20 + M40
        # C10 = M10 + M30
        # C11 = M00 - M10 + M20 + M50
        # Where M_s is the result of the s-th multiplication.
        # So for a given s, the contribution to C_block_b is C_coeff[s][b] * M_s.
        # And M_s itself is a 2x2 matrix, so it contributes to all 4 sub-blocks of C_block_b.
        # Wait, no. M_s is a 2x2 matrix. C_block_b is a 2x2 matrix.
        # So M_s is added to C_block_b with a scalar coefficient.
        
        # Let's re-verify the Strassen C formula.
        # C = [[C00, C01], [C10, C11]]
        # C00 = M1 + M4 - M5 + M7 (using 1-based indexing)
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Using 0-based indexing (s=0..6):
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        # So for each s, the coefficient for C_block_b is:
        # s=0: C00:1, C01:0, C10:0, C11:1
        # s=1: C00:0, C01:0, C10:1, C11:-1
        # s=2: C00:0, C01:1, C10:0, C11:1
        # s=3: C00:1, C01:0, C10:1, C11:0
        # s=4: C00:-1, C01:1, C10:0, C11:0
        # s=5: C00:0, C01:0, C10:0, C11:1
        # s=6: C00:1, C01:0, C10:0, C11:0
    ]
    
    C_coeffs = [
        {0: 1, 3: 1}, # s0
        {2: 1, 3: -1},# s1
        {1: 1, 3: 1}, # s2
        {0: 1, 2: 1}, # s3
        {0: -1, 1: 1},# s4
        {3: 1},       # s5
        {0: 1}        # s6
    ]
    
    # Now, for each s and t:
    # col = s * 7 + t
    # U4[i, col] = sum_{block_a} A_outer[s][block_a] * U2[sub_idx_a, t]
    # where sub_idx_a is the index of i within the block_a.
    
    # We need a mapping from flat index i to (block_a, sub_idx_a).
    # i=0 -> block 0, sub 0
    # i=1 -> block 0, sub 1
    # i=2 -> block 1, sub 0
    # i=3 -> block 1, sub 1
    # i=4 -> block 0, sub 2
    # i=5 -> block 0, sub 3
    # i=6 -> block 1, sub 2
    # i=7 -> block 1, sub 3
    # i=8 -> block 2, sub 0
    # i=9 -> block 2, sub 1
    # i=10 -> block 3, sub 0
    # i=11 -> block 3, sub 1
    # i=12 -> block 2, sub 2
    # i=13 -> block 2, sub 3
    # i=14 -> block 3, sub 2
    # i=15 -> block 3, sub 3
    
    # Let's create a helper to get (block, sub_idx) from flat index.
    def flat_to_block_sub(idx):
        # 4x4 matrix, blocks are 2x2.
        # Row block: idx // 4 // 2? No.
        # idx = r * 4 + c.
        # r_block = r // 2, c_block = c // 2.
        # r_sub = r % 2, c_sub = c % 2.
        # sub_idx = r_sub * 2 + c_sub.
        r = idx // 4
        c = idx % 4
        r_block = r // 2
        c_block = c // 2
        r_sub = r % 2
        c_sub = c % 2
        block = r_block * 2 + c_block
        sub_idx = r_sub * 2 + c_sub
        return block, sub_idx
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U part
            for i in range(16):
                block_a, sub_a = flat_to_block_sub(i)
                if block_a in A_outer[s]:
                    coeff_a = A_outer[s][block_a]
                    U4[i][col] += coeff_a * U2[sub_a][t]
            
            # V part
            for j in range(16):
                block_b, sub_b = flat_to_block_sub(j)
                if block_b in B_outer[s]:
                    coeff_b = B_outer[s][block_b]
                    V4[j][col] += coeff_b * V2[sub_b][t]
            
            # W part
            for k in range(16):
                block_c, sub_c = flat_to_block_sub(k)
                if block_c in C_coeffs[s]:
                    coeff_c = C_coeffs[s][block_c]
                    W4[k][col] += coeff_c * W2[sub_c][t]
    
    return U4, V4, W4

def _construct_110():
    """
    Constructs a rank-110 decomposition for 5x5x5 matrix multiplication.
    Uses Strassen for the 4x4x4 block (rank 49) and schoolbook for the rest.
    """
    n, m, p = 5, 5, 5
    
    U = [[0]*110 for _ in range(25)]
    V = [[0]*110 for _ in range(25)]
    W = [[0]*110 for _ in range(25)]
    
    # 1. Strassen for the 4x4x4 block (indices 0..15)
    U4, V4, W4 = _strassen_4x4()
    
    for i in range(16):
        for j in range(16):
            for k in range(16):
                # The Strassen decomposition gives U4[i, r], V4[j, r], W4[k, r]
                # We need to map these to the 5x5 matrices.
                # The 4x4 block corresponds to rows/cols 0..3 of the 5x5 matrix.
                # So U4[i, r] goes to U[i, r] in the 5x5 decomposition?
                # Wait, U is (n*m, rank) = (25, 110).
                # The 4x4 block uses indices 0..15 for both U and V.
                # So U[i, r] for i in 0..15, r in 0..48 gets U4[i, r].
                # V[j, r] for j in 0..15, r in 0..48 gets V4[j, r].
                # W[k, r] for k in 0..15, r in 0..48 gets W4[k, r].
                
                # But we also have the cross terms.
                # The rank is 49 + 61 = 110.
                # The first 49 columns are for the 4x4 block.
                # The next 61 columns are for the cross terms.
                
                # For the 4x4 part:
                if i < 16:
                    U[i][r] = U4[i][r]
                if j < 16:
                    V[j][r] = V4[j][r]
                if k < 16:
                    W[k][r] = W4[k][r]
    
    # 2. Cross terms
    # We need to compute C[4, :] = A[4, :] @ B
    # C[:, 4] = A @ B[:, 4]
    # C[4, 4] = A[4, :] @ B[:, 4] (already included in the above two)
    
    # Let's enumerate the cross terms.
    # C[4, k] for k=0..3:
    # C[4, k] = sum_{j=0..4} A[4, j] B[j, k]
    # This involves A[4, 0..3] and B[0..3, 0..3] (which are in the 4x4 block) and A[4, 4] and B[4, k].
    # Wait, the 4x4 block only handles A[0..3, 0..3] and B[0..3, 0..3].
    # So for C[4, k], we need:
    # sum_{j=0..3} A[4, j] B[j, k] + A[4, 4] B[4, k]
    
    # Similarly for C[i, 4]:
    # C[i, 4] = sum_{j=0..3} A[i, j] B[j, 4] + A[i, 4] B[4, 4]
    
    # And C[4, 4] = sum_{j=0..3} A[4, j] B[j, 4] + A[4, 4] B[4, 4]
    
    # We can use schoolbook for these cross terms.
    # Number of terms:
    # C[4, 0..3]: 4 * 5 = 20 terms?
    # C[0..3, 4]: 4 * 5 = 20 terms?
    # C[4, 4]: 5 terms?
    # Total = 45 terms?
    # But we have 61 columns left.
    # Let's re-evaluate the rank.
    # Rank(4x4x4) = 49.
    # Cross terms:
    # A[4, 0..4] @ B[0..4, 0..3] -> 5 * 4 = 20 terms.
    # A[0..3, 0..4] @ B[0..4, 4] -> 4 * 5 = 20 terms.
    # A[4, 0..4] @ B[0..4, 4] -> 5 terms.
    # Total = 45 terms.
    # 49 + 45 = 94.
    # Why did I think 110?
    # Maybe I was using a different decomposition.
    # Let's stick with 94.
    
    # Let's implement the 94-rank decomposition.
    # First 49 columns: Strassen 4x4.
    # Next 45 columns: Schoolbook for cross terms.
    
    # Let's reset U, V, W for 94 rank.
    rank = 94
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    # 1. Strassen 4x4 (cols 0..48)
    U4, V4, W4 = _strassen_4x4()
    for i in range(16):
        for r in range(49):
            U[i][r] = U4[i][r]
            V[i][r] = V4[i][r]
            W[i][r] = W4[i][r]
    
    # 2. Cross terms (cols 49..93)
    col = 49
    
    # C[4, k] = sum_{j=0..4} A[4, j] B[j, k] for k=0..3
    for k in range(4):
        for j in range(5):
            U[4 * 5 + j][col] = 1  # A[4, j]
            V[j * 5 + k][col] = 1  # B[j, k]
            W[k * 5 + 4][col] = 1  # C[4, k] -> W[k*n+4, r] = W[k*5+4, r]
            col += 1
    
    # C[i, 4] = sum_{j=0..4} A[i, j] B[j, 4] for i=0..3
    for i in range(4):
        for j in range(5):
            U[i * 5 + j][col] = 1  # A[i, j]
            V[j * 5 + 4][col] = 1  # B[j, 4]
            W[4 * 5 + i][col] = 1  # C[i, 4] -> W[4*n+i, r] = W[4*5+i, r]
            col += 1
    
    # C[4, 4] = sum_{j=0..4} A[4, j] B[j, 4]
    for j in range(5):
        U[4 * 5 + j][col] = 1  # A[4, j]
        V[j * 5 + 4][col] = 1  # B[j, 4]
        W[4 * 5 + 4][col] = 1  # C[4, 4] -> W[4*5+4, r]
        col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Try rank 94
    result = _construct_110()
    
    # Verify
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

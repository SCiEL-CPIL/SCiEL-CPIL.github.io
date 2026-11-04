"""Program for rank-minimal exact tensor decomposition of 4x4x9 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U, V, W for 2x2x2 matrix multiplication (Strassen).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r] with n=m=p=2
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen formulas for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Row indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # A is 2x2: A[i,j], B is 2x2: B[j,k], C is 2x2: C[i,k]
    
    # M1 = (A11+A22)(B11+B22)
    # U contributes to A: A11 (i=0,j=0) -> U[0,0]=1, A22 (i=1,j=1) -> U[3,0]=1
    # V contributes to B: B11 (j=0,k=0) -> V[0,0]=1, B22 (j=1,k=1) -> V[3,0]=1
    # W: C11 gets +1, C22 gets +1
    # C11: i=0,k=0 -> W[0*2+0]=W[0]
    # C22: i=1,k=1 -> W[1*2+1]=W[3]
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2 = (A21+A22)B11
    # U: A21 (i=1,j=0) -> U[1*2+0]=U[2], A22 (i=1,j=1) -> U[3]
    # V: B11 (j=0,k=0) -> V[0]
    # W: C21 gets +1, C22 gets -1
    # C21: i=1,k=0 -> W[1*2+0]=W[2]
    # C22: i=1,k=1 -> W[3]
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    
    # M3 = A11(B12-B22)
    # U: A11 (i=0,j=0) -> U[0]
    # V: B12 (j=0,k=1) -> V[0*2+1]=V[1], B22 (j=1,k=1) -> V[3]
    # W: C12 gets +1, C22 gets +1
    # C12: i=0,k=1 -> W[0*2+1]=W[1]
    # C22: i=1,k=1 -> W[3]
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[1,2] = 1; W[3,2] = 1
    
    # M4 = A22(B21-B11)
    # U: A22 (i=1,j=1) -> U[3]
    # V: B21 (j=1,k=0) -> V[1*2+0]=V[2], B11 (j=0,k=0) -> V[0]
    # W: C11 gets +1, C21 gets +1
    # C11: i=0,k=0 -> W[0]
    # C21: i=1,k=0 -> W[2]
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[2,3] = 1
    
    # M5 = (A11+A12)B22
    # U: A11 (i=0,j=0) -> U[0], A12 (i=0,j=1) -> U[1]
    # V: B22 (j=1,k=1) -> V[3]
    # W: C11 gets -1, C12 gets +1
    # C11: i=0,k=0 -> W[0]
    # C12: i=0,k=1 -> W[1]
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[1,4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # U: A21 (i=1,j=0) -> U[2], A11 (i=0,j=0) -> U[0]
    # V: B11 (j=0,k=0) -> V[0], B12 (j=0,k=1) -> V[1]
    # W: C22 gets +1
    # C22: i=1,k=1 -> W[3]
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # U: A12 (i=0,j=1) -> U[1], A22 (i=1,j=1) -> U[3]
    # V: B21 (j=1,k=0) -> V[2], B22 (j=1,k=1) -> V[3]
    # W: C11 gets +1
    # C11: i=0,k=0 -> W[0]
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _embed_2x2_to_4x4(U2, V2, W2, block_i, block_j, block_k):
    """Embed a 2x2x2 decomposition into 4x4x4 at specified block positions.
    block_i, block_j, block_k in {0,1} indicate which 2x2 block of the 4x4 matrix.
    Returns U4 (16, rank), V4 (16, rank), W4 (16, rank)
    """
    rank = U2.shape[1]
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))
    
    for r in range(rank):
        for i2 in range(2):
            for j2 in range(2):
                # U index: i*m + j = i*2 + j for 2x2
                u_idx_2 = i2 * 2 + j2
                if U2[u_idx_2, r] != 0:
                    # Map to 4x4: i = block_i*2 + i2, j = block_j*2 + j2
                    i = block_i * 2 + i2
                    j = block_j * 2 + j2
                    U4[i * 4 + j, r] = U2[u_idx_2, r]
        
        for j2 in range(2):
            for k2 in range(2):
                # V index: j*p + k = j*2 + k for 2x2
                v_idx_2 = j2 * 2 + k2
                if V2[v_idx_2, r] != 0:
                    j = block_j * 2 + j2
                    k = block_k * 2 + k2
                    V4[j * 4 + k, r] = V2[v_idx_2, r]
        
        for k2 in range(2):
            for i2 in range(2):
                # W index: k*n + i = k*2 + i for 2x2
                w_idx_2 = k2 * 2 + i2
                if W2[w_idx_2, r] != 0:
                    k = block_k * 2 + k2
                    i = block_i * 2 + i2
                    W4[k * 4 + i, r] = W2[w_idx_2, r]
    
    return U4, V4, W4

def _strassen_4x4():
    """Return U, V, W for 4x4x4 matrix multiplication using Strassen recursion.
    Rank should be 7^2 = 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2()
    
    # For 4x4, we have 4 blocks of 2x2 in each dimension.
    # The 4x4x4 tensor is a sum over block positions.
    # Actually, Strassen for 4x4 uses the 2x2 decomposition recursively.
    # The 4x4 multiplication can be seen as a 2x2 block matrix where each block is 2x2.
    # So we apply Strassen to the block matrix, where each "entry" is a 2x2 matrix.
    # This means we need to do 7 multiplications of 2x2 matrices, each of which costs 7.
    # Total rank = 7 * 7 = 49.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    col_offset = 0
    for bi in range(2):  # block row of A (and C)
        for bj in range(2):  # block col of A (and row of B)
            for bk in range(2):  # block col of B (and col of C)
                # This doesn't quite work because Strassen combines blocks.
                # Let me think again.
                
                # Actually, the standard way: to multiply 4x4 matrices A and B,
                # view them as 2x2 block matrices with 2x2 blocks.
                # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
                # Then C = A*B can be computed via Strassen on these blocks.
                # Each block multiplication (2x2 x 2x2) uses 7 scalar multiplications.
                # So total = 7 * 7 = 49.
                
                pass
    
    # Let me do it properly. The 49 rank decomposition comes from:
    # There are 7 Strassen "slots", each requiring a 2x2x2 multiplication (rank 7).
    # For each Strassen slot s (0..6), and each 2x2x2 rank term t (0..6),
    # we have one rank-1 term.
    
    for s in range(7):  # Strassen slot
        for t in range(7):  # 2x2x2 rank term within this slot
            col = s * 7 + t
            # The U for this slot-t comes from U2[:, t] embedded at the appropriate blocks
            # determined by which blocks Strassen slot s uses.
            
            # Strassen slot s involves specific blocks of A and B.
            # Let's define for each slot s, which (i2, j2) blocks of A are used with what signs,
            # and which (j2, k2) blocks of B are used with what signs,
            # and which (k2, i2) blocks of C are affected.
            
            # Actually, it's easier to just compose: 
            # For Strassen slot s, we need to multiply (linear combo of A blocks) * (linear combo of B blocks).
            # The linear combo of A blocks is a 2x2 matrix M_A_s where M_A_s[i2,j2] is the coefficient.
            # Similarly M_B_s[j2,k2].
            # The product is a 2x2 matrix whose (i2,k2) entry goes to C[i2,k2] with coefficient from W2.
            
            # Let me define the Strassen combinations explicitly for 2x2 block matrices:
            # Slot 0: (A00+A11) * (B00+B11) -> contributes to C00, C11
            # Slot 1: (A10+A11) * B00 -> contributes to C10, C11
            # Slot 2: A00 * (B01-B11) -> contributes to C01, C11
            # Slot 3: A11 * (B10-B00) -> contributes to C00, C10
            # Slot 4: (A00+A01) * B11 -> contributes to C00, C01
            # Slot 5: (A10-A00) * (B00+B01) -> contributes to C11
            # Slot 6: (A01-A11) * (B10+B11) -> contributes to C00
            
            # For each slot, the A part is a 2x2 matrix of coefficients, same for B part.
            # The C part tells which C blocks get contributions and with what signs.
            
            # Let me define these as 2x2 matrices:
            # a_coeffs[s][i2][j2], b_coeffs[s][j2][k2], c_coeffs[s][k2][i2]
            
            a_coeffs = [
                [[1, 0], [0, 1]],   # slot 0: A00+A11
                [[0, 0], [1, 1]],   # slot 1: A10+A11
                [[1, 0], [0, 0]],   # slot 2: A00
                [[0, 0], [0, 1]],   # slot 3: A11
                [[1, 1], [0, 0]],   # slot 4: A00+A01
                [[-1, 0], [1, 0]],  # slot 5: A10-A00
                [[0, 1], [0, -1]],  # slot 6: A01-A11
            ]
            b_coeffs = [
                [[1, 0], [0, 1]],   # slot 0: B00+B11
                [[1, 0], [0, 0]],   # slot 1: B00
                [[0, 1], [0, -1]],  # slot 2: B01-B11
                [[-1, 0], [1, 0]],  # slot 3: B10-B00
                [[0, 0], [0, 1]],   # slot 4: B11
                [[1, 1], [0, 0]],   # slot 5: B00+B01
                [[0, 1], [1, 1]],   # slot 6: B10+B11
            ]
            c_coeffs = [
                [[1, 0], [0, 1]],   # slot 0: C00, C11
                [[0, 1], [0, -1]],  # slot 1: C10, -C11  (wait, let me re-check)
            ]
            
            # I'm getting confused with the exact signs. Let me use a cleaner approach.
            # I'll just use the known rank-49 decomposition by direct construction.
            # Actually, let me just verify by using numpy to check if my construction works.
            
            pass
    
    # Let me take a different, more reliable approach:
    # Use the fact that rank(4x4x4) = 49 via Strassen.
    # I'll construct it by explicitly defining the 49 rank-1 terms.
    
    # Define Strassen 2x2 block combinations more carefully.
    # For multiplying 4x4 matrices as 2x2 block matrices:
    # C = AB where A, B are 4x4 viewed as 2x2 blocks of 2x2.
    
    # The 7 intermediate products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product, which has rank 7 (Strassen).
    # So total rank = 7 * 7 = 49.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    U2, V2, W2 = _strassen_2x2()
    
    # For each Strassen slot s (0..6), the A combination is a 2x2 matrix A_s
    # and B combination is a 2x2 matrix B_s.
    # The product A_s * B_s contributes to C blocks with coefficients.
    
    # A_s as 2x2 (coefficients for A00, A01, A10, A11):
    A_s = [
        [[1, 0], [0, 1]],   # M1: A00 + A11
        [[0, 0], [1, 1]],   # M2: A10 + A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: A00 + A01
        [[-1, 0], [1, 0]],  # M6: -A00 + A10
        [[0, 1], [0, -1]],  # M7: A01 - A11
    ]
    B_s = [
        [[1, 0], [0, 1]],   # M1: B00 + B11
        [[1, 0], [0, 0]],   # M2: B00
        [[0, 1], [0, -1]],  # M3: B01 - B11
        [[-1, 0], [1, 0]],  # M4: -B00 + B10
        [[0, 0], [0, 1]],   # M5: B11
        [[1, 1], [0, 0]],   # M6: B00 + B01
        [[0, 1], [1, 1]],   # M7: B10 + B11
    ]
    # C contributions: C_s[i2][k2] = coefficient for C[i2][k2]
    C_s = [
        [[1, 0], [0, 1]],   # M1: C00 + C11
        [[0, 1], [0, -1]],  # M2: C10 - C11  (hmm, need to verify)
        [[0, 1], [0, 1]],   # M3: C01 + C11
        [[1, 0], [1, 0]],   # M4: C00 + C10
        [[-1, 1], [0, 0]],  # M5: -C00 + C01
        [[0, 0], [0, 1]],   # M6: C11
        [[1, 0], [0, 0]],   # M7: C00
    ]
    
    # Let me verify C_s by checking: C00 = M1 + M4 - M5 + M7
    # From above: M1 contributes +1 to C00, M4 contributes +1 to C00, M5 contributes -1 to C00, M7 contributes +1 to C00.
    # So C00 coefficient = 1 + 1 - 1 + 1 = 2. That's wrong, should be 1.
    
    # I think I have the signs wrong. Let me re-derive.
    # Actually, the Strassen formulas give:
    # C00 = M1 - M5 + M7 + M4
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So:
    # M1: C00 += 1, C11 += 1
    # M2: C10 += 1, C11 -= 1
    # M3: C01 += 1, C11 += 1
    # M4: C00 += 1, C10 += 1
    # M5: C00 -= 1, C01 += 1
    # M6: C11 += 1
    # M7: C00 += 1
    
    C_s = [
        [[1, 0], [0, 1]],   # M1: C00=1, C11=1
        [[0, 0], [1, -1]],  # M2: C10=1, C11=-1
        [[0, 1], [0, 1]],   # M3: C01=1, C11=1
        [[1, 0], [1, 0]],   # M4: C00=1, C10=1
        [[-1, 1], [0, 0]],  # M5: C00=-1, C01=1
        [[0, 0], [0, 1]],   # M6: C11=1
        [[1, 0], [0, 0]],   # M7: C00=1
    ]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # U: for each (i2, j2) block, coefficient is A_s[i2][j2] * U2[i2*2+j2, t]
            for i2 in range(2):
                for j2 in range(2):
                    val = A_s[s][i2][j2] * U2[i2 * 2 + j2, t]
                    if val != 0:
                        i = i2 * 2
                        j = j2 * 2
                        U4[i * 4 + j, col] = val
            
            # V: for each (j2, k2) block, coefficient is B_s[j2][k2] * V2[j2*2+k2, t]
            for j2 in range(2):
                for k2 in range(2):
                    val = B_s[s][j2][k2] * V2[j2 * 2 + k2, t]
                    if val != 0:
                        j = j2 * 2
                        k = k2 * 2
                        V4[j * 4 + k, col] = val
            
            # W: for each (k2, i2) block, coefficient is C_s[k2][i2] * W2[k2*2+i2, t]
            for k2 in range(2):
                for i2 in range(2):
                    val = C_s[s][k2][i2] * W2[k2 * 2 + i2, t]
                    if val != 0:
                        k = k2 * 2
                        i = i2 * 2
                        W4[k * 4 + i, col] = val
    
    return U4, V4, W4

def construct():
    """Construct rank-114 decomposition for 4x4x9 matrix multiplication.
    
    Decomposes into two 4x4x4 (rank 49 each) and one 4x4x1 (rank 16) part.
    Total rank = 49 + 49 + 16 = 114.
    
    U: (16, 114), V: (36, 114), W: (36, 114)
    Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=4, m=4, p=9
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    U4, V4, W4 = _strassen_4x4()
    
    rank = 114
    U = np.zeros((n * m, rank))  # (16, 114)
    V = np.zeros((m * p, rank))  # (36, 114)
    W = np.zeros((p * n, rank))  # (36, 114)
    
    # Part 1: First 4x4x4, using columns 0-3 of B (k=0,1,2,3)
    # U part: same as U4
    U[:, 0:49] = U4
    
    # V part: rows j*9+k where k in 0..3
    for j in range(4):
        for k in range(4):
            V[j * 9 + k, 0:49] = V4[j * 4 + k, 0:49]
    
    # W part: rows k*4+i where k in 0..3
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, 0:49]
    
    # Part 2: Second 4x4x4, using columns 4-7 of B (k=4,5,6,7)
    col_offset = 49
    U[:, col_offset:col_offset+49] = U4
    
    for j in range(4):
        for k in range(4):
            V[j * 9 + (4 + k), col_offset:col_offset+49] = V4[j * 4 + k, 0:49]
    
    for k in range(4):
        for i in range(4):
            W[(4 + k) * 4 + i, col_offset:col_offset+49] = W4[k * 4 + i, 0:49]
    
    # Part 3: 4x4x1, using column 8 of B (k=8)
    # This is just the standard rank-16 decomposition for multiplying 4x4 by 4x1.
    col_offset = 98
    for i in range(4):
        for j in range(4):
            col = col_offset + i * 4 + j
            U[i * 4 + j, col] = 1
            V[j * 9 + 8, col] = 1
            W[8 * 4 + i, col] = 1
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

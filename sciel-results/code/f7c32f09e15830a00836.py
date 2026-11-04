import numpy as np
import time

def _strassen_4x4x4():
    """Return U, V, W for 4x4x4 Strassen decomposition (rank 49)."""
    # Strassen 2x2x2 has rank 7
    # 4x4x4 via two levels of Strassen: rank 7^3 = 343? No, that's not right.
    # Actually 4x4x4: split into 2x2 blocks, use Strassen on each level.
    # Level 1: 2x2x2 Strassen (rank 7) applied to 2x2 blocks
    # But 4x4 = 2x2 blocks of 2x2, so we need 7 multiplications of 2x2x2
    # Each 2x2x2 has rank 7, so total rank = 7 * 7 = 49
    
    # Let me build this explicitly.
    # 2x2 Strassen: M1=(A11+A22)(B11+B22), M2=(A21+A22)B11, M3=A11(B21-B22), 
    #   M4=A22(B11-B21), M5=(A11+A12)B22, M6=(A21-A11)(B11+B12), M7=(A12-A22)(B21+B22)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    
    # For 4x4: A is 4x4, split into four 2x2 blocks.
    # Use Strassen to compute 4 products of 2x2 matrices, but we need 4x4 result.
    # Actually, 4x4 = [[A11,A12],[A21,A22]] * [[B11,B12],[B21,B22]] where each is 2x2
    # Result C is 4x4 = [[C11,C12],[C21,C22]] where each is 2x2
    # C11 = A11*B11 + A12*B21, etc.
    # Each Ci is a 2x2 product, so we can use Strassen for each.
    # But that would be 4 * 7 = 28 ranks. Wait, that's not right either.
    
    # Let me think again. 4x4x4 matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k] for i,j,k in 0..3
    # Split: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # Then C[2*i1+i2, 2*k1+k2] = sum_{j1,j2} A[2*i1+i2, 2*j1+j2] B[2*j1+j2, 2*k1+k2]
    # = sum_{j1} (sum_{j2} A[2*i1+i2, 2*j1+j2] B[2*j1+j2, 2*k1+k2])
    # This is like a block matrix multiplication where each "element" is a 2x2 matrix.
    # So we need to multiply 2x2 block matrices, which requires 7 Strassen multiplications.
    # Each Strassen multiplication involves 7 products of 2x2 matrices.
    # Each 2x2 product has rank 7 (Strassen) or 4 (schoolbook).
    # So total rank = 7 * 7 = 49 (using Strassen at both levels)
    
    # Let me build the 2x2 Strassen decomposition first.
    # For 2x2x2: U is (4, 7), V is (4, 7), W is (4, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # Strassen 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B21-B22)
    # M4 = A22*(B11-B21)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U entries (A-side): U[i*2+j, r]
    # V entries (B-side): V[j*2+k, r]
    # W entries (C-side): W[k*2+i, r]
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (A11+A22)*(B11+B22)
    # A11 is at (0,0), A22 is at (1,1) -> U indices: 0*2+0=0, 1*2+1=3
    # B11 is at (0,0), B22 is at (1,1) -> V indices: 0*2+0=0, 1*2+1=3
    # C11 += M1 -> W[0*2+0, 0] += 1
    # C22 += M1 -> W[1*2+1, 0] += 1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1  # C11 and C22
    
    # Wait, C11 is at output index k*2+i where k=0 (col), i=0 (row) -> 0*2+0=0
    # C22 is at k=1, i=1 -> 1*2+1=3
    # But C11 = M1+M4-M5+M7, so W[0,0] gets contributions from M1, M4, M5, M7
    
    # Let me redo this more carefully.
    # For each Strassen term Mr, we have:
    # Mr = (sum of A entries with coefficients) * (sum of B entries with coefficients)
    # And Mr contributes to C entries with coefficients.
    
    # M1 = (A00+A11)*(B00+B11)
    # U: A00 at idx 0, A11 at idx 3 -> U[0,0]=1, U[3,0]=1
    # V: B00 at idx 0, B11 at idx 3 -> V[0,0]=1, V[3,0]=1
    # C: C00 += M1, C11 += M1 -> W[0,0]=1, W[3,0]=1
    
    # M2 = (A10+A11)*B00
    # U: A10 at idx 2, A11 at idx 3 -> U[2,1]=1, U[3,1]=1
    # V: B00 at idx 0 -> V[0,1]=1
    # C: C10 += M2 -> W[0*2+1, 1] = W[2,1] = 1
    
    # M3 = A00*(B10-B11)
    # U: A00 at idx 0 -> U[0,2]=1
    # V: B10 at idx 2, B11 at idx 3 -> V[2,2]=1, V[3,2]=-1
    # C: C01 += M3 -> W[1*2+0, 2] = W[2,2] = 1
    
    # M4 = A11*(B00-B10)
    # U: A11 at idx 3 -> U[3,3]=1
    # V: B00 at idx 0, B10 at idx 2 -> V[0,3]=1, V[2,3]=-1
    # C: C00 += M4, C10 += M4 -> W[0,3]=1, W[2,3]=1
    
    # M5 = (A00+A01)*B11
    # U: A00 at idx 0, A01 at idx 1 -> U[0,4]=1, U[1,4]=1
    # V: B11 at idx 3 -> V[3,4]=1
    # C: C01 += M5, C00 -= M5 -> W[2,4]=1, W[0,4]=-1
    
    # M6 = (A10-A00)*(B00+B01)
    # U: A10 at idx 2, A00 at idx 0 -> U[2,5]=1, U[0,5]=-1
    # V: B00 at idx 0, B01 at idx 1 -> V[0,6]=1, V[1,6]=1
    # C: C11 += M6 -> W[3,5]=1
    
    # M7 = (A01-A11)*(B10+B11)
    # U: A01 at idx 1, A11 at idx 3 -> U[1,6]=1, U[3,6]=-1
    # V: B10 at idx 2, B11 at idx 3 -> V[2,6]=1, V[3,6]=1
    # C: C00 += M7, C11 -= M7 -> W[0,6]=1, W[3,6]=-1
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ])
    V2 = np.array([
        [1, 1, 0, 1, 0, 1, 0],    # B00
        [0, 0, 0, 0, 0, 1, 0],    # B01
        [0, 0, 1, -1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]    # B11
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 1, 1, 0, 0, 0],    # C10
        [1, 0, 0, 0, 0, 1, -1]    # C11
    ])
    
    return U2, V2, W2

def _expand_4x4x4(U2, V2, W2):
    """Expand 2x2 Strassen to 4x4x4 using block structure."""
    # 4x4 = 2x2 blocks of 2x2
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = [[C00, C01], [C10, C11]] where each Ci is 2x2
    # Cij = sum_k Aik * Bkj (block multiplication)
    # Use Strassen for the 2x2 block multiplication:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B00-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each Mk is a 2x2 product, decomposed with Strassen (rank 7)
    # So total rank = 7 * 7 = 49
    
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block indices: A is 4x4, split into 2x2 blocks
    # A[2*i1+i2, 2*j1+j2] is in block A[i1,j1] at position (i2,j2)
    # B[2*j1+j2, 2*k1+k2] is in block B[j1,k1] at position (j2,k2)
    # C[2*i1+i2, 2*k1+k2] is in block C[i1,k1] at position (i2,k2)
    
    # For each Strassen term at the block level (r1 = 0..6),
    # and each Strassen term at the sub-block level (r2 = 0..6),
    # we have a rank-1 term.
    
    # U4[2*i1+i2, 2*j1+j2, r] = U2_sub[i2*2+j2, r2] * U2_block[i1*2+j1, r1]
    # But wait, the block-level Strassen combines blocks, so:
    # For block-level term r1, the A-side is a linear combination of blocks.
    # Each block is a 2x2 matrix, and we further decompose the product with sub-block Strassen.
    
    # Actually, this is getting complicated. Let me think differently.
    # The 4x4x4 tensor can be seen as a 2x2x2 tensor where each "entry" is a 2x2x2 tensor.
    # The outer 2x2x2 is decomposed with Strassen (rank 7).
    # Each inner 2x2x2 is also decomposed with Strassen (rank 7).
    # Total rank = 7 * 7 = 49.
    
    # U4[i, j, r] where r = r1*7 + r2
    # i = 2*i1 + i2, j = 2*j1 + j2
    # U4[2*i1+i2, 2*j1+j2, r1*7+r2] = U2_block[i1*2+j1, r1] * U2_sub[i2*2+j2, r2]
    # Wait, that's not quite right. The block-level decomposition says:
    # A_block[i1,j1] contributes with coefficient U2_block[i1*2+j1, r1]
    # And within each block, the 2x2 matrix A[i2,j2] (within block [i1,j1]) 
    # contributes with coefficient U2_sub[i2*2+j2, r2]
    # So the total coefficient is the product.
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = 2*i1 + i2
                            j = 2*j1 + j2
                            U[i*m+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = 2*j1 + j2
                            k = 2*k1 + k2
                            V[j*p+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = 2*k1 + k2
                            i = 2*i1 + i2
                            W[k*n+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 4, 4, 7
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    U2, V2, W2 = _strassen_4x4x4()
    U44, V44, W44 = _expand_4x4x4(U2, V2, W2)
    
    # Build 4x4x3 schoolbook decomposition (rank 48)
    # C[i,k] = sum_j A[i,j] B[j,k] for i=0..3, k=0..2
    # U[i*4+j, r] = 1, V[j*3+k, r] = 1, W[k*4+i, r] = 1
    U43 = np.zeros((16, 48))
    V43 = np.zeros((12, 48))
    W43 = np.zeros((28, 48))  # Wait, W shape should be (p*n, rank) = (7*4, rank) = (28, rank)
    # But for the 4x4x3 sub-problem, the output is 4x3, so W should have 4*3=12 rows?
    # No, W is for the full problem. Let me reconsider.
    
    # Actually, for the full 4x4x7 problem:
    # U: (16, rank) - A matrix entries
    # V: (28, rank) - B matrix entries (4*7)
    # W: (28, rank) - C matrix entries (7*4)
    
    # For the 4x4x4 sub-problem (k=0..3):
    # U44: (16, 49)
    # V44: (28, 49) but only first 16 rows (j*4+k for k=0..3) are non-zero
    # W44: (28, 49) but only first 16 rows (k*4+i for k=0..3) are non-zero
    
    # For the 4x4x3 sub-problem (k=4..6):
    # U43: (16, 48)
    # V43: (28, 48) but only rows 16..27 (j*4+k for k=4..6) are non-zero
    # W43: (28, 48) but only rows 16..27 (k*4+i for k=4..6) are non-zero
    
    # Let me rebuild V and W for the sub-problems with correct shapes.
    
    # 4x4x4 sub-problem:
    V44_full = np.zeros((28, 49))
    W44_full = np.zeros((28, 49))
    # V44[j*4+k, r] for j=0..3, k=0..3 -> rows 0..15
    V44_full[:16, :] = V44
    # W44[k*4+i, r] for k=0..3, i=0..3 -> rows 0..15
    W44_full[:16, :] = W44
    
    # 4x4x3 sub-problem (schoolbook):
    U43_full = np.zeros((16, 48))
    V43_full = np.zeros((28, 48))
    W43_full = np.zeros((28, 48))
    r = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):  # k=0..2 for the sub-problem, actual k=4..6
                U43_full[i*4+j, r] = 1
                V43_full[j*7+(4+k), r] = 1  # B[j, 4+k]
                W43_full[(4+k)*4+i, r] = 1  # C[4+k, i]
                r += 1
    
    # Concatenate
    U = np.hstack([U44, U43_full])
    V = np.hstack([V44_full, V43_full])
    W = np.hstack([W44_full, W43_full])
    
    rank = U.shape[1]
    
    # Verify
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

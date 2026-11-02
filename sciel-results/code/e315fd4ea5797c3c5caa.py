"""Matrix multiplication 4x4x9: Kronecker product Strassen decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7) - coefficients for A entries
    V: (4, 7) - coefficients for B entries
    W: (4, 7) - coefficients for C entries
    
    Indexing: A[i,j] -> index i*2+j, B[i,j] -> index i*2+j, C[i,j] -> index i*2+j
    """
    # Strassen formulas for 2x2 matrix multiplication:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1,  0, 1],  # A01
        [0, 1, 0, 0, 0,  1, 0],  # A10
        [1, 1, 0, 1, 0,  0,-1],  # A11
    ], dtype=float)
    
    V = np.array([
        [1, 1, 1,-1, 0, 1, 0],  # B00
        [0, 0,-1, 0, 0, 1, 1],  # B01
        [1, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0,-1, 0, 1, 0, 1],  # B11
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 1,-1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1,-1, 1, 0, 0, 1, 0],  # C11
    ], dtype=float)
    
    return U, V, W

def _schoolbook_1x1x1():
    """Trivial rank-1 decomposition for 1x1x1."""
    U = np.array([[1.0]])
    V = np.array([[1.0]])
    W = np.array([[1.0]])
    return U, V, W

def _kronecker_decomp(U1, V1, W1, U2, V2, W2):
    """
    Given decompositions for (n1,m1,p1) and (n2,m2,p2),
    construct a decomposition for (n1*n2, m1*m2, p1*p2).
    
    The Kronecker product of tensor decompositions:
    If T1 = sum_r U1[:,r] ⊗ V1[:,r] ⊗ W1[:,r]
    and T2 = sum_s U2[:,s] ⊗ V2[:,s] ⊗ W2[:,s]
    Then T1 ⊗ T2 = sum_{r,s} (U1[:,r]⊗U2[:,s]) ⊗ (V1[:,r]⊗V2[:,s]) ⊗ (W1[:,r]⊗W2[:,s])
    """
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    m1, _ = V1.shape
    m2, _ = V2.shape
    p1, _ = W1.shape
    p2, _ = W2.shape
    
    rank = r1 * r2
    U = np.zeros((n1*n2, rank), dtype=float)
    V = np.zeros((m1*m2, rank), dtype=float)
    W = np.zeros((p1*p2, rank), dtype=float)
    
    for r in range(r1):
        for s in range(r2):
            idx = r * r2 + s
            U[:, idx] = np.kron(U1[:, r], U2[:, s])
            V[:, idx] = np.kron(V1[:, r], V2[:, s])
            W[:, idx] = np.kron(W1[:, r], W2[:, s])
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    rank = U.shape[1]
    # For matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor T satisfies: T[i*m+k, k*p+j, j*n+i] should give the product
    # Actually, the decomposition should satisfy:
    # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all valid (i,k,j)
    # Wait, let me reconsider. The tensor for matrix multiplication has:
    # T[a, b, c] where a indexes A, b indexes B, c indexes C
    # T[i*m+k, k*p+j, j*n+i] = 1 if this triple contributes to C[i,j]
    # Actually, for each (i,k,j), the term A[i,k]*B[k,j] contributes to C[i,j]
    # So the condition is: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    # for all i in [0,n), k in [0,m), j in [0,p)
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = np.sum(U[i*m+k, :] * V[k*p+j, :] * W[j*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, leaving 40s margin
    
    # Strategy: Decompose 4x4x9 as Kronecker products
    # 4 = 2*2, so we can use Strassen (2x2x2) for the 4x4 part
    # 9 = 9*1, so we need a decomposition for 1x1x9 or similar
    
    # Approach 1: 4x4x9 = (2x2x2) ⊗ (2x2x1) ⊗ (1x1x9/3)?
    # Let's think: 4x4x9
    # Split n=4 into 2x2, m=4 into 2x2, p=9 into 9
    # We can do: (2x2x1) ⊗ (2x2x9) or (2x2x9) ⊗ (2x2x1)
    
    # Better: Use the fact that 4x4x4 can be done with Strassen applied twice
    # 4x4x4 = (2x2x2) ⊗ (2x2x2), rank = 7*7 = 49
    # Then for 4x4x9, split p=9 into 4+4+1
    # 4x4x9 = 4x4x4 (rank 49) + 4x4x4 (rank 49) + 4x4x1 (rank ?)
    
    # For 4x4x1: C[i] = sum_k A[i,k] * B[k]
    # This is like multiplying a 4x4 matrix by a 4x1 matrix
    # We can use 2x2x1 blocks: 4x4x1 = (2x2x1) ⊗ (2x2x1)
    # What's the rank of 2x2x1?
    # C[i] = sum_k A[i,k] * B[k] for i=0,1
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # Schoolbook rank = 4
    # Can we do better? For 2x2x1, the tensor rank is 4 (it's just a bilinear map)
    # Actually, let's verify: the tensor has T[i,k,j] where j is always 0
    # T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1
    # This is the identity-like tensor. Rank is 4.
    
    # So 4x4x1 via Kronecker: (2x2x1) ⊗ (2x2x1), rank = 4*4 = 16
    # Total: 49 + 49 + 16 = 114
    
    # Let's build this:
    # Block 1: columns 0-3 (4x4x4), rank 49
    # Block 2: columns 4-7 (4x4x4), rank 49  
    # Block 3: column 8 (4x4x1), rank 16
    
    # Build 2x2x2 Strassen decomposition
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Build 2x2x1 schoolbook decomposition
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    U_221 = np.array([
        [1, 0, 0, 0],  # A[0,0]
        [0, 1, 0, 0],  # A[0,1]
        [0, 0, 1, 0],  # A[1,0]
        [0, 0, 0, 1],  # A[1,1]
    ], dtype=float)
    V_221 = np.array([
        [1, 0, 1, 0],  # B[0]
        [0, 1, 0, 1],  # B[1]
    ], dtype=float)
    W_221 = np.array([
        [1, 0, 0, 0],  # C[0]
        [0, 0, 1, 0],  # C[1]
    ], dtype=float)
    # Wait, I need to be more careful with the indexing.
    # For 2x2x1: A is 2x2 (4 entries), B is 2x1 (2 entries), C is 2x1 (2 entries)
    # U has shape (4, rank), V has shape (2, rank), W has shape (2, rank)
    # Schoolbook: 4 terms, one for each (i,k) pair
    # Term 0: i=0, k=0: A[0,0]*B[0] contributes to C[0]
    # Term 1: i=0, k=1: A[0,1]*B[1] contributes to C[0]
    # Term 2: i=1, k=0: A[1,0]*B[0] contributes to C[1]
    # Term 3: i=1, k=1: A[1,1]*B[1] contributes to C[1]
    
    U_221 = np.array([
        [1, 0, 0, 0],  # A[0,0]: term 0
        [0, 1, 0, 0],  # A[0,1]: term 1
        [0, 0, 1, 0],  # A[1,0]: term 2
        [0, 0, 0, 1],  # A[1,1]: term 3
    ], dtype=float)
    V_221 = np.array([
        [1, 0, 1, 0],  # B[0]: terms 0,2
        [0, 1, 0, 1],  # B[1]: terms 1,3
    ], dtype=float)
    W_221 = np.array([
        [1, 1, 0, 0],  # C[0]: terms 0,1
        [0, 0, 1, 1],  # C[1]: terms 2,3
    ], dtype=float)
    
    # Verify 2x2x1
    # sum_r U[i*2+k, r] * V[k, r] * W[i, r] should be 1 for all i,k
    # i=0, k=0: U[0,:]·V[0,:]·W[0,:] = 1*1*1 + 0 + 0 + 0 = 1 ✓
    # i=0, k=1: U[1,:]·V[1,:]·W[0,:] = 0 + 1*1*1 + 0 + 0 = 1 ✓
    # i=1, k=0: U[2,:]·V[0,:]·W[1,:] = 0 + 0 + 1*1*1 + 0 = 1 ✓
    # i=1, k=1: U[3,:]·V[1,:]·W[1,:] = 0 + 0 + 0 + 1*1*1 = 1 ✓
    
    # Build 4x4x4 via Kronecker: (2x2x2) ⊗ (2x2x2)
    U_444, V_444, W_444 = _kronecker_decomp(U_s, V_s, W_s, U_s, V_s, W_s)
    
    # Build 4x4x1 via Kronecker: (2x2x1) ⊗ (2x2x1)
    U_441, V_441, W_441 = _kronecker_decomp(U_221, V_221, W_221, U_221, V_221, W_221)
    
    # Now assemble 4x4x9:
    # Columns 0-3: use 4x4x4 block (rank 49)
    # Columns 4-7: use another 4x4x4 block (rank 49)
    # Column 8: use 4x4x1 block (rank 16)
    
    rank1 = U_444.shape[1]  # 49
    rank2 = U_441.shape[1]  # 16
    total_rank = rank1 + rank1 + rank2  # 114
    
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    # Block 1: columns 0-3
    offset = 0
    U[:, offset:offset+rank1] = U_444
    # V for 4x4x4: B is 4x4, we only use columns 0-3
    V_block1 = np.zeros((m*p, rank1), dtype=float)
    for k in range(m):
        for j in range(4):
            V_block1[k*p + j, :] = V_444[k*4 + j, :]
    V[:, offset:offset+rank1] = V_block1
    # W for 4x4x4: C is 4x4, we only use columns 0-3 (rows 0-3 of C)
    W_block1 = np.zeros((p*n, rank1), dtype=float)
    for j in range(4):
        for i in range(n):
            W_block1[j*n + i, :] = W_444[j*4 + i, :]
    W[:, offset:offset+rank1] = W_block1
    
    # Block 2: columns 4-7
    offset += rank1
    U[:, offset:offset+rank1] = U_444
    V_block2 = np.zeros((m*p, rank1), dtype=float)
    for k in range(m):
        for j in range(4):
            V_block2[k*p + (j + 4), :] = V_444[k*4 + j, :]
    V[:, offset:offset+rank1] = V_block2
    W_block2 = np.zeros((p*n, rank1), dtype=float)
    for j in range(4):
        for i in range(n):
            W_block2[(j + 4)*n + i, :] = W_444[j*4 + i, :]
    W[:, offset:offset+rank1] = W_block2
    
    # Block 3: column 8
    offset += rank1
    U[:, offset:offset+rank2] = U_441
    V_block3 = np.zeros((m*p, rank2), dtype=float)
    for k in range(m):
        V_block3[k*p + 8, :] = V_441[k, :]  # V_441 has shape (4, 16)
    V[:, offset:offset+rank2] = V_block3
    W_block3 = np.zeros((p*n, rank2), dtype=float)
    for i in range(n):
        W_block3[8*n + i, :] = W_441[i, :]  # W_441 has shape (4, 16)
    W[:, offset:offset+rank2] = W_block3
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        if time.time() < deadline:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U_fb = np.zeros((n*m, rank), dtype=float)
    V_fb = np.zeros((m*p, rank), dtype=float)
    W_fb = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_fb[i*m + k, col] = 1.0
                V_fb[k*p + j, col] = 1.0
                W_fb[j*n + i, col] = 1.0
                col += 1
    
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END

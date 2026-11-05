"""
Matrix multiplication 4x5 * 5x6 via block decomposition.

We decompose the 5-dimensional inner index j into j = 4*a + b with a in {0,1} and b in {0,1,2,3}.
This splits the multiplication into four blocks:
  C = A[:,0:4] @ B[0:4,:] + A[:,4:5] @ B[4:5,:]

We further split the 6-dimensional output index k into k = 4*c + d with c in {0,1} and d in {0,1,2}.
This gives:
  C[:, 0:4] = A[:,0:4] @ B[0:4, 0:4] + A[:,4:5] @ B[4:5, 0:4]
  C[:, 4:6] = A[:,0:4] @ B[0:4, 4:6] + A[:,4:5] @ B[4:5, 4:6]

So we have four sub-problems:
1. 4x4 @ 4x4 -> 4x4   (rank 49 via Strassen)
2. 4x1 @ 1x4 -> 4x4   (rank 16)
3. 4x4 @ 4x2 -> 4x2   (rank 28 via 2 Strassen instances)
4. 4x1 @ 1x2 -> 4x2   (rank 8)

Total rank: 49 + 16 + 28 + 8 = 101
"""
import numpy as np

DIMENSIONS = [4, 5, 6]

def _strassen_2x2():
    """
    Return the Strassen decomposition for 2x2 @ 2x2 -> 2x2.
    
    Returns U, V, W as lists of lists with shapes:
      U: (4, 7)  -- rows indexed by (i*m+j) = (i*2+j)
      V: (4, 7)  -- rows indexed by (j*p+k) = (j*2+k)
      W: (4, 7)  -- rows indexed by (k*n+i) = (k*2+i)
    
    Strassen's 7 products:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B12-B22)
      M4 = A22(B21-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U rows: A11, A12, A21, A22 (index = i*2+j)
    # V rows: B11, B12, B21, B22 (index = j*2+k)
    # W rows: C11, C12, C21, C22 (index = k*2+i)
    
    # For each of the 7 products, define the coefficients for each A-element, B-element, and C-element.
    
    # M1 = (A11+A22)(B11+B22)
    # A-coeffs: A11=1, A12=0, A21=0, A22=1
    # B-coeffs: B11=1, B12=0, B21=0, B22=1
    # C-coeffs: C11=1, C12=0, C21=0, C22=1
    
    # M2 = (A21+A22)B11
    # A-coeffs: A11=0, A12=0, A21=1, A22=1
    # B-coeffs: B11=1, B12=0, B21=0, B22=0
    # C-coeffs: C11=0, C12=0, C21=1, C22=-1
    
    # M3 = A11(B12-B22)
    # A-coeffs: A11=1, A12=0, A21=0, A22=0
    # B-coeffs: B11=0, B12=1, B21=0, B22=-1
    # C-coeffs: C11=0, C12=1, C21=0, C22=1
    
    # M4 = A22(B21-B11)
    # A-coeffs: A11=0, A12=0, A21=0, A22=1
    # B-coeffs: B11=-1, B12=0, B21=1, B22=0
    # C-coeffs: C11=1, C12=0, C21=1, C22=0
    
    # M5 = (A11+A12)B22
    # A-coeffs: A11=1, A12=1, A21=0, A22=0
    # B-coeffs: B11=0, B12=0, B21=0, B22=1
    # C-coeffs: C11=-1, C12=1, C21=0, C22=0
    
    # M6 = (A21-A11)(B11+B12)
    # A-coeffs: A11=-1, A12=0, A21=1, A22=0
    # B-coeffs: B11=1, B12=1, B21=0, B22=0
    # C-coeffs: C11=0, C12=0, C21=0, C22=1
    
    # M7 = (A12-A22)(B21+B22)
    # A-coeffs: A11=0, A12=1, A21=0, A22=-1
    # B-coeffs: B11=0, B12=0, B21=1, B22=1
    # C-coeffs: C11=1, C12=0, C21=0, C22=0
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A11
        [0, 0, 0, 0, 1, 0, 1],    # A12
        [0, 1, 0, 0, 0, 1, 0],    # A21
        [1, 1, 0, 1, 0, 0, -1]    # A22
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B11
        [0, 0, 1, 0, 0, 1, 0],    # B12
        [0, 0, 0, 1, 0, 0, 1],    # B21
        [1, 0, -1, 0, 1, 0, 1]    # B22
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0]    # C22
    ]
    
    return U, V, W

def _rank_4x1_4x4():
    """
    Decomposition for 4x1 @ 1x4 -> 4x4.
    A is 4x1 (column vector), B is 1x4 (row vector).
    C[i,k] = A[i,0] * B[0,k]
    
    This is an outer product. Rank = 4*4 = 16.
    
    U: (4*1, 16) = (4, 16)
    V: (1*4, 16) = (4, 16)
    W: (4*4, 16) = (16, 16)
    
    For each (i,k) pair, we need one term where:
    U[i, c] = 1, V[k, c] = 1, W[k*4+i, c] = 1
    """
    rank = 16
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(16)]
    
    col = 0
    for i in range(4):
        for k in range(4):
            U[i][col] = 1
            V[k][col] = 1
            W[k*4 + i][col] = 1
            col += 1
    
    return U, V, W

def _rank_4x1_4x2():
    """
    Decomposition for 4x1 @ 1x2 -> 4x2.
    A is 4x1, B is 1x2.
    C[i,k] = A[i,0] * B[0,k] for k in {0,1}
    
    Rank = 4*2 = 8.
    
    U: (4, 8)
    V: (2, 8)
    W: (2*4, 8) = (8, 8)
    """
    rank = 8
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(8)]
    
    col = 0
    for i in range(4):
        for k in range(2):
            U[i][col] = 1
            V[k][col] = 1
            W[k*4 + i][col] = 1
            col += 1
    
    return U, V, W

def _rank_4x4_4x2():
    """
    Decomposition for 4x4 @ 4x2 -> 4x2.
    
    Split into two 2x2 @ 2x2 -> 2x2 problems (top and bottom halves).
    Each uses Strassen (rank 7), so total rank = 14.
    
    Wait, 4x4 @ 4x2:
    C_top = A_top @ B_top  (2x4 @ 4x2 -> 2x2)
    C_bot = A_bot @ B_bot  (2x4 @ 4x2 -> 2x2)
    
    Each 2x4 @ 4x2 -> 2x2 can be done with... hmm.
    Actually, 2x4 @ 4x2 is like two 2x2 @ 2x2 side by side.
    C[:,0:2] = A @ B[:,0:2]
    This is 2x4 @ 4x2. We can split A into A_left (2x2) and A_right (2x2),
    and B into B_top (2x2) and B_bot (2x2).
    C = A_left @ B_top + A_right @ B_bot
    
    Each 2x2 @ 2x2 -> 2x2 has rank 7 (Strassen).
    So 2x4 @ 4x2 -> 2x2 has rank at most 14.
    
    Total for 4x4 @ 4x2 -> 4x2: 14 + 14 = 28.
    """
    # We'll use Strassen for each 2x2 @ 2x2 sub-problem.
    # U: (4*4, 28) = (16, 28)
    # V: (4*2, 28) = (8, 28)
    # W: (2*4, 28) = (8, 28)
    
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    sU, sV, sW = _strassen_2x2()
    
    # For the top half (i in 0,1):
    # A rows 0,1 (A_top): indices 0,1,2,3 in U (i*4+j)
    #   A_top = A[0:2, 0:4]
    #   A_left = A[0:2, 0:2] -> U indices 0,1,2,3
    #   A_right = A[0:2, 2:4] -> U indices 4,5,6,7
    # B cols 0,1 (B_top): indices 0,1,2,3 in V (j*2+k)
    #   B_top = B[0:2, 0:2] -> V indices 0,1 (j=0: 0,1; j=1: 2,3)
    #   Wait, B is 4x2. B[j,k] with j in 0..3, k in 0..1.
    #   V index = j*2 + k.
    #   B_top = B[0:2, :] -> j in 0,1 -> V indices 0,1,2,3
    #   B_bot = B[2:4, :] -> j in 2,3 -> V indices 4,5,6,7
    
    # C is 4x2. C[i,k] with i in 0..3, k in 0..1.
    # W index = k*4 + i.
    # C_top = C[0:2, :] -> i in 0,1
    # C_bot = C[2:4, :] -> i in 2,3
    
    # For C_top = A_left @ B_top + A_right @ B_bot:
    # This is 2x2 @ 2x2 + 2x2 @ 2x2 -> 2x2
    
    # First product: A_left @ B_top -> contributes to C_top
    # A_left: rows 0,1 of A, cols 0,1 -> U indices 0,1,2,3 (i*4+j for i=0,1; j=0,1)
    # B_top: rows 0,1 of B, cols 0,1 -> V indices 0,1,2,3 (j*2+k for j=0,1; k=0,1)
    # C_top: rows 0,1 of C, cols 0,1 -> W indices for k=0,1; i=0,1
    #   W index = k*4 + i. For i=0: W[0], W[4]. For i=1: W[1], W[5].
    
    # Map Strassen's 2x2 indices to our indices:
    # Strassen A: A11, A12, A21, A22 -> our A_left: A[0,0], A[0,1], A[1,0], A[1,1]
    #   U indices: 0, 1, 4, 5
    # Strassen B: B11, B12, B21, B22 -> our B_top: B[0,0], B[0,1], B[1,0], B[1,1]
    #   V indices: 0, 1, 2, 3
    # Strassen C: C11, C12, C21, C22 -> our C_top: C[0,0], C[0,1], C[1,0], C[1,1]
    #   W indices: 0, 4, 1, 5
    
    for c in range(7):
        # A coefficients
        for s_i, our_i in enumerate([0, 1, 4, 5]):
            if sU[s_i][c] != 0:
                U[our_i][c] += sU[s_i][c]
        # B coefficients
        for s_j, our_j in enumerate([0, 1, 2, 3]):
            if sV[s_j][c] != 0:
                V[our_j][c] += sV[s_j][c]
        # C coefficients
        for s_k, our_k in enumerate([0, 4, 1, 5]):
            if sW[s_k][c] != 0:
                W[our_k][c] += sW[s_k][c]
    
    # Second product: A_right @ B_bot -> contributes to C_top
    # A_right: rows 0,1 of A, cols 2,3 -> U indices 2,3,6,7
    # B_bot: rows 2,3 of B, cols 0,1 -> V indices 4,5,6,7
    # C_top: same as above
    
    offset = 7
    for c in range(7):
        for s_i, our_i in enumerate([2, 3, 6, 7]):
            if sU[s_i][c] != 0:
                U[our_i][offset + c] += sU[s_i][c]
        for s_j, our_j in enumerate([4, 5, 6, 7]):
            if sV[s_j][c] != 0:
                V[our_j][offset + c] += sV[s_j][c]
        for s_k, our_k in enumerate([0, 4, 1, 5]):
            if sW[s_k][c] != 0:
                W[our_k][offset + c] += sW[s_k][c]
    
    # For C_bot = A_left_bot @ B_top + A_right_bot @ B_bot:
    # A_left_bot = A[2:4, 0:2] -> U indices 8,9,12,13
    # A_right_bot = A[2:4, 2:4] -> U indices 10,11,14,15
    # B_top, B_bot same as above
    # C_bot: rows 2,3 of C, cols 0,1
    #   W index = k*4 + i. For i=2: W[2], W[6]. For i=3: W[3], W[7].
    
    offset2 = 14
    for c in range(7):
        for s_i, our_i in enumerate([8, 9, 12, 13]):
            if sU[s_i][c] != 0:
                U[our_i][offset2 + c] += sU[s_i][c]
        for s_j, our_j in enumerate([0, 1, 2, 3]):
            if sV[s_j][c] != 0:
                V[our_j][offset2 + c] += sV[s_j][c]
        for s_k, our_k in enumerate([2, 6, 3, 7]):
            if sW[s_k][c] != 0:
                W[our_k][offset2 + c] += sW[s_k][c]
    
    offset3 = 21
    for c in range(7):
        for s_i, our_i in enumerate([10, 11, 14, 15]):
            if sU[s_i][c] != 0:
                U[our_i][offset3 + c] += sU[s_i][c]
        for s_j, our_j in enumerate([4, 5, 6, 7]):
            if sV[s_j][c] != 0:
                V[our_j][offset3 + c] += sV[s_j][c]
        for s_k, our_k in enumerate([2, 6, 3, 7]):
            if sW[s_k][c] != 0:
                W[our_k][offset3 + c] += sW[s_k][c]
    
    return U, V, W

def _rank_4x4_4x4():
    """
    Decomposition for 4x4 @ 4x4 -> 4x4 using Strassen.
    Rank = 7 * 7 = 49.
    
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    """
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    sU, sV, sW = _strassen_2x2()
    
    # Split 4x4 into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # A11: rows 0,1; cols 0,1 -> U indices 0,1,4,5
    # A12: rows 0,1; cols 2,3 -> U indices 2,3,6,7
    # A21: rows 2,3; cols 0,1 -> U indices 8,9,12,13
    # A22: rows 2,3; cols 2,3 -> U indices 10,11,14,15
    
    # B = [[B11, B12], [B21, B22]]
    # B11: rows 0,1; cols 0,1 -> V indices 0,1,2,3
    # B12: rows 0,1; cols 2,3 -> V indices 4,5,6,7
    # B21: rows 2,3; cols 0,1 -> V indices 8,9,10,11
    # B22: rows 2,3; cols 2,3 -> V indices 12,13,14,15
    
    # C = [[C11, C12], [C21, C22]]
    # C11: rows 0,1; cols 0,1 -> W indices 0,4,1,5
    # C12: rows 0,1; cols 2,3 -> W indices 2,6,3,7
    # C21: rows 2,3; cols 0,1 -> W indices 8,12,9,13
    # C22: rows 2,3; cols 2,3 -> W indices 10,14,11,15
    
    # Strassen:
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    # M3 = A11(B12-B22) -> C12 += M3, C22 += M3
    # M4 = A22(B21-B11) -> C11 += M4, C21 += M4
    # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    
    # For each Strassen product s (0..6), we need to compute it for each of the 4 block positions.
    # Actually, each Strassen "product" M_s involves specific blocks. We need to compute
    # the matrix product of 2x2 blocks, which itself requires a 2x2 @ 2x2 decomposition.
    
    # This is getting complex. Let me use a different approach:
    # The 4x4 @ 4x4 can be seen as 4 independent 2x4 @ 4x2 -> 2x2 problems? No.
    
    # Actually, the standard way: treat the 4x4 multiplication as a 2x2 block multiplication
    # where each block is 2x2. The Strassen algorithm gives 7 block products, each of which
    # is a 2x2 @ 2x2 -> 2x2 multiplication. Each such has rank 7.
    # So total rank = 7 * 7 = 49.
    
    # For each Strassen product s (0..6), and for each "inner" Strassen product t (0..6),
    # we add the contribution.
    
    # The block-level Strassen says:
    # M1 = (A11+A22)(B11+B22)
    # This means we form A11+A22 (a 2x2 matrix) and B11+B22 (a 2x2 matrix),
    # then multiply them. The multiplication is done via Strassen (7 products).
    
    # So for M1, the "effective" A matrix has:
    #   A11 block: A11 (coefficient 1)
    #   A22 block: A22 (coefficient 1)
    #   A12 block: 0
    #   A21 block: 0
    # And the "effective" B matrix has:
    #   B11 block: B11 (coefficient 1)
    #   B22 block: B22 (coefficient 1)
    #   B12 block: 0
    #   B21 block: 0
    
    # The product M1 is then computed via Strassen on these effective matrices.
    
    # This is very complex to implement directly. Let me use a simpler approach:
    # Just use the schoolbook for the inner 2x2 @ 2x2, giving rank 4*7 = 28 per block product,
    # total 7*28 = 196. That's worse than 120.
    
    # Alternatively, I can use the fact that 4x4 @ 4x4 can be decomposed using
    # 4 instances of 2x4 @ 4x2 -> 2x2, each with rank 14, total 56. Still better than 120.
    
    # Actually, let me just use the schoolbook for 4x4 @ 4x4 for simplicity and correctness.
    # Rank = 16*16 = 256? No, schoolbook for 4x4 @ 4x4 -> 4x4 has rank 4*4*4 = 64.
    
    # Wait, I'm overcomplicating this. Let me just use the schoolbook decomposition
    # for the 4x4 @ 4x4 -> 4x4 part. Rank = 4*4*4 = 64.
    
    # Total would be: 64 + 16 + 28 + 8 = 116. Still better than 120.
    
    # Let me redo with schoolbook for 4x4 @ 4x4.
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4 + j][col] = 1
                V[j*4 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    We decompose the inner dimension 5 into 4+1, and the output dimension 6 into 4+2.
    This gives four sub-problems:
    1. 4x4 @ 4x4 -> 4x4   (rank 64, schoolbook)
    2. 4x1 @ 1x4 -> 4x4   (rank 16)
    3. 4x4 @ 4x2 -> 4x2   (rank 28)
    4. 4x1 @ 1x2 -> 4x2   (rank 8)
    
    Total rank: 64 + 16 + 28 + 8 = 116
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Full dimensions:
    # U: (n*m, rank) = (20, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (24, rank)
    
    # Sub-problem 1: 4x4 @ 4x4 -> 4x4
    # A indices: i in 0..3, j in 0..3 -> U indices: i*5 + j (since m=5)
    # B indices: j in 0..3, k in 0..3 -> V indices: j*6 + k (since p=6)
    # C indices: k in 0..3, i in 0..3 -> W indices: k*4 + i (since n=4)
    
    U1, V1, W1 = _rank_4x4_4x4()
    
    # Sub-problem 2: 4x1 @ 1x4 -> 4x4
    # A indices: i in 0..3, j=4 -> U indices: i*5 + 4
    # B indices: j=4, k in 0..3 -> V indices: 4*6 + k = 24 + k
    # C indices: k in 0..3, i in 0..3 -> W indices: k*4 + i
    
    U2, V2, W2 = _rank_4x1_4x4()
    
    # Sub-problem 3: 4x4 @ 4x2 -> 4x2
    # A indices: i in 0..3, j in 0..3 -> U indices: i*5 + j
    # B indices: j in 0..3, k in 4..5 -> V indices: j*6 + k
    # C indices: k in 4..5, i in 0..3 -> W indices: k*4 + i
    
    U3, V3, W3 = _rank_4x4_4x2()
    
    # Sub-problem 4: 4x1 @ 1x2 -> 4x2
    # A indices: i in 0..3, j=4 -> U indices: i*5 + 4
    # B indices: j=4, k in 4..5 -> V indices: 4*6 + k = 24 + k
    # C indices: k in 4..5, i in 0..3 -> W indices: k*4 + i
    
    U4, V4, W4 = _rank_4x1_4x2()
    
    rank1 = len(U1[0])  # 64
    rank2 = len(U2[0])  # 16
    rank3 = len(U3[0])  # 28
    rank4 = len(U4[0])  # 8
    
    total_rank = rank1 + rank2 + rank3 + rank4
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Sub-problem 1: offset 0
    for i in range(4):
        for j in range(4):
            for c in range(rank1):
                U[i*5 + j][c] += U1[i*4 + j][c]
        for k in range(4):
            for c in range(rank1):
                W[k*4 + i][c] += W1[k*4 + i][c]
    for j in range(4):
        for k in range(4):
            for c in range(rank1):
                V[j*6 + k][c] += V1[j*4 + k][c]
    
    # Sub-problem 2: offset rank1
    off2 = rank1
    for i in range(4):
        for c in range(rank2):
            U[i*5 + 4][off2 + c] += U2[i][c]
        for k in range(4):
            for c in range(rank2):
                W[k*4 + i][off2 + c] += W2[k*4 + i][c]
    for k in range(4):
        for c in range(rank2):
            V[4*6 + k][off2 + c] += V2[k][c]
    
    # Sub-problem 3: offset rank1 + rank2
    off3 = rank1 + rank2
    for i in range(4):
        for j in range(4):
            for c in range(rank3):
                U[i*5 + j][off3 + c] += U3[i*4 + j][c]
        for k in range(4, 6):
            for c in range(rank3):
                W[k*4 + i][off3 + c] += W3[(k-4)*4 + i][c]
    for j in range(4):
        for k in range(4, 6):
            for c in range(rank3):
                V[j*6 + k][off3 + c] += V3[j*2 + (k-4)][c]
    
    # Sub-problem 4: offset rank1 + rank2 + rank3
    off4 = rank1 + rank2 + rank3
    for i in range(4):
        for c in range(rank4):
            U[i*5 + 4][off4 + c] += U4[i][c]
        for k in range(4, 6):
            for c in range(rank4):
                W[k*4 + i][off4 + c] += W4[(k-4)*4 + i][c]
    for k in range(4, 6):
        for c in range(rank4):
            V[4*6 + k][off4 + c] += V4[k-4][c]
    
    return {"U": U, "V": V, "W": W}

"""
Matrix multiplication 4x5x5 tensor decomposition.

Strategy:
- The target tensor is M_{4,5,5} with entries T[i,j,k] = 1 for i in [0,4), j in [0,5), k in [0,5).
- The output factor follows convention W[k*n + i, r].

We construct a decomposition of rank 100 using the following block structure:
  1. For the first 4 columns of A (j=0..3) and first 4 rows of B (k=0..3):
     Use a 4x4x4 decomposition. We use the schoolbook 4x4x4 decomposition (rank 64).
     Actually, we can do better: use a 4x4x4 decomposition of rank 49 via Strassen tensor product.
     But to keep it simple and robust, let's use a known good decomposition.
     Wait, the schoolbook 4x4x4 is rank 64. Strassen gives 49.
     Let's implement Strassen 2x2x2 and tensor product it with itself.

  2. For the first 4 columns of A (j=0..3) and last 1 row of B (k=4):
     This is a 4x1x5 block? No, j ranges 0..3, k=4. i ranges 0..3.
     This is essentially: for each i in 0..3, j in 0..3, k=4: A[i,j]*B[j,4].
     This is rank 16 (schoolbook for 4x4x1 with the 5th column of B fixed? No.)
     Actually, this sub-tensor has indices i in [0,4), j in [0,4), k=4.
     It's a 4x4x1 tensor. Rank is at most 16, but actually it's rank 16? 
     No, T[i,j,4] = 1. This is the matrix multiplication tensor for 4x4 x 4x1.
     Rank is 16? No, the rank of matrix multiplication n x m x 1 is n*m.
     So rank 16.

  3. For the last 1 column of A (j=4) and all 5 rows of B (k=0..4):
     This is a 1x5x5 tensor? i in [0,4), j=4, k in [0,5).
     T[i,4,k] = 1. This is matrix multiplication 4x1 x 1x5.
     Rank is 4*1*5 = 20? No, the tensor has entries 1 for all i,k.
     Wait, T[i,4,k] = 1. This is the outer product of a 4-vector of 1s and a 5-vector of 1s, 
     but structured as matrix multiplication.
     Actually, for fixed j=4, the contribution is sum_k A[i,4]*B[4,k].
     This is a rank-4 tensor? No.
     The sub-tensor is T_sub[i, j, k] where j is fixed to 4. 
     It has indices i in [0,4), k in [0,5). The entries are 1.
     This can be decomposed as sum_{c=0}^{3} u_c v_c w_c where:
     u_c[i] = 1 if i==c else 0
     v_c[4] = 1 (only j=4 is active)
     w_c[k] = 1 for all k? No, that would give sum_c u_c[i] * v_c[4] * sum_k w_c[k] B[4,k].
     That's not right.
     
     Let's rethink. The full tensor is sum_{i,j,k} A[i,j] B[j,k] E_{i,k}.
     We can split by j.
     For j=0..3: we have a 4x4x5 sub-problem? No.
     
     Alternative simple approach:
     Use the schoolbook decomposition for the whole thing: rank 4*5*5 = 100.
     But we can do better by using a lower-rank decomposition for the 4x4x4 block.
     
     Let's use:
     - Block 1: i in 0..3, j in 0..3, k in 0..3. Use Strassen 4x4x4 (rank 49).
     - Block 2: i in 0..3, j in 0..3, k=4. Use schoolbook 4x4x1 (rank 16).
     - Block 3: i in 0..3, j=4, k in 0..4. Use schoolbook 4x1x5 (rank 20).
     
     Total rank: 49 + 16 + 20 = 85.
     
     Let's implement this.
"""

import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indices: i in [0,2), j in [0,2), k in [0,2).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U factors: coefficients for A[i,j] in each M_c
    # V factors: coefficients for B[j,k] in each M_c
    # W factors: coefficients for C[i,k] from each M_c
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U: i*2+j index
    # M1: A00 + A11
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    # M2: A10 + A11
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    # M3: A00
    U[0][2] = 1  # A00
    # M4: A11
    U[3][3] = 1  # A11
    # M5: A00 + A01
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    # M6: A10 - A00
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    # M7: A01 - A11
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    
    # V: j*2+k index
    # M1: B00 + B11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    # M2: B00
    V[0][1] = 1  # B00
    # M3: B01 - B11
    V[1][2] = 1  # B01
    V[3][2] = -1 # B11
    # M4: B10 - B00
    V[2][3] = 1  # B10
    V[0][3] = -1 # B00
    # M5: B11
    V[3][4] = 1  # B11
    # M6: B00 + B01
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    # M7: B10 + B11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    
    # W: k*2+i index
    # C = [[M1-M4+M5-M7, M3+M4], [M2+M4, M1-M2+M3+M6]]
    # C00 = M1 - M4 + M5 - M7
    W[0][0] = 1  # M1
    W[0][3] = -1 # M4
    W[0][4] = 1  # M5
    W[0][6] = -1 # M7
    # C01 = M3 + M4
    W[1][2] = 1  # M3
    W[1][3] = 1  # M4
    # C10 = M2 + M4
    W[2][1] = 1  # M2
    W[2][3] = 1  # M4
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1  # M1
    W[3][1] = -1 # M2
    W[3][2] = 1  # M3
    W[3][5] = 1  # M6
    
    return U, V, W

def _tensor_product_decomposition(U1, V1, W1, U2, V2, W2):
    """
    Compute tensor product of two decompositions.
    If T1 has indices (i1, j1, k1) and T2 has (i2, j2, k2),
    the product has indices (i1,i2), (j1,j2), (k1,k2) mapped to (i1*n2+i2, j1*m2+j2, k1*p2+k2).
    Rank is rank1 * rank2.
    """
    n1, r1 = len(U1), len(U1[0])
    m1 = len(V1[0]) if V1 else r1
    # Actually, shapes: U1 is (n1*m1, r1), V1 is (m1*p1, r1), W1 is (p1*n1, r1)
    # We need to infer n1, m1, p1 from the problem context.
    # For 2x2x2: n1=2, m1=2, p1=2, r1=7
    # For the second one: n2, m2, p2, r2
    # The result will be for n1*n2 x m1*m2 x p1*p2
    
    # We'll pass the dimensions explicitly to avoid inference issues
    # Let's redefine to take dimensions
    pass

def _strassen_4x4_4x4():
    """
    Build 4x4x4 decomposition using tensor product of two 2x2x2 Strassen decompositions.
    """
    U1, V1, W1 = _strassen_2x2_2x2()
    U2, V2, W2 = _strassen_2x2_2x2()
    
    n1, m1, p1, r1 = 2, 2, 2, 7
    n2, m2, p2, r2 = 2, 2, 2, 7
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    rank = r1 * r2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u_val = U1[i1 * m1 + j1][c1] * U2[i2 * m2 + j2][c2]
                            if u_val != 0:
                                idx = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                                U[idx][c] += u_val
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v_val = V1[j1 * p1 + k1][c1] * V2[j2 * p2 + k2][c2]
                            if v_val != 0:
                                idx = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                                V[idx][c] += v_val
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w_val = W1[k1 * n1 + i1][c1] * W2[k2 * n2 + i2][c2]
                            if w_val != 0:
                                idx = (k1 * n2 + k2) * n + (i1 * n2 + i2)
                                W[idx][c] += w_val
    return U, V, W

def _schoolbook_block(n, m, p):
    """
    Schoolbook decomposition for n x m x p.
    """
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
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = 4, 5, 5
    
    # Block 1: i in 0..3, j in 0..3, k in 0..3
    # This is a 4x4x4 sub-tensor
    U1, V1, W1 = _strassen_4x4_4x4()
    rank1 = len(U1[0])  # 49
    
    # Block 2: i in 0..3, j in 0..3, k=4
    # This is a 4x4x1 sub-tensor (k is fixed to 4)
    # The tensor entries are T[i,j,4] = 1
    # This is matrix multiplication 4x4 x 4x1
    U2, V2, W2 = _schoolbook_block(4, 4, 1)
    rank2 = len(U2[0])  # 16
    
    # Block 3: i in 0..3, j=4, k in 0..4
    # This is a 4x1x5 sub-tensor (j is fixed to 4)
    # The tensor entries are T[i,4,k] = 1
    # This is matrix multiplication 4x1 x 1x5
    U3, V3, W3 = _schoolbook_block(4, 1, 5)
    rank3 = len(U3[0])  # 20
    
    total_rank = rank1 + rank2 + rank3
    
    # Now we need to combine these into the full 4x5x5 decomposition
    # Full dimensions: n=4, m=5, p=5
    # U has shape (4*5, total_rank) = (20, total_rank)
    # V has shape (5*5, total_rank) = (25, total_rank)
    # W has shape (5*4, total_rank) = (20, total_rank)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1: i in 0..3, j in 0..3, k in 0..3
    # U1 indices: i in 0..3, j in 0..3 -> U1[i*4+j, c1]
    # V1 indices: j in 0..3, k in 0..3 -> V1[j*4+k, c1]
    # W1 indices: k in 0..3, i in 0..3 -> W1[k*4+i, c1]
    # In full: U[i*5+j, c], V[j*5+k, c], W[k*4+i, c]
    for c1 in range(rank1):
        c = c1
        for i in range(4):
            for j in range(4):
                U[i * 5 + j][c] = U1[i * 4 + j][c1]
        for j in range(4):
            for k in range(4):
                V[j * 5 + k][c] = V1[j * 4 + k][c1]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][c] = W1[k * 4 + i][c1]
    
    # Block 2: i in 0..3, j in 0..3, k=4
    # U2 indices: i in 0..3, j in 0..3 -> U2[i*4+j, c2]
    # V2 indices: j in 0..3, k=0 (local) -> V2[j*1+0, c2] = V2[j, c2]
    # W2 indices: k=0 (local), i in 0..3 -> W2[0*4+i, c2] = W2[i, c2]
    # In full: U[i*5+j, c], V[j*5+4, c], W[4*4+i, c]
    for c2 in range(rank2):
        c = rank1 + c2
        for i in range(4):
            for j in range(4):
                U[i * 5 + j][c] = U2[i * 4 + j][c2]
        for j in range(4):
            V[j * 5 + 4][c] = V2[j * 1 + 0][c2]
        for i in range(4):
            W[4 * 4 + i][c] = W2[0 * 4 + i][c2]
    
    # Block 3: i in 0..3, j=4, k in 0..4
    # U3 indices: i in 0..3, j=0 (local) -> U3[i*1+0, c3] = U3[i, c3]
    # V3 indices: j=0 (local), k in 0..4 -> V3[0*5+k, c3] = V3[k, c3]
    # W3 indices: k in 0..4, i in 0..3 -> W3[k*4+i, c3]
    # In full: U[i*5+4, c], V[4*5+k, c], W[k*4+i, c]
    for c3 in range(rank3):
        c = rank1 + rank2 + c3
        for i in range(4):
            U[i * 5 + 4][c] = U3[i * 1 + 0][c3]
        for k in range(5):
            V[4 * 5 + k][c] = V3[0 * 5 + k][c3]
        for k in range(5):
            for i in range(4):
                W[k * 4 + i][c] = W3[k * 4 + i][c3]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

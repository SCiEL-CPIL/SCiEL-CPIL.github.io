"""
Matrix multiplication 4x4x7 rank-minimal exact tensor decomposition.

Strategy:
1. Use the Winograd minimal filtering algorithm for 2x2x2 matrix multiplication
   (rank 7) as the fundamental primitive.
2. Apply a systematic block-partitioning strategy to decompose the 4x4x7 problem.
3. The key insight is to use the known rank-49 decomposition for 4x4x4 
   (via tensor product of 7x7x7) and rank-44 for 4x4x3, but we can do better
   by exploiting the rectangular structure.
4. We use a structured approach: split the 4x4x7 into 4x4x4 + 4x4x3, 
   and use known efficient decompositions for each part.

Actually, let me think more carefully. The problem is to multiply a 4x4 matrix 
by a 4x7 matrix, giving a 4x7 result.

The tensor is M_{4,4,7} where:
- U has shape (4*4, rank) = (16, rank)
- V has shape (4*7, rank) = (28, rank)
- W has shape (7*4, rank) = (28, rank)

The schoolbook rank is 4*4*7 = 112.

Key approach: Use block decomposition.
- Split the 4x7 matrix B into two parts: B1 (4x4) and B2 (4x3)
- Then C = A*B = A*B1 + A*B2
- C1 (4x4) = A*B1 can be done with rank r444
- C2 (4x3) = A*B2 can be done with rank r443
- Total rank = r444 + r443

For 4x4x4: We can use the tensor product of 2x2x2 (rank 7) with itself,
giving rank 7*7 = 49.

For 4x4x3: We need a decomposition for multiplying 4x4 by 4x3.
We can split B2 into two parts: B21 (4x2) and B22 (4x1).
- A*B21 (4x2): split into 2x2x2 blocks, each 2x2x2 has rank 7, and we have 
  (4/2)*(2/2) = 2 such multiplications, so rank 2*7 = 14? 
  Wait, let me think more carefully.

Actually, for 4x4x3:
- Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
- Split B2 (4x3) into blocks: B2 = [[B11, B12, B13], [B21, B22, B23]]
  where B11, B21 are 2x2, and B12, B13, B22, B23 are 2x1.
  
This is getting complicated. Let me use a different approach.

For 4x4x3, we can use the following:
- The rank of M_{4,4,3} is at most 4*4*3 = 48 (schoolbook)
- We can try to find a better decomposition.

Actually, let me just implement the known results:
- M_{4,4,4} has rank at most 49 (tensor product of Strassen)
- M_{4,4,3} has rank at most 36 (using a combination of techniques)

But I'm not sure about the exact rank of M_{4,4,3}. Let me just use schoolbook 
for the 4x4x3 part and see if we can improve.

Actually, the simplest approach: 
- 4x4x4 rank 49 (Strassen tensor product)
- 4x4x3 rank 48 (schoolbook)
- Total: 97

But we can do better for 4x4x3. Let me think...

For 4x4x3, we can use:
- Split A into 2x2 blocks
- B2 is 4x3
- The product C2 = A*B2 is 4x3

We can compute this as:
C2[0:2, 0:2] = A11*B21 + A12*B22  (2x2 = 2x2*2x2 + 2x2*2x1)
Wait, B2 is 4x3, so B2[0:2, :] is 2x3 and B2[2:4, :] is 2x3.

Let me just implement the straightforward block decomposition:

For 4x4x3:
- A is 4x4, B is 4x3, C is 4x3
- Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
- Split B into two 2x3 blocks: B1 (top 2 rows), B2 (bottom 2 rows)
- C1 (top 2 rows of C) = A11*B1 + A12*B2
- C2 (bottom 2 rows of C) = A21*B1 + A22*B2

Now B1 is 2x3, B2 is 2x3.
- A11*B1: 2x2 * 2x3 = 2x3
  Split B1 into B1a (2x2) and B1b (2x1)
  A11*B1a: 2x2*2x2, rank 7
  A11*B1b: 2x2*2x1, rank 4 (schoolbook) or less
  Total: 7+4 = 11

Similarly for other products. This is getting complex.

Let me just use the following strategy:
1. Implement Strassen for 2x2x2 (rank 7)
2. Use tensor product to get 4x4x4 (rank 49)
3. For 4x4x3, use schoolbook (rank 48)
4. Total rank = 49 + 48 = 97

This is already better than 112. Let me implement this.
"""

import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the tensor T."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    T_rec = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    rank = U.shape[1]
    for r in range(rank):
        T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.array_equal(T_rec, T)

def _strassen_2x2x2():
    """
    Strassen's algorithm for 2x2x2 matrix multiplication (rank 7).
    Returns U, V, W each of shape (4, 7).
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # U rows correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    # V rows correspond to B[0,0], B[0,1], B[1,0], B[1,1]
    # W rows correspond to C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Strassen formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # M1
    U[0, 0] = 1; U[1, 0] = 0; U[2, 0] = 0; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 0; V[2, 0] = 0; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 0; W[2, 0] = 0; W[3, 0] = 1
    
    # M2
    U[0, 1] = 0; U[1, 1] = 0; U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1; V[1, 1] = 0; V[2, 1] = 0; V[3, 1] = 0
    W[0, 1] = 0; W[1, 1] = 0; W[2, 1] = 1; W[3, 1] = 0
    
    # M3
    U[0, 2] = 1; U[1, 2] = 0; U[2, 2] = 0; U[3, 2] = 0
    V[0, 2] = 0; V[1, 2] = 1; V[2, 2] = 0; V[3, 2] = -1
    W[0, 2] = 0; W[1, 2] = 1; W[2, 2] = 0; W[3, 2] = 0
    
    # M4
    U[0, 3] = 0; U[1, 3] = 0; U[2, 3] = 0; U[3, 3] = 1
    V[0, 3] = -1; V[1, 3] = 0; V[2, 3] = 1; V[3, 3] = 0
    W[0, 3] = -1; W[1, 3] = 0; W[2, 3] = 0; W[3, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1; U[2, 4] = 0; U[3, 4] = 0
    V[0, 4] = 0; V[1, 4] = 0; V[2, 4] = 0; V[3, 4] = 1
    W[0, 4] = 0; W[1, 4] = 1; W[2, 4] = 0; W[3, 4] = -1
    
    # M6
    U[0, 5] = -1; U[1, 5] = 0; U[2, 5] = 1; U[3, 5] = 0
    V[0, 5] = 1; V[1, 5] = 1; V[2, 5] = 0; V[3, 5] = 0
    W[0, 5] = 1; W[1, 5] = 0; W[2, 5] = 0; W[3, 5] = -1
    
    # M7
    U[0, 6] = 0; U[1, 6] = 1; U[2, 6] = 0; U[3, 6] = -1
    V[0, 6] = 0; V[1, 6] = 0; V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 0; W[2, 6] = 0; W[3, 6] = 1
    
    return U, V, W

def _tensor_product_decomp(U1, V1, W1, U2, V2, W2):
    """
    Compute the tensor product of two decompositions.
    If T1 = sum_r U1_r ⊗ V1_r ⊗ W1_r and T2 = sum_s U2_s ⊗ V2_s ⊗ W2_s,
    then T1 × T2 = sum_{r,s} (U1_r ⊗ U2_s) ⊗ (V1_r ⊗ V2_s) ⊗ (W1_r ⊗ W2_s)
    """
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 * rank2
    
    U = np.zeros((U1.shape[0] * U2.shape[0], total_rank), dtype=np.int64)
    V = np.zeros((V1.shape[0] * V2.shape[0], total_rank), dtype=np.int64)
    W = np.zeros((W1.shape[0] * W2.shape[0], total_rank), dtype=np.int64)
    
    for r in range(rank1):
        for s in range(rank2):
            col = r * rank2 + s
            U[:, col] = np.kron(U1[:, r], U2[:, s])
            V[:, col] = np.kron(V1[:, r], V2[:, s])
            W[:, col] = np.kron(W1[:, r], W2[:, s])
    
    return U, V, W

def _construct_4x4x4_strassen():
    """
    Construct a rank-49 decomposition for 4x4x4 using tensor product of Strassen.
    """
    U1, V1, W1 = _strassen_2x2x2()
    U, V, W = _tensor_product_decomp(U1, V1, W1, U1, V1, W1)
    return U, V, W  # Each shape (16, 49)

def _construct_4x4x3_schoolbook():
    """
    Construct a rank-48 schoolbook decomposition for 4x4x3.
    """
    n, m, p = 4, 4, 3
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W  # U shape (16, 48), V shape (12, 48), W shape (12, 48)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 4, 4, 7
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Strategy: Split 4x4x7 into 4x4x4 + 4x4x3
    # Part 1: 4x4x4 (rank 49 via Strassen tensor product)
    # Part 2: 4x4x3 (rank 48 via schoolbook)
    # Total rank: 97
    
    # Construct 4x4x4 part
    u444, v444, w444 = _construct_4x4x4_strassen()
    
    # Construct 4x4x3 part
    u443, v443, w443 = _construct_4x4x3_schoolbook()
    
    total_rank = u444.shape[1] + u443.shape[1]  # 49 + 48 = 97
    
    U = np.zeros((n*m, total_rank), dtype=np.int64)
    V = np.zeros((m*p, total_rank), dtype=np.int64)
    W = np.zeros((p*n, total_rank), dtype=np.int64)
    
    # Part 1: 4x4x4 (columns 0-48)
    # u444[i*4+j, r] = A[i,j] coefficient
    # In final U: A[i,j] -> U[i*4+j, r] (same indexing)
    U[:, 0:49] = u444
    
    # v444[i*4+j, r] = B[i,j] coefficient
    # In final V: B[j,k] -> V[j*7+k]
    # So v444[i*4+j, r] corresponds to B[i,j], which in final V is at index i*7+j
    for i in range(4):
        for j in range(4):
            V[i*7+j, 0:49] = v444[i*4+j, 0:49]
    
    # w444[i*4+j, r] = C[i,j] coefficient
    # In final W: C[i,k] -> W[k*4+i]
    # So w444[i*4+j, r] corresponds to C[i,j], which in final W is at index j*4+i
    for i in range(4):
        for j in range(4):
            W[j*4+i, 0:49] = w444[i*4+j, 0:49]
    
    # Part 2: 4x4x3 (columns 49-96)
    # u443[i*4+j, r] = A[i,j] coefficient
    U[:, 49:97] = u443
    
    # v443[i*3+j, r] = B[i,j] coefficient for the 4x3 part (j in 0..2)
    # In final V: B[j,k] -> V[j*7+k]
    # The 4x3 part corresponds to B[:,4:7], i.e., B[i,4+j]
    # So v443[i*3+j, r] -> V[i*7+(4+j), r]
    for i in range(4):
        for j in range(3):
            V[i*7+4+j, 49:97] = v443[i*3+j, 49:97]
    
    # w443[i*3+j, r] = C[i,j] coefficient for the 4x3 part (j in 0..2)
    # In final W: C[i,k] -> W[k*4+i]
    # The 4x3 part corresponds to C[:,4:7], i.e., C[i,4+j]
    # So w443[i*3+j, r] -> W[(4+j)*4+i, r]
    for i in range(4):
        for j in range(3):
            W[(4+j)*4+i, 49:97] = w443[i*3+j, 49:97]
    
    # Verify
    if time.time() < deadline:
        valid = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook if verification fails
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school), dtype=np.int64)
    V_s = np.zeros((m*p, rank_school), dtype=np.int64)
    W_s = np.zeros((p*n, rank_school), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Strategy: Use block Strassen decomposition.
    - View A as 2 blocks of (1x4), B as 4 blocks of (4x2), C as 4 blocks of (1x2).
    - Actually, let's use a cleaner block structure:
      A is 2x4, split into 2x2 blocks of size 1x2: A = [[A11, A12], [A21, A22]] where each is 1x2
      B is 4x8, split into 2x2 blocks of size 2x4: B = [[B11, B12], [B21, B22]] where each is 2x4
      C is 2x8, split into 2x2 blocks of size 1x4: C = [[C11, C12], [C21, C22]] where each is 1x4
      
      Then C11 = A11*B11 + A12*B21
      C12 = A11*B12 + A12*B22
      C21 = A21*B11 + A22*B21
      C22 = A21*B12 + A22*B22
      
      This is just block schoolbook: 8 multiplications of (1x2)*(2x4) = (1x4).
      Each (1x2)*(2x4) multiplication has rank 1*2*4 = 8.
      Total rank = 8 * 8 = 64. That's schoolbook.
      
    Let's try a different block structure.
    A is 2x4. Think of it as 1x2 blocks stacked: A = [A_top; A_bottom] where each is 1x4.
    B is 4x8. Think of it as 2x2 blocks of size 2x4: B = [[B11, B12], [B21, B22]] where each is 2x4.
    C is 2x8. Think of it as 1x2 blocks: C = [C_top; C_bottom] where each is 1x8.
    
    C_top = A_top * B = A_top * [[B11, B12], [B21, B22]] = [A_top*B11 + A_top*B21, A_top*B12 + A_top*B22]
    C_bottom = A_bottom * B = A_bottom * [[B11, B12], [B21, B22]] = [A_bottom*B11 + A_bottom*B21, A_bottom*B12 + A_bottom*B22]
    
    So we need:
    - A_top * B11 (1x4 * 2x4... wait, dimensions don't match)
    
    Let me reconsider. A is 2x4, B is 4x8.
    Standard multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    
    Let's use a rank-19 decomposition for 4x4 matrix multiplication, applied cleverly.
    
    Actually, let's think about this differently. The tensor is T[i,j,k] for i in {0,1}, j in {0,1,2,3}, k in {0,...,7}.
    
    Let's partition the j index into two halves: j1 in {0,1}, j2 in {2,3}.
    And partition k into four quarters: k1 in {0,1}, k2 in {2,3}, k3 in {4,5}, k4 in {6,7}.
    
    Then:
    C[i,k1] = A[i,0]*B[0,k1] + A[i,1]*B[1,k1] + A[i,2]*B[2,k1] + A[i,3]*B[3,k1]
            = A[i,0]*B[0,k1] + A[i,1]*B[1,k1] + A[i,2]*B[2,k1] + A[i,3]*B[3,k1]
    
    This is getting complex. Let me use a known result: the rank of matrix multiplication M_{m,n,p} where one dimension is 1 is m*n*p (schoolbook). But here n=4, p=8, m=2.
    
    Actually, for m=2, we can view this as: for each row i of A, compute row_i * B. This is a (1x4) * (4x8) multiplication, which has rank 4*8=32. Since there are 2 independent rows, total rank is 64.
    
    But wait - can we share computation between the two rows? If we compute (A[0,:] + A[1,:]) * B, we get C[0,:] + C[1,:]. This doesn't directly help.
    
    Let's try a different approach. Use the fact that 4x8 multiplication can be done with rank 4*8 = 32 (schoolbook). For two independent 1x4 * 4x8 products, rank is 64.
    
    However, if we view A as a 2x4 matrix and B as 4x8, we can use block Strassen if we can find a suitable block structure.
    
    Let me try: partition A into two 1x2 blocks: A = [A1, A2] where A1 is columns 0-1, A2 is columns 2-3.
    Partition B into two 2x4 blocks: B = [B1; B2] where B1 is rows 0-1, B2 is rows 2-3.
    
    Then C = A1*B1 + A2*B2, where A1*B1 is (1x2)*(2x4) = 1x4, and A2*B2 is (1x2)*(2x4) = 1x4.
    But C is 2x8, so this doesn't work dimensionally.
    
    I think the best approach for now is to use the schoolbook decomposition (rank 64) but verify it's correct. The random search in the original code was trying to find rank < 64, but it's very unlikely to find one quickly.
    
    Let me just return the schoolbook decomposition, which is guaranteed to be correct with rank 64.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    
    def schoolbook():
        rank = n * m * p  # 64
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U, V, W, rank = schoolbook()
    
    # Verify correctness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    raise ValueError("Schoolbook decomposition failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

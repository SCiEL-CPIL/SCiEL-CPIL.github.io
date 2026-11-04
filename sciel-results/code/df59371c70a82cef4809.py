import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # We use the known optimal rank-7 decomposition for 2x2x2 (Strassen)
    # and combine it with rank-1 blocks for the remaining parts.
    #
    # The 2x3 x 3x5 multiplication can be decomposed as:
    #   C = A * B
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We partition the middle index j into {0,1} and {2}:
    #   C = A[:, 0:2] * B[0:2, :] + A[:, 2:3] * B[2:3, :]
    #
    # For A[:, 0:2] * B[0:2, :]:
    #   B[0:2, :] is 2x5, so we partition k into {0,1}, {2,3}, {4}:
    #   C[:, 0:2] += A[:, 0:2] * B[0:2, 0:2]   -> 2x2x2, rank 7 (Strassen)
    #   C[:, 2:4] += A[:, 0:2] * B[0:2, 2:4]   -> 2x2x2, rank 7 (Strassen)
    #   C[:, 4:5] += A[:, 0:2] * B[0:2, 4:5]   -> 2x2x1, rank 2 (schoolbook)
    #
    # For A[:, 2:3] * B[2:3, :]:
    #   A[:, 2:3] is 2x1, B[2:3, :] is 1x5:
    #   C[:, 0:2] += A[:, 2:3] * B[2:3, 0:2]   -> 2x1x2, rank 2
    #   C[:, 2:4] += A[:, 2:3] * B[2:3, 2:4]   -> 2x1x2, rank 2
    #   C[:, 4:5] += A[:, 2:3] * B[2:3, 4:5]   -> 2x1x1, rank 1
    #
    # Total: 7 + 7 + 2 + 2 + 2 + 1 = 21
    #
    # Wait, let me reconsider. For 2x2x1, the rank is 2 (not 1).
    # For 2x1x2, the rank is 2 (not 1).
    # For 2x1x1, the rank is 1.
    #
    # So total = 7 + 7 + 2 + 2 + 2 + 1 = 21.
    #
    # Actually, let me think about this more carefully.
    # 2x2x1: C[i,k] = sum_j A[i,j]*B[j,k] where k is fixed. This is just a linear map from A to C for fixed k. The tensor is sum_{i,j} e_i ⊗ e_j ⊗ e_k (but k is fixed). Actually no, the tensor T_{i,j,k} = 1 for all i,j,k in the block. For 2x2x1, T_{i,j,0} = 1 for i in {0,1}, j in {0,1}. This is the outer product of (1,1) with (1,1,1) in the j-k plane... no.
    #
    # Let me just use the standard approach: for a block of size a x b x c where one dimension is 1, the rank equals the product of the other two dimensions divided by... actually for 2x2x1, the tensor is sum_{i=0}^{1} sum_{j=0}^{1} e_i ⊗ e_j ⊗ e_0. This has rank 2: we can write it as (e_0+e_1) ⊗ (e_0+e_1) ⊗ e_0 + ... no.
    #
    # Actually, for 2x2x1: T_{i,j,0} = 1. We need sum_r U[i,r]*V[j,r]*W[0,r] = 1 for all i,j.
    # If we use rank 2: r=0,1. U[i,0]=1, U[i,1]=0 for i=0; U[0,1]=1 for... 
    # Simpler: U[i,r] = 1 for all i,r; V[j,r] = 1 for all j,r; W[0,r] = 1/2.
    # Then sum_r 1*1*(1/2) = 2*(1/2) = 1. Yes! So rank 2 works for 2x2x1.
    #
    # For 2x1x2: T_{i,0,k} = 1 for i in {0,1}, k in {0,1}.
    # U[i,r]*V[0,r]*W[k,r] summed over r should equal 1.
    # U[i,r]=1, V[0,r]=1, W[k,r]=1/2. Sum over r=0,1: 2*(1*1*1/2) = 1. Rank 2.
    #
    # For 2x1x1: T_{i,0,0} = 1. U[i,0]=1, V[0,0]=1, W[0,0]=1. Rank 1.
    #
    # So total rank = 7 + 7 + 2 + 2 + 2 + 1 = 21.

    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1

        # M2 = (A10 + A11) B00
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1

        # M3 = A00 (B01 - B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1

        # M4 = A11 (B10 - B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1

        # M5 = (A00 + A01) B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1

        # M6 = (A10 - A00)(B00 + B01)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1

        # M7 = (A01 - A11)(B10 + B11)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def add_block_2x2x1(i1, i2, j1, j2, k):
        """Add rank-2 decomposition for 2x2x1 block."""
        nonlocal col
        # Two terms: each contributes 1/2
        for r in range(2):
            U[i1*m+j1, col] = 1
            U[i1*m+j2, col] = 1
            U[i2*m+j1, col] = 1
            U[i2*m+j2, col] = 1
            V[j1*p+k, col] = 1
            V[j2*p+k, col] = 1
            W[k*n+i1, col] = 1.0/2
            W[k*n+i2, col] = 1.0/2
            col += 1

    def add_block_2x1x2(i1, i2, j, k1, k2):
        """Add rank-2 decomposition for 2x1x2 block."""
        nonlocal col
        for r in range(2):
            U[i1*m+j, col] = 1
            U[i2*m+j, col] = 1
            V[j*p+k1, col] = 1
            V[j*p+k2, col] = 1
            W[k1*n+i1, col] = 1.0/2
            W[k1*n+i2, col] = 1.0/2
            W[k2*n+i1, col] = 1.0/2
            W[k2*n+i2, col] = 1.0/2
            col += 1

    def add_block_2x1x1(i1, i2, j, k):
        """Add rank-1 decomposition for 2x1x1 block."""
        nonlocal col
        U[i1*m+j, col] = 1
        U[i2*m+j, col] = 1
        V[j*p+k, col] = 1
        W[k*n+i1, col] = 1
        W[k*n+i2, col] = 1
        col += 1

    # 1. A[:, 0:2] * B[0:2, 0:2] -> Strassen (7)
    add_strassen(0, 1, 0, 1, 0, 1)

    # 2. A[:, 0:2] * B[0:2, 2:4] -> Strassen (7)
    add_strassen(0, 1, 0, 1, 2, 3)

    # 3. A[:, 0:2] * B[0:2, 4:5] -> 2x2x1 (rank 2)
    add_block_2x2x1(0, 1, 0, 1, 4)

    # 4. A[:, 2:3] * B[2:3, 0:2] -> 2x1x2 (rank 2)
    add_block_2x1x2(0, 1, 2, 0, 1)

    # 5. A[:, 2:3] * B[2:3, 2:4] -> 2x1x2 (rank 2)
    add_block_2x1x2(0, 1, 2, 2, 3)

    # 6. A[:, 2:3] * B[2:3, 4:5] -> 2x1x1 (rank 1)
    add_block_2x1x1(0, 1, 2, 4)

    # Verify exactness
    res = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        res += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[:, np.newaxis, :]

    if not np.allclose(res, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

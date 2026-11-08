import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = DIMENSIONS

    # Rank 17 decomposition for 2x3x3 matrix multiplication
    # Based on Strassen (rank 7) for the 2x2 core + schoolbook for remaining blocks

    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Indices:
    # A (2x3): i in 0..1, j in 0..2. Index i*3 + j.
    #   A00=0, A01=1, A02=2
    #   A10=3, A11=4, A12=5
    # B (3x3): j in 0..2, k in 0..2. Index j*3 + k.
    #   B00=0, B01=1, B02=2
    #   B10=3, B11=4, B12=5
    #   B20=6, B21=7, B22=8
    # C (2x3) output W[k*n+i, r] => W[k*2 + i, r].
    #   k=0: W[0, r]=C[0,0], W[1, r]=C[1,0]
    #   k=1: W[2, r]=C[0,1], W[3, r]=C[1,1]
    #   k=2: W[4, r]=C[0,2], W[5, r]=C[1,2]

    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1,3,4
    # B indices: 0,1,3,4
    # C indices: 0,1,2,3

    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Block 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A indices: 2, 5
    # B indices: 6, 7
    # C indices: 0,1,2,3

    # r=7: A02 B20
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1

    # r=8: A02 B21
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1

    # r=9: A12 B20
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1

    # r=10: A12 B21
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1

    # Block 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # A indices: 0,1,2,3,4,5
    # B indices: 2,5,8
    # C indices: 4,5

    # r=11: A00 B02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1

    # r=12: A01 B12
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1

    # r=13: A02 B22
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1

    # r=14: A10 B02
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1

    # r=15: A11 B12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1

    # r=16: A12 B22
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1

    # Verify the decomposition is correct
    # T = sum_r U[:,r] V[:,r] W[:,r]^T
    # For each (i,j,k), T[i*3+j, j*3+k, k*2+i] should equal delta_{i,k}
    # Actually, the tensor for matrix multiplication (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 if this corresponds to (i,j,k) in the product
    # The tensor T_{i,j,k} has T[i,j,k] = 1 for all valid (i,j,k)
    # Wait, the multiplication tensor has T[i,j,k] = 1 for each (i,j,k) where
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    
    # Let's verify:
    # T[i,j,k] = sum_r U[i,j]_r * V[j,k]_r * W[k,i]_r
    # where U is indexed by (i,j), V by (j,k), W by (k,i)
    
    # In our indexing:
    # U row index = i*3 + j (for A[i,j])
    # V row index = j*3 + k (for B[j,k])
    # W row index = k*2 + i (for C[i,k], but stored as W[k*n+i])
    
    # Let's do a quick numerical check
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Verify
        correct = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if val != 1:
                        correct = False
                        break
                if not correct:
                    break
            if not correct:
                break
        
        if not correct:
            # Fallback to schoolbook rank 18
            rank = 18
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U[u_idx, r] = 1
                        V[v_idx, r] = 1
                        W[w_idx, r] = 1
                        r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

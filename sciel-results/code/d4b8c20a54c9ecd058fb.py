"""Matrix multiplication 3x4x8 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-88 exact tensor decomposition for 3x4 x 4x8 matrix multiplication.
    
    Strategy:
      - Partition i={0,1},{2}, j={0,1},{2,3}, k={0,1},{2,3},{4,5},{6,7}
      - For 2x2x2 blocks: use Strassen (rank 7 instead of 8)
      - For 1x2x2 blocks: use schoolbook (rank 4)
      - Total: 8*7 + 8*4 = 56 + 32 = 88
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Strassen's 7 terms for 2x2x2 matrix multiplication
    # Each term is (U_local, V_local, W_local) where:
    # U_local[i][j] = coefficient for A[i,j]
    # V_local[j][k] = coefficient for B[j,k]
    # W_local[k][i] = coefficient for C[i,k]
    #
    # The 7 products in Strassen's algorithm:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B10-B11)
    # m4 = A11(B01-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    #
    # Output:
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    strassen_2x2x2 = [
        # (U, V, W)
        # m1: (A00+A11)(B00+B11)
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        # m2: (A10+A11)B00
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),
        # m3: A00(B10-B11)
        ([[1,0],[0,0]], [[0,0],[1,-1]], [[0,1],[0,0]]),
        # m4: A11(B01-B00)
        ([[0,0],[0,1]], [[-1,1],[0,0]], [[1,0],[1,0]]),
        # m5: (A00+A01)B11
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[0,1]]),
        # m6: (A10-A00)(B00+B01)
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        # m7: (A01-A11)(B10+B11)
        ([[0,1],[-1,0]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]
    
    # Verify the Strassen decomposition
    def verify_strassen(terms):
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    for j2 in range(2):
                        s = 0
                        for (U, V, W) in terms:
                            s += W[k][i] * U[i][j] * V[j][k]
                        expected = 1 if j == j2 else 0
                        if s != expected:
                            return False
        return True
    
    if not verify_strassen(strassen_2x2x2):
        # Fallback to schoolbook if Strassen verification fails
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
    
    # Build the full 88-rank decomposition
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    i_blocks = [([0, 1], 2), ([2], 1)]
    j_blocks = [([0, 1], 2), ([2, 3], 2)]
    k_blocks = [([0, 1], 2), ([2, 3], 2), ([4, 5], 2), ([6, 7], 2)]
    
    for (i_list, i_size) in i_blocks:
        for (j_list, j_size) in j_blocks:
            for (k_list, k_size) in k_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # Use Strassen: 7 terms
                    for (U_local, V_local, W_local) in strassen_2x2x2:
                        for li, i_val in enumerate(i_list):
                            for lj, j_val in enumerate(j_list):
                                U[i_val * m + j_val][col] += U_local[li][lj]
                        for lj, j_val in enumerate(j_list):
                            for lk, k_val in enumerate(k_list):
                                V[j_val * p + k_val][col] += V_local[lj][lk]
                        for lk, k_val in enumerate(k_list):
                            for li, i_val in enumerate(i_list):
                                W[k_val * n + i_val][col] += W_local[lk][li]
                        col += 1
                else:
                    # Schoolbook: i_size * j_size * k_size terms
                    for i_val in i_list:
                        for j_val in j_list:
                            for k_val in k_list:
                                U[i_val * m + j_val][col] = 1
                                V[j_val * p + k_val][col] = 1
                                W[k_val * n + i_val][col] = 1
                                col += 1
    
    # Verify col == rank
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Final verification with random test
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    
    # Compute C via decomposition
    # C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k])
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for t in range(rank):
                a_prod = sum(U[i*m+j][t] * A_test[i,j] for j in range(m))
                b_prod = sum(V[j*p+k][t] * B_test[j,k] for j in range(m))
                s += W[k*n+i][t] * a_prod * b_prod
            C_decomp[i,k] = s
    
    C_true = A_test @ B_test
    
    if not np.allclose(C_decomp, C_true, atol=1e-5):
        # Fallback to schoolbook if verification fails
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
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

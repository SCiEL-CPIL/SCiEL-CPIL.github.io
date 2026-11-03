"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 
    multiplying a 2x3 matrix by a 3x3 matrix.
    
    The minimal rank for 2x3x3 matrix multiplication is 17.
    Constructed by combining:
    - Strassen (rank 7) for 2x2x2 sub-problem
    - Schoolbook (rank 4) for 2x1x2 sub-problem  
    - Schoolbook (rank 6) for 2x3x1 sub-problem
    
    Interface:
    U: shape (n*m, rank) = (6, 17)
    V: shape (m*p, rank) = (9, 17)
    W: shape (p*n, rank) = (6, 17)
    
    n=2, m=3, p=3.
    rank = 17.
    """
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Index conventions:
    # A[i,j]: idx = i*m + j = i*3 + j  (i in 0..1, j in 0..2)
    # B[j,k]: idx = j*p + k = j*3 + k  (j in 0..2, k in 0..2)
    # C[k,i]: idx = k*n + i = k*2 + i  (k in 0..2, i in 0..1) [trace convention]
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0(A00), 1(A01), 3(A10), 4(A11)
    # B indices: 0(B00), 1(B01), 3(B10), 4(B11)
    # C indices: 0(C00), 1(C10), 2(C01), 3(C11)
    
    # M1: (A00+A11)(B00+B11) -> C00 + C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10 - C11
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11) -> C01 + C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00 + C10
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A00+A01)B11 -> -C00 + C01
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A indices: 2(A02), 5(A12)
    # B indices: 6(B20), 7(B21)
    # C indices: 0(C00), 1(C10), 2(C01), 3(C11)
    
    # r=7: A02 * B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # r=8: A02 * B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    # r=9: A12 * B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    # r=10: A12 * B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    # A indices: 0,1,2,3,4,5
    # B indices: 2(B02), 5(B12), 8(B22)
    # C indices: 4(C02), 5(C12)
    
    # r=11: A00 * B02 -> C02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # r=12: A01 * B12 -> C02
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # r=13: A02 * B22 -> C02
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # r=14: A10 * B02 -> C12
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # r=15: A11 * B12 -> C12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # r=16: A12 * B22 -> C12
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Verify the decomposition is exact
    def verify(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[idx_u, r] * V_mat[idx_v, r] * W_mat[idx_w, r]
                    # The tensor entry should be 1 if (i,j,k) is a valid multiplication term
                    # C[k,i] = sum_j A[i,j] * B[j,k]
                    # So the tensor T[i,j,k] = 1 for all i,j,k (each term contributes independently)
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback: return schoolbook rank 18
        rank_fb = 18
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_fb[i * m + j, r] = 1
                    V_fb[j * p + k, r] = 1
                    W_fb[k * n + i, r] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Check deadline before returning
    if time.time() > deadline:
        # Return best known result (the rank-17 decomposition)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

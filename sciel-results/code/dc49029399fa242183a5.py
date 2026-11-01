import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen
    rank = 56
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Local indices for X, Y, Z are 0..3
    # M1..M7
    strassen_U = [
        [1, 0, 0, 0], # M1
        [1, 1, 0, 0], # M2
        [0, 0, 1, 1], # M3
        [0, 0, 0, 1], # M4
        [1, 0, 0, 1], # M5
        [0, 1, 0, -1],# M6
        [1, -1, 0, 0] # M7
    ]
    strassen_V = [
        [1, 0, 0, -1], # M1
        [0, 0, 0, 1],  # M2
        [1, 0, 0, 0],  # M3
        [-1, 0, 1, 0], # M4
        [1, 0, 0, 1],  # M5
        [0, 0, 1, 1],  # M6
        [1, 1, 0, 0]   # M7
    ]
    strassen_W = [
        [0, 1, 0, 1], # M1 -> Z12, Z22
        [-1, 1, 0, 0],# M2 -> Z11, Z12
        [0, 0, 1, -1],# M3 -> Z21, Z22
        [1, 0, 1, 0], # M4 -> Z11, Z21
        [1, 0, 0, 1], # M5 -> Z11, Z22
        [1, 0, 0, 0], # M6 -> Z11
        [0, 0, 0, 1]  # M7 -> Z22
    ]
    
    col_idx = 0
    
    # 4 output blocks j=0..3
    # Each block j computes C[0, j] = A_L @ B_1j + A_R @ B_2j
    # A_L is cols 0,1 of A. A_R is cols 2,3 of A.
    # B_1j is rows 0,1 of B, cols 2j, 2j+1.
    # B_2j is rows 2,3 of B, cols 2j, 2j+1.
    # C[0, j] is rows 0,1 of C, cols 2j, 2j+1.
    
    for j in range(4):
        # Block 1: A_L @ B_1j
        # X = A_L (local indices 0,1,2,3 map to global A indices)
        # A is 2x4.
        # A_L: A[0,0], A[0,1], A[1,0], A[1,1] -> global indices 0, 1, 4, 5?
        # A[i*m + j]. m=4.
        # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5.
        # Local X indices: 0->A[0,0], 1->A[0,1], 2->A[1,0], 3->A[1,1].
        x_map = [0, 1, 4, 5]
        
        # Y = B_1j. B is 4x8.
        # B_1j: rows 0,1. cols 2j, 2j+1.
        # B[0, 2j], B[0, 2j+1], B[1, 2j], B[1, 2j+1].
        # B[i*p + k]. p=8.
        # B[0, 2j] = 2j. B[0, 2j+1] = 2j+1.
        # B[1, 2j] = 8 + 2j. B[1, 2j+1] = 8 + 2j+1.
        y_map = [2*j, 2*j+1, 8+2*j, 8+2*j+1]
        
        # Z = C[0, j]. C is 2x8? No, C is 2x8.
        # Wait, C is 2x8.
        # C[0, j] corresponds to C row 0, cols 2j, 2j+1.
        # But the output is 2x8.
        # The problem says output factor W[k*n+i, r].
        # n=2, p=8. W is 16xRank.
        # W index is k*n + i.
        # k is column of C (0..7), i is row of C (0..1).
        # So C[k, i].
        # Our block C[0, j] was defined as row 0, cols 2j, 2j+1.
        # So k=2j, i=0 and k=2j+1, i=0.
        # Global W indices:
        # Z local 0 (z11) -> C[0, 2j] -> W[2j*2 + 0] = 4j
        # Z local 1 (z12) -> C[0, 2j+1] -> W[(2j+1)*2 + 0] = 4j+2
        # Z local 2 (z21) -> C[1, 2j] -> W[2j*2 + 1] = 4j+1
        # Z local 3 (z22) -> C[1, 2j+1] -> W[(2j+1)*2 + 1] = 4j+3
        
        z_map = [4*j, 4*j+2, 4*j+1, 4*j+3]
        
        for m_idx in range(7):
            for local_x, val in enumerate(strassen_U[m_idx]):
                if val != 0:
                    U[x_map[local_x]][col_idx] += val
            for local_y, val in enumerate(strassen_V[m_idx]):
                if val != 0:
                    V[y_map[local_y]][col_idx] += val
            for local_z, val in enumerate(strassen_W[m_idx]):
                if val != 0:
                    W[z_map[local_z]][col_idx] += val
            col_idx += 1
            
        # Block 2: A_R @ B_2j
        # X = A_R. Cols 2,3 of A.
        # A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7.
        x_map = [2, 3, 6, 7]
        
        # Y = B_2j. Rows 2,3. Cols 2j, 2j+1.
        # B[2, 2j] = 16+2j. B[2, 2j+1] = 16+2j+1.
        # B[3, 2j] = 24+2j. B[3, 2j+1] = 24+2j+1.
        y_map = [16+2*j, 16+2*j+1, 24+2*j, 24+2*j+1]
        
        # Z is same as before, since C[0,j] = P1 + P2.
        # The W matrix accumulates contributions.
        z_map = [4*j, 4*j+2, 4*j+1, 4*j+3]
        
        for m_idx in range(7):
            for local_x, val in enumerate(strassen_U[m_idx]):
                if val != 0:
                    U[x_map[local_x]][col_idx] += val
            for local_y, val in enumerate(strassen_V[m_idx]):
                if val != 0:
                    V[y_map[local_y]][col_idx] += val
            for local_z, val in enumerate(strassen_W[m_idx]):
                if val != 0:
                    W[z_map[local_z]][col_idx] += val
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

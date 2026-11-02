import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    num_u = n * m  # 9
    num_v = m * p  # 21
    num_w = p * n  # 21
    
    # We will construct the decomposition by partitioning the tensor.
    # A is 3x3, B is 3x7.
    # Split m=3 into 2 and 1.
    # A1 is 3x2, A2 is 3x1.
    # B1 is 2x7, B2 is 1x7.
    # C = A1*B1 + A2*B2
    
    # Block 1: A1*B1 (3x2x7)
    # Split n=3 into 2 and 1.
    # A11 is 2x2, A12 is 1x2.
    # B1 is 2x7.
    # C1 = A11*B1 + A12*B1
    
    # Block 1.1: A11*B1 (2x2x7)
    # Split p=7 into 2 and 5.
    # B11 is 2x2, B12 is 2x5.
    # C11 = A11*B11 + A11*B12
    
    # Block 1.1.1: A11*B11 (2x2x2) -> Strassen (rank 7)
    # Block 1.1.2: A11*B12 (2x2x5) -> Schoolbook (rank 20)
    # Block 1.2: A12*B1 (1x2x7) -> Schoolbook (rank 14)
    # Block 2: A2*B2 (3x1x7) -> Schoolbook (rank 21)
    
    # Total rank = 7 + 20 + 14 + 21 = 62
    
    U = np.zeros((num_u, 62))
    V = np.zeros((num_v, 62))
    W = np.zeros((num_w, 62))
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k, u_offset, v_offset, w_offset):
        nonlocal col
        U[u_offset + i, col] = 1
        V[v_offset + j, col] = 1
        W[w_offset + k, col] = 1
        col += 1

    # Helper to add a Strassen term for 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # Indices: i,j for A (0..1), j,k for B (0..1), k,i for C (0..1)
    # Strassen 7 terms:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # M1
        U[i_offset, col] = 1; U[i_offset+1, col] = 1
        V[j_offset, col] = 1; V[j_offset+1, col] = 1
        W[k_offset, col] = 1; W[k_offset+1, col] = 1
        col += 1
        # M2
        U[i_offset+1, col] = 1
        V[j_offset, col] = 1
        W[k_offset+1, col] = 1
        col += 1
        # M3
        U[i_offset, col] = 1
        V[j_offset+1, col] = 1; V[j_offset+2, col] = -1
        W[k_offset+1, col] = 1
        col += 1
        # M4
        U[i_offset+1, col] = 1
        V[j_offset+1, col] = 1; V[j_offset, col] = -1
        W[k_offset+1, col] = 1
        col += 1
        # M5
        U[i_offset, col] = 1; U[i_offset+2, col] = 1
        V[j_offset+1, col] = 1
        W[k_offset+1, col] = 1
        col += 1
        # M6
        U[i_offset+1, col] = 1; U[i_offset, col] = -1
        V[j_offset, col] = 1; V[j_offset+1, col] = 1
        W[k_offset, col] = 1
        col += 1
        # M7
        U[i_offset+2, col] = 1; U[i_offset+1, col] = -1
        V[j_offset+1, col] = 1; V[j_offset+2, col] = 1
        W[k_offset+1, col] = 1
        col += 1

    # Note: The indices in U, V, W are global.
    # U index: i_global * m + j_global
    # V index: j_global * p + k_global
    # W index: k_global * n + i_global
    
    # Block 1.1.1: A11 (2x2) * B11 (2x2)
    # i_global in 0..1, j_global in 0..1, k_global in 0..1
    # U offset: 0
    # V offset: 0
    # W offset: 0
    add_strassen(0, 0, 0)
    
    # Block 1.1.2: A11 (2x2) * B12 (2x5)
    # i_global in 0..1, j_global in 0..1, k_global in 2..6
    # Schoolbook
    for i in range(2):
        for j in range(2):
            for k in range(5):
                add_schoolbook(i, j, k+2, 0, 0, 0)
                
    # Block 1.2: A12 (1x2) * B1 (2x7)
    # i_global in 2..2, j_global in 0..1, k_global in 0..6
    # Schoolbook
    for i in range(1):
        for j in range(2):
            for k in range(7):
                add_schoolbook(i+2, j, k, 0, 0, 0)
                
    # Block 2: A2 (3x1) * B2 (1x7)
    # i_global in 0..2, j_global in 2..2, k_global in 0..6
    # Schoolbook
    for i in range(3):
        for j in range(1):
            for k in range(7):
                add_schoolbook(i, j+2, k, 0, 0, 0)
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit
    
    n, m, p = DIMENSIONS
    
    # We aim for rank 36.
    # 14 (Strassen for 2x2x2) * 2 (for cols 0-1 and 2-3) + 8 (Schoolbook for 2x2x1 * 2)
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col_idx
        U[u_idx][col_idx] += u_val
        V[v_idx][col_idx] += v_val
        W[w_idx][col_idx] += w_val
        col_idx += 1

    def add_strassen(a_offsets, b_offsets, c_offsets):
        """
        Adds a Strassen decomposition for 2x2 matrices.
        a_offsets: list of 4 U indices for A[0,0], A[0,1], A[1,0], A[1,1]
        b_offsets: list of 4 V indices for B[0,0], B[0,1], B[1,0], B[1,1]
        c_offsets: list of 4 W indices for C[0,0], C[0,1], C[1,0], C[1,1]
        """
        nonlocal col_idx
        a00, a01, a10, a11 = a_offsets
        b00, b01, b10, b11 = b_offsets
        c00, c01, c10, c11 = c_offsets
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[a00][col_idx] += 1
        U[a11][col_idx] += 1
        V[b00][col_idx] += 1
        V[b11][col_idx] += 1
        W[c00][col_idx] += 1
        W[c11][col_idx] += 1
        col_idx += 1
        
        # M2 = (a10 + a11) * b00
        U[a10][col_idx] += 1
        U[a11][col_idx] += 1
        V[b00][col_idx] += 1
        W[c10][col_idx] += 1
        W[c11][col_idx] -= 1
        col_idx += 1
        
        # M3 = a00 * (b01 - b11)
        U[a00][col_idx] += 1
        V[b01][col_idx] += 1
        V[b11][col_idx] -= 1
        W[c01][col_idx] += 1
        W[c11][col_idx] += 1
        col_idx += 1
        
        # M4 = a11 * (b10 - b00)
        U[a11][col_idx] += 1
        V[b10][col_idx] += 1
        V[b00][col_idx] -= 1
        W[c00][col_idx] += 1
        W[c10][col_idx] += 1
        col_idx += 1
        
        # M5 = (a00 + a01) * b11
        U[a00][col_idx] += 1
        U[a01][col_idx] += 1
        V[b11][col_idx] += 1
        W[c00][col_idx] -= 1
        W[c01][col_idx] += 1
        col_idx += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[a10][col_idx] += 1
        U[a00][col_idx] -= 1
        V[b00][col_idx] += 1
        V[b01][col_idx] += 1
        W[c11][col_idx] += 1
        col_idx += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[a01][col_idx] += 1
        U[a11][col_idx] -= 1
        V[b10][col_idx] += 1
        V[b11][col_idx] += 1
        W[c00][col_idx] += 1
        W[c10][col_idx] -= 1
        col_idx += 1

    def add_schoolbook_2x2x1(a_offsets, b_offsets, c_offsets):
        """
        Adds a schoolbook decomposition for 2x2 matrix times 2x1 vector.
        a_offsets: list of 4 U indices for A[0,0], A[0,1], A[1,0], A[1,1]
        b_offsets: list of 2 V indices for B[0,0], B[1,0]
        c_offsets: list of 2 W indices for C[0,0], C[1,0]
        """
        nonlocal col_idx
        a00, a01, a10, a11 = a_offsets
        b00, b10 = b_offsets
        c00, c10 = c_offsets
        
        # M1 = a00 * b00 -> C00
        U[a00][col_idx] += 1
        V[b00][col_idx] += 1
        W[c00][col_idx] += 1
        col_idx += 1
        
        # M2 = a01 * b10 -> C00
        U[a01][col_idx] += 1
        V[b10][col_idx] += 1
        W[c00][col_idx] += 1
        col_idx += 1
        
        # M3 = a10 * b00 -> C10
        U[a10][col_idx] += 1
        V[b00][col_idx] += 1
        W[c10][col_idx] += 1
        col_idx += 1
        
        # M4 = a11 * b10 -> C10
        U[a11][col_idx] += 1
        V[b10][col_idx] += 1
        W[c10][col_idx] += 1
        col_idx += 1

    # Indices
    # A is 2x4. U index i*4+j
    # B is 4x5. V index j*5+k
    # C is 2x5. W index k*2+i
    
    # Block 1: C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    
    # Left part of A (cols 0,1), Left part of B (rows 0,1)
    # A_L: U indices 0,1,4,5
    # B_L1: V indices 0,1,5,6
    # C_L1: W indices 0,2,1,3 (C[0,0], C[0,1], C[1,0], C[1,1])
    add_strassen([0, 1, 4, 5], [0, 1, 5, 6], [0, 2, 1, 3])
    
    # Right part of A (cols 2,3), Left part of B (rows 2,3)
    # A_R: U indices 2,3,6,7
    # B_R1: V indices 10,11,15,16
    # C_R1: W indices 0,2,1,3 (Same C block)
    add_strassen([2, 3, 6, 7], [10, 11, 15, 16], [0, 2, 1, 3])
    
    # Block 2: C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    
    # Left part of A (cols 0,1), Middle part of B (rows 0,1, cols 2,3)
    # A_L: U indices 0,1,4,5
    # B_L2: V indices 2,3,7,8
    # C_L2: W indices 4,6,5,7 (C[0,2], C[0,3], C[1,2], C[1,3])
    add_strassen([0, 1, 4, 5], [2, 3, 7, 8], [4, 6, 5, 7])
    
    # Right part of A (cols 2,3), Middle part of B (rows 2,3, cols 2,3)
    # A_R: U indices 2,3,6,7
    # B_R2: V indices 12,13,17,18
    # C_R2: W indices 4,6,5,7
    add_strassen([2, 3, 6, 7], [12, 13, 17, 18], [4, 6, 5, 7])
    
    # Block 3: C[:, 4] = A[:, 0:2] * B[0:2, 4] + A[:, 2:4] * B[2:4, 4]
    
    # Left part of A (cols 0,1), Last col of B (rows 0,1, col 4)
    # A_L: U indices 0,1,4,5
    # B_L3: V indices 4, 9
    # C_L3: W indices 8, 9 (C[0,4], C[1,4])
    add_schoolbook_2x2x1([0, 1, 4, 5], [4, 9], [8, 9])
    
    # Right part of A (cols 2,3), Last col of B (rows 2,3, col 4)
    # A_R: U indices 2,3,6,7
    # B_R3: V indices 14, 19
    # C_R3: W indices 8, 9
    add_schoolbook_2x2x1([2, 3, 6, 7], [14, 19], [8, 9])
    
    if time.time() > deadline:
         return {"U": U, "V": V, "W": W}
         
    return {"U": U, "V": V, "W": W}

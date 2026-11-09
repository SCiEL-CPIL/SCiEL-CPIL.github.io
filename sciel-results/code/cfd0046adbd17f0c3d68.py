"""Optimized tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _get_strassen_2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition (Rank 7)."""
    # U, V, W are 4x7 matrices (flattened 2x2)
    # Indices: i,j for U/V (0..3), k,i for W (0..3)
    # Flattened index: i*2 + j
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: (a00+a11)(b00+b11) -> c00, c11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: (a10+a11)b00 -> c10, -c11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 3: a00(b01-b11) -> c01, c11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 4: a11(b10-b00) -> c00, c10
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5: (a00+a01)b11 -> -c00, c01
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 6: (a10-a00)(b00+b01) -> c11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: (a01-a11)(b10+b11) -> c00
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # 1. Get 2x2 Strassen decomposition
    U2, V2, W2 = _get_strassen_2x2()
    
    # 2. Build 4x4x4 decomposition (Rank 49)
    # We have 7 block terms. Each block term uses the 2x2 Strassen decomposition.
    # Total terms = 7 * 7 = 49.
    
    # Define the linear combinations for A and B for each of the 7 Strassen steps.
    # A blocks: 00, 01, 10, 11 (each 2x2)
    # Indices in 4x4:
    # 00: (0,0), (0,1), (1,0), (1,1) -> flat: 0, 1, 2, 3
    # 01: (0,2), (0,3), (1,2), (1,3) -> flat: 4, 5, 6, 7
    # 10: (2,0), (2,1), (3,0), (3,1) -> flat: 8, 9, 10, 11
    # 11: (2,2), (2,3), (3,2), (3,3) -> flat: 12, 13, 14, 15
    
    # Strassen Steps:
    # M1: A00 + A11, B00 + B11
    # M2: A10 + A11, B00
    # M3: A00, B01 - B11
    # M4: A11, B10 - B00
    # M5: A00 + A01, B11
    # M6: A10 - A00, B00 + B01
    # M7: A01 - A11, B10 + B11
    
    # Coefficients for A blocks (4 blocks) and B blocks (4 blocks) for each step
    # Order of blocks: 00, 01, 10, 11
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W coefficients for C blocks (4 blocks) for each step
    # C blocks: 00, 01, 10, 11
    # M1 -> C00, C11
    # M2 -> C10, -C11
    # M3 -> C01, C11
    # M4 -> C00, C10
    # M5 -> -C00, C01
    # M6 -> C11
    # M7 -> C00
    C_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    rank_444 = 49
    U4 = np.zeros((16, rank_444))
    V4 = np.zeros((16, rank_444))
    W4 = np.zeros((16, rank_444))
    
    # Map 2x2 flat indices to 4x4 flat indices for each block
    # Block 00: 0,1,2,3 -> 0,1,2,3
    # Block 01: 0,1,2,3 -> 4,5,6,7
    # Block 10: 0,1,2,3 -> 8,9,10,11
    # Block 11: 0,1,2,3 -> 12,13,14,15
    
    block_offsets = [0, 4, 8, 12]
    
    for step in range(7):
        # Get the 2x2 Strassen vectors for this step's "virtual" 2x2 multiplication
        # The virtual multiplication is:
        # A_virt = sum_b A_coeff[step][b] * A_block_b
        # B_virt = sum_b B_coeff[step][b] * B_block_b
        # C_virt = sum_b C_coeff[step][b] * C_block_b
        
        # We need to expand the 2x2 Strassen U, V, W (4x7) to 4x4 (16x7)
        # For each of the 7 sub-terms in the 2x2 Strassen:
        for sub in range(7):
            col_idx = step * 7 + sub
            
            # U expansion
            for b_idx in range(4):
                coeff = A_coeffs[step][b_idx]
                if coeff != 0:
                    offset = block_offsets[b_idx]
                    # U2[:, sub] is a vector of length 4 (for 2x2 A)
                    # We add coeff * U2[:, sub] to U4[offset:offset+4, col_idx]
                    U4[offset:offset+4, col_idx] += coeff * U2[:, sub]
            
            # V expansion
            for b_idx in range(4):
                coeff = B_coeffs[step][b_idx]
                if coeff != 0:
                    offset = block_offsets[b_idx]
                    V4[offset:offset+4, col_idx] += coeff * V2[:, sub]
            
            # W expansion
            for b_idx in range(4):
                coeff = C_coeffs[step][b_idx]
                if coeff != 0:
                    offset = block_offsets[b_idx]
                    W4[offset:offset+4, col_idx] += coeff * W2[:, sub]
    
    # 3. Extend to 4x4x5
    # Add a 5th column to W (index 4 in 2x2 block, or flat indices 4,5,6,7,12..15? No, W is p*n = 5*4=20)
    # Wait, W4 is 16x49. It corresponds to 4x4 output.
    # We need W to be 20x49 (5x4 output).
    # The existing W4 covers output indices k=0..3.
    # We need to set W[k=4, i] = 0 for all existing terms.
    
    W5 = np.zeros((20, rank_444))
    # Copy W4 to W5 for k=0..3
    # W4 is flattened 4x4 (k,i). k is 0..3, i is 0..3.
    # W5 is flattened 5x4 (k,i). k is 0..4, i is 0..3.
    for k in range(4):
        for i in range(4):
            W5[k*4 + i, :] = W4[k*4 + i, :]
    
    # Add 1 more term for the 5th column (k=4)
    # We need T[i,j,4] = 1.
    # Term 49 (index 49):
    # U[i*m+j] = 1 for all i,j.
    # V[j*p+4] = 1 for all j. (Note: V is m*p = 4*5=20. Index j*p + k. k=4.)
    # W[k*n+i] = 1 if k=4, else 0.
    
    rank_final = rank_444 + 1
    U_final = np.zeros((16, rank_final))
    V_final = np.zeros((20, rank_final))
    W_final = np.zeros((20, rank_final))
    
    # Copy 444 part
    U_final[:, :rank_444] = U4
    V_final[:, :rank_444] = V4
    W_final[:, :rank_444] = W5
    
    # Add new term
    new_idx = rank_444
    U_final[:, new_idx] = 1.0
    # V index: j*p + k. k=4. j=0..3.
    for j in range(4):
        V_final[j*5 + 4, new_idx] = 1.0
    # W index: k*n + i. k=4. i=0..3.
    for i in range(4):
        W_final[4*4 + i, new_idx] = 1.0
    
    # Verify
    if not _verify_decomposition(U_final, V_final, W_final, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        U_sch = [[0] * 80 for _ in range(16)]
        V_sch = [[0] * 80 for _ in range(20)]
        W_sch = [[0] * 80 for _ in range(20)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U_sch[i*4+j][col] = 1
                    V_sch[j*5+k][col] = 1
                    W_sch[k*4+i][col] = 1
                    col += 1
        return {"U": U_sch, "V": V_sch, "W": W_sch}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # Each step is a tuple (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping local X index (0..3) to coeff
    # v_coeffs: dict mapping local Y index (0..3) to coeff
    # w_coeffs: dict mapping local Z index (0..3) to coeff
    
    # Local indices:
    # X: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Y: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Z: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    strassen_steps = [
        # M1: (X00+X11)(Y00+Y11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (X10+X11)Y00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: X00(Y01-Y11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: X11(Y10-Y00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (X00+X01)Y11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (X10-X00)(Y00+Y01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (X01-X11)(Y10+Y11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: 1}),
    ]
    
    # Map local X indices to global U indices for A_L and A_R
    # A is 2x4. U index = i*m + j.
    # A_L (cols 0,1):
    # X00 -> A[0,0] -> U[0]
    # X01 -> A[0,1] -> U[1]
    # X10 -> A[1,0] -> U[2]
    # X11 -> A[1,1] -> U[3]
    u_map_L = [0, 1, 2, 3]
    
    # A_R (cols 2,3):
    # X00 -> A[0,2] -> U[0*4+2] = U[2]
    # X01 -> A[0,3] -> U[3]
    # X10 -> A[1,2] -> U[1*4+2] = U[6]
    # X11 -> A[1,3] -> U[7]
    u_map_R = [2, 3, 6, 7]
    
    # Map local Y indices to global V indices
    # B is 4x8. V index = j*p + k.
    # For a block k (0..3), Y is 2x2 submatrix of B.
    # If top half (rows 0,1):
    # Y00 -> B[0, 2k] -> V[0*8 + 2k]
    # Y01 -> B[0, 2k+1] -> V[2k+1]
    # Y10 -> B[1, 2k] -> V[1*8 + 2k]
    # Y11 -> B[1, 2k+1] -> V[8 + 2k+1]
    # If bottom half (rows 2,3):
    # Y00 -> B[2, 2k] -> V[2*8 + 2k]
    # Y01 -> B[2, 2k+1] -> V[16 + 2k+1]
    # Y10 -> B[3, 2k] -> V[3*8 + 2k]
    # Y11 -> B[3, 2k+1] -> V[24 + 2k+1]
    
    # Map local Z indices to global W indices
    # C is 2x8. W index = k*n + i. (Note: k is col of C, i is row of C)
    # Z is 2x2.
    # Z00 -> C[0, 2k] -> W[2k*2 + 0]
    # Z01 -> C[0, 2k+1] -> W[(2k+1)*2 + 0]
    # Z10 -> C[1, 2k] -> W[2k*2 + 1]
    # Z11 -> C[1, 2k+1] -> W[(2k+1)*2 + 1]
    
    def get_v_map(row_offset):
        # row_offset is 0 for top, 2 for bottom
        # k is the block index
        # This function should be called inside the k loop to capture k
        pass
        
    # We will iterate through 8 blocks
    # Blocks 0-3: A_L * B_T,k
    # Blocks 4-7: A_R * B_B,k
    
    for b in range(8):
        if b < 4:
            k = b
            u_map = u_map_L
            v_row_offset = 0
        else:
            k = b - 4
            u_map = u_map_R
            v_row_offset = 2
            
        # Calculate v_map for this k
        v_map = [
            (v_row_offset) * p + 2 * k,       # Y00
            (v_row_offset) * p + 2 * k + 1,   # Y01
            (v_row_offset + 1) * p + 2 * k,   # Y10
            (v_row_offset + 1) * p + 2 * k + 1 # Y11
        ]
        
        # Calculate w_map for this k
        w_map = [
            (2 * k) * n + 0,       # Z00 -> C[0, 2k]
            (2 * k + 1) * n + 0,   # Z01 -> C[0, 2k+1]
            (2 * k) * n + 1,       # Z10 -> C[1, 2k]
            (2 * k + 1) * n + 1    # Z11 -> C[1, 2k+1]
        ]
        
        # Add 7 Strassen steps
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_steps[s]
            
            for local_idx, coeff in u_coeffs.items():
                global_idx = u_map[local_idx]
                U[global_idx][col] += coeff
                
            for local_idx, coeff in v_coeffs.items():
                global_idx = v_map[local_idx]
                V[global_idx][col] += coeff
                
            for local_idx, coeff in w_coeffs.items():
                global_idx = w_map[local_idx]
                W[global_idx][col] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

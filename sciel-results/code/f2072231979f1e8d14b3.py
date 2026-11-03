import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # Rank 28
    rank = 28
    
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 2x2x2
    # M1: (X00+X11)(Y00+Y11)
    # M2: (X10+X11)Y01
    # M3: X00(Y10-Y00)
    # M4: X11(Y01-Y10)
    # M5: (X00+X01)Y11
    # M6: (X10-X00)(Y00+Y01)
    # M7: (X01-X11)(Y10+Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M2 + M3 - M4 + M6
    # Z10 = -M1 + M2 + M5 + M6
    # Z11 = M1 - M3 + M4 + M7
    
    def add_strassen_block(u_offset, v_offset, w_offset, col_offset):
        # u_offset: index of X00 in U (0-3 relative to 2x2 block)
        # v_offset: index of Y00 in V (0-3 relative to 2x2 block)
        # w_offset: index of Z00 in W (0-3 relative to 2x2 block)
        
        # M1
        U[u_offset, col_offset] += 1       # X00
        U[u_offset+3, col_offset] += 1     # X11
        V[v_offset, col_offset] += 1       # Y00
        V[v_offset+3, col_offset] += 1     # Y11
        W[w_offset, col_offset] += 1       # Z00
        W[w_offset+1, col_offset] += 0     # Z01
        W[w_offset+2, col_offset] += -1    # Z10
        W[w_offset+3, col_offset] += 1     # Z11
        
        # M2
        U[u_offset+2, col_offset+1] += 1   # X10
        U[u_offset+3, col_offset+1] += 1   # X11
        V[v_offset+1, col_offset+1] += 1   # Y01
        W[w_offset+1, col_offset+1] += 1   # Z01
        W[w_offset+2, col_offset+1] += 1   # Z10
        
        # M3
        U[u_offset, col_offset+2] += 1     # X00
        V[v_offset+2, col_offset+2] += 1   # Y10
        V[v_offset, col_offset+2] += -1    # Y00
        W[w_offset+1, col_offset+2] += 1   # Z01
        W[w_offset+3, col_offset+2] += -1  # Z11
        
        # M4
        U[u_offset+3, col_offset+3] += 1   # X11
        V[v_offset+1, col_offset+3] += 1   # Y01
        V[v_offset+2, col_offset+3] += -1  # Y10
        W[w_offset, col_offset+3] += 1     # Z00
        W[w_offset+1, col_offset+3] += -1  # Z01
        W[w_offset+3, col_offset+3] += 1   # Z11
        
        # M5
        U[u_offset, col_offset+4] += 1     # X00
        U[u_offset+1, col_offset+4] += 1   # X01
        V[v_offset+3, col_offset+4] += 1   # Y11
        W[w_offset, col_offset+4] += -1    # Z00
        W[w_offset+2, col_offset+4] += 1   # Z10
        
        # M6
        U[u_offset+2, col_offset+5] += 1   # X10
        U[u_offset, col_offset+5] += -1    # X00
        V[v_offset, col_offset+5] += 1     # Y00
        V[v_offset+1, col_offset+5] += 1   # Y01
        W[w_offset+1, col_offset+5] += 1   # Z01
        W[w_offset+2, col_offset+5] += 1   # Z10
        
        # M7
        U[u_offset+1, col_offset+6] += 1   # X01
        U[u_offset+3, col_offset+6] += -1  # X11
        V[v_offset+2, col_offset+6] += 1   # Y10
        V[v_offset+3, col_offset+6] += 1   # Y11
        W[w_offset, col_offset+6] += 1     # Z00
        W[w_offset+3, col_offset+6] += 1   # Z11

    # Block 1: A_L (cols 0,1) * B_T1 (rows 0,1, cols 0,1)
    # A_L is A[0:2, 0:2] -> U indices 0,1,2,3 (row-major: 00,01,10,11)
    # B_T1 is B[0:2, 0:2] -> V indices 0,1,2,3
    # Result is C[0:2, 0:2] -> W indices 0,1,2,3
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: A_L (cols 0,1) * B_T2 (rows 0,1, cols 2,3)
    # A_L same -> U indices 0,1,2,3
    # B_T2 is B[0:2, 2:4] -> V indices 4,5,6,7 (row 0: cols 2,3; row 1: cols 2,3)
    # V index for B[j,k] is j*p + k.
    # B_T1: j=0,1; k=0,1. V idx: 0,1, 4,5? No.
    # V shape (m*p, rank) = (16, 28).
    # V index = j*p + k.
    # B_T1 (j=0,1; k=0,1):
    # j=0,k=0 -> 0
    # j=0,k=1 -> 1
    # j=1,k=0 -> 4
    # j=1,k=1 -> 5
    # So V offsets for B_T1 are 0, 1, 4, 5.
    # My add_strassen_block assumes contiguous 0,1,2,3.
    # I need to pass explicit indices or modify the function.
    
    # Let's rewrite add_strassen_block to take lists of indices.
    
    def add_strassen_block_indices(u_idx, v_idx, w_idx, col_offset):
        # u_idx: [u00, u01, u10, u11]
        # v_idx: [v00, v01, v10, v11]
        # w_idx: [w00, w01, w10, w11]
        
        # M1
        U[u_idx[0], col_offset] += 1
        U[u_idx[3], col_offset] += 1
        V[v_idx[0], col_offset] += 1
        V[v_idx[3], col_offset] += 1
        W[w_idx[0], col_offset] += 1
        W[w_idx[2], col_offset] += -1
        W[w_idx[3], col_offset] += 1
        
        # M2
        U[u_idx[2], col_offset+1] += 1
        U[u_idx[3], col_offset+1] += 1
        V[v_idx[1], col_offset+1] += 1
        W[w_idx[1], col_offset+1] += 1
        W[w_idx[2], col_offset+1] += 1
        
        # M3
        U[u_idx[0], col_offset+2] += 1
        V[v_idx[2], col_offset+2] += 1
        V[v_idx[0], col_offset+2] += -1
        W[w_idx[1], col_offset+2] += 1
        W[w_idx[3], col_offset+2] += -1
        
        # M4
        U[u_idx[3], col_offset+3] += 1
        V[v_idx[1], col_offset+3] += 1
        V[v_idx[2], col_offset+3] += -1
        W[w_idx[0], col_offset+3] += 1
        W[w_idx[1], col_offset+3] += -1
        W[w_idx[3], col_offset+3] += 1
        
        # M5
        U[u_idx[0], col_offset+4] += 1
        U[u_idx[1], col_offset+4] += 1
        V[v_idx[3], col_offset+4] += 1
        W[w_idx[0], col_offset+4] += -1
        W[w_idx[2], col_offset+4] += 1
        
        # M6
        U[u_idx[2], col_offset+5] += 1
        U[u_idx[0], col_offset+5] += -1
        V[v_idx[0], col_offset+5] += 1
        V[v_idx[1], col_offset+5] += 1
        W[w_idx[1], col_offset+5] += 1
        W[w_idx[2], col_offset+5] += 1
        
        # M7
        U[u_idx[1], col_offset+6] += 1
        U[u_idx[3], col_offset+6] += -1
        V[v_idx[2], col_offset+6] += 1
        V[v_idx[3], col_offset+6] += 1
        W[w_idx[0], col_offset+6] += 1
        W[w_idx[3], col_offset+6] += 1

    # Define indices
    # A (2x4) -> U (8x28). Index i*m + j.
    # A_L (cols 0,1):
    # A00: 0*4+0 = 0
    # A01: 0*4+1 = 1
    # A10: 1*4+0 = 4
    # A11: 1*4+1 = 5
    u_L = [0, 1, 4, 5]
    
    # A_R (cols 2,3):
    # A02: 0*4+2 = 2
    # A03: 0*4+3 = 3
    # A12: 1*4+2 = 6
    # A13: 1*4+3 = 7
    u_R = [2, 3, 6, 7]
    
    # B (4x4) -> V (16x28). Index j*p + k.
    # B_T1 (rows 0,1; cols 0,1):
    # B00: 0*4+0 = 0
    # B01: 0*4+1 = 1
    # B10: 1*4+0 = 4
    # B11: 1*4+1 = 5
    v_T1 = [0, 1, 4, 5]
    
    # B_T2 (rows 0,1; cols 2,3):
    # B02: 0*4+2 = 2
    # B03: 0*4+3 = 3
    # B12: 1*4+2 = 6
    # B13: 1*4+3 = 7
    v_T2 = [2, 3, 6, 7]
    
    # B_B1 (rows 2,3; cols 0,1):
    # B20: 2*4+0 = 8
    # B21: 2*4+1 = 9
    # B30: 3*4+0 = 12
    # B31: 3*4+1 = 13
    v_B1 = [8, 9, 12, 13]
    
    # B_B2 (rows 2,3; cols 2,3):
    # B22: 2*4+2 = 10
    # B23: 2*4+3 = 11
    # B32: 3*4+2 = 14
    # B33: 3*4+3 = 15
    v_B2 = [10, 11, 14, 15]
    
    # C (4x2) -> W (8x28). Index k*n + i.
    # C_T1 (cols 0,1; rows 0,1):
    # C00: 0*2+0 = 0
    # C01: 0*2+1 = 1
    # C10: 1*2+0 = 2
    # C11: 1*2+1 = 3
    w_T1 = [0, 1, 2, 3]
    
    # C_T2 (cols 2,3; rows 0,1):
    # C02: 2*2+0 = 4
    # C03: 2*2+1 = 5
    # C12: 3*2+0 = 6
    # C13: 3*2+1 = 7
    w_T2 = [4, 5, 6, 7]
    
    # Block 1: A_L * B_T1 -> C_T1
    add_strassen_block_indices(u_L, v_T1, w_T1, 0)
    
    # Block 2: A_L * B_T2 -> C_T2
    add_strassen_block_indices(u_L, v_T2, w_T2, 7)
    
    # Block 3: A_R * B_B1 -> C_T1
    add_strassen_block_indices(u_R, v_B1, w_T1, 14)
    
    # Block 4: A_R * B_B2 -> C_T2
    add_strassen_block_indices(u_R, v_B2, w_T2, 21)
    
    return {"U": U, "V": V, "W": W}

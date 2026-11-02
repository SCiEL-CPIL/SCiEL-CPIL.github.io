import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 14
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:02, 3:03, 4:10, 5:11, 6:12, 7:13
    # C indices: 0:00, 1:10, 2:01, 3:11, 4:02, 5:12, 6:03, 7:13
    
    # Helper to add a Strassen block
    # offset: rank offset (0 or 7)
    # b_col_offset: 0 for left block, 2 for right block
    def add_strassen_block(offset, b_col_offset):
        # B indices for this block
        b00 = b_col_offset
        b01 = b_col_offset + 1
        b10 = b_col_offset + 4
        b11 = b_col_offset + 5
        
        # C indices for this block
        c00 = b_col_offset * n
        c01 = (b_col_offset + 1) * n
        c10 = b_col_offset * n + 1
        c11 = (b_col_offset + 1) * n + 1
        
        # P1 = (A00 + A11)(B00 + B11)
        r = offset + 0
        U[0, r] = 1; U[3, r] = 1
        V[b00, r] = 1; V[b11, r] = 1
        W[c00, r] = 1; W[c11, r] = 1
        
        # P2 = (A10 + A11)B00
        r = offset + 1
        U[2, r] = 1; U[3, r] = 1
        V[b00, r] = 1
        W[c10, r] = 1
        
        # P3 = A00(B01 - B11)
        r = offset + 2
        U[0, r] = 1
        V[b01, r] = 1; V[b11, r] = -1
        W[c01, r] = 1; W[c11, r] = -1
        
        # P4 = A11(B10 - B00)
        r = offset + 3
        U[3, r] = 1
        V[b10, r] = 1; V[b00, r] = -1
        W[c00, r] = -1; W[c10, r] = 1
        
        # P5 = (A00 + A01)B11
        r = offset + 4
        U[0, r] = 1; U[1, r] = 1
        V[b11, r] = 1
        W[c01, r] = 1; W[c11, r] = 1
        
        # P6 = (A01 - A10)(B00 + B01)
        r = offset + 5
        U[1, r] = 1; U[2, r] = -1
        V[b00, r] = 1; V[b01, r] = 1
        # P6 is not used in the final sum for standard Strassen reconstruction
        # Wait, let's re-verify the standard Strassen reconstruction.
        # C00 = P1 - P4 + P7
        # C01 = P3 + P5 + P7
        # C10 = P2 + P4
        # C11 = P1 - P3 + P5
        # Yes, P6 is not in the sum. So W coefficients for P6 are 0.
        
        # P7 = (A10 - A00)(B11 + B01)
        r = offset + 6
        U[2, r] = 1; U[0, r] = -1
        V[b11, r] = 1; V[b01, r] = 1
        W[c00, r] = 1; W[c01, r] = 1

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

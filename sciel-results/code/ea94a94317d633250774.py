import numpy as np

def construct():
    n, m, p = 2, 4, 5
    # We construct two blocks of 2x2x5
    # Each 2x2x5 is sum of 2x2x2 (Strassen, rank 7) and 2x2x3 (Schoolbook, rank 12)
    # Total rank per block = 19
    # Total rank = 38
    
    rank = 38
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a 2x2x2 Strassen block
    # Indices for A: i in {0,1}, j in {0,1}
    # Indices for B: j in {0,1}, k in {0,1}
    # Indices for C: i in {0,1}, k in {0,1}
    # Offset in rank: r_offset
    # Offset in j for A/B: j_offset (0 or 2)
    
    def add_strassen_block(r_offset, j_offset):
        # U: (i, j)
        # V: (j, k)
        # W: (k, i)
        
        # M1
        U[0*m + (j_offset+0), r_offset+0] = 1
        U[1*m + (j_offset+1), r_offset+0] = 1
        V[(j_offset+0)*p + 0, r_offset+0] = 1
        V[(j_offset+1)*p + 1, r_offset+0] = 1
        W[0*n + 0, r_offset+0] = 1 # k=0, i=0
        W[1*n + 1, r_offset+0] = 1 # k=1, i=1
        
        # M2
        U[1*m + (j_offset+0), r_offset+1] = 1
        U[1*m + (j_offset+1), r_offset+1] = 1
        V[(j_offset+0)*p + 0, r_offset+1] = 1
        W[0*n + 1, r_offset+1] = 1 # k=0, i=1
        W[1*n + 1, r_offset+1] = -1 # k=1, i=1
        
        # M3
        U[0*m + (j_offset+0), r_offset+2] = 1
        V[(j_offset+1)*p + 0, r_offset+2] = 1
        V[(j_offset+1)*p + 1, r_offset+2] = -1
        W[1*n + 0, r_offset+2] = 1 # k=1, i=0
        W[1*n + 1, r_offset+2] = 1 # k=1, i=1
        
        # M4
        U[1*m + (j_offset+1), r_offset+3] = 1
        V[(j_offset+0)*p + 1, r_offset+3] = 1
        V[(j_offset+0)*p + 0, r_offset+3] = -1
        W[0*n + 0, r_offset+3] = 1 # k=0, i=0
        W[0*n + 1, r_offset+3] = 1 # k=0, i=1
        
        # M5
        U[0*m + (j_offset+0), r_offset+4] = 1
        U[0*m + (j_offset+1), r_offset+4] = 1
        V[(j_offset+1)*p + 1, r_offset+4] = 1
        W[0*n + 0, r_offset+4] = -1 # k=0, i=0
        W[1*n + 0, r_offset+4] = 1 # k=1, i=0
        
        # M6
        U[1*m + (j_offset+0), r_offset+5] = 1
        U[0*m + (j_offset+0), r_offset+5] = -1
        V[(j_offset+0)*p + 0, r_offset+5] = 1
        V[(j_offset+1)*p + 0, r_offset+5] = 1
        W[1*n + 1, r_offset+5] = 1 # k=1, i=1
        
        # M7
        U[0*m + (j_offset+1), r_offset+6] = 1
        U[1*m + (j_offset+1), r_offset+6] = -1
        V[(j_offset+1)*p + 1, r_offset+6] = 1
        V[(j_offset+0)*p + 1, r_offset+6] = 1
        W[0*n + 0, r_offset+6] = 1 # k=0, i=0

    def add_schoolbook_223_block(r_offset, j_offset, k_offset):
        # 2x2x3 schoolbook
        # A: 2x2, B: 2x3
        # 12 terms
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    idx = r_offset + i*6 + j*3 + k
                    U[i*m + (j_offset+j), idx] = 1
                    V[(j_offset+j)*p + (k_offset+k), idx] = 1
                    W[(k_offset+k)*n + i, idx] = 1

    # Block 1: j in {0,1}
    # k in {0,1} for Strassen part
    add_strassen_block(0, 0)
    # k in {2,3,4} for Schoolbook part
    add_schoolbook_223_block(7, 0, 2)
    
    # Block 2: j in {2,3}
    # k in {0,1} for Strassen part
    add_strassen_block(19, 2)
    # k in {2,3,4} for Schoolbook part
    add_schoolbook_223_block(26, 2, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a rank-42 tensor decomposition for multiplying a 2x2 matrix by a 2x12 matrix.
    
    The approach uses a Strassen-like decomposition of the 2x2 matrix multiplication,
    applied to each 2-column block of the 12-column matrix B.
    
    For 2x2 * 2x2, Strassen's algorithm uses 7 multiplications:
    M1 = (A00+A11)*(B00+B11)
    M2 = A00*B10
    M3 = (A10-A00)*(B00+B01)
    M4 = A11*B11
    M5 = (A00+A01)*B10
    M6 = (A11-A01)*B01
    M7 = (A01-A10)*(B10+B11)
    
    Then:
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For 2x2 * 2x12, we split B into six 2-column blocks and apply this to each,
    giving 6 * 7 = 42 rank-1 terms.
    """
    n, m, p = DIMENSIONS  # 2, 2, 12
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=np.int64)  # 4 x 42
    V = np.zeros((m * p, rank), dtype=np.int64)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=np.int64)  # 24 x 42
    
    # Strassen patterns for 2x2 * 2x2
    # U rows: A00=0, A10=1, A01=2, A11=3
    # V rows: B00=0, B10=1, B01=2, B11=3
    # W rows: C00=0, C10=1, C01=2, C11=3
    
    # 7 Strassen multiplications with their U, V, W patterns
    # Each entry is (u_pattern, v_pattern, w_pattern) where patterns are lists of (row, coeff)
    strassen_terms = [
        # M1 = (A00+A11)*(B00+B11)
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (3, 1)],
            'W': [(0, 1), (3, 1)],  # contributes to C00 and C11
        },
        # M2 = A00*B10
        {
            'U': [(0, 1)],
            'V': [(1, 1)],
            'W': [(1, 1), (3, -1)],  # contributes to C10 and C11
        },
        # M3 = (A10-A00)*(B00+B01)
        {
            'U': [(1, 1), (0, -1)],
            'V': [(0, 1), (2, 1)],
            'W': [(2, 1), (3, 1)],  # contributes to C01 and C11
        },
        # M4 = A11*B11
        {
            'U': [(3, 1)],
            'V': [(3, 1)],
            'W': [(0, -1), (1, 1)],  # contributes to C00 and C10
        },
        # M5 = (A00+A01)*B10
        {
            'U': [(0, 1), (2, 1)],
            'V': [(1, 1)],
            'W': [(0, 1), (2, 1)],  # contributes to C00 and C01
        },
        # M6 = (A11-A01)*B01
        {
            'U': [(3, 1), (2, -1)],
            'V': [(2, 1)],
            'W': [(3, 1)],  # contributes to C11
        },
        # M7 = (A01-A10)*(B10+B11)
        {
            'U': [(2, 1), (1, -1)],
            'V': [(1, 1), (3, 1)],
            'W': [(0, -1)],  # contributes to C00
        },
    ]
    
    # For each of the 6 column blocks in B (each block has 2 columns),
    # assign 7 rank-1 terms
    for block in range(6):
        for t_idx in range(7):
            col = block * 7 + t_idx
            term = strassen_terms[t_idx]
            
            # Fill U (same for all blocks)
            for row, coeff in term['U']:
                U[row, col] = coeff
            
            # Fill V (depends on block)
            # V row indices: B00=0, B10=12, B01=1, B11=13 for block 0
            # In general for block b: B00=12b, B10=12b+12, B01=12b+1, B11=12b+13
            # Wait, let me re-examine the indexing.
            # V has shape (m*p, rank) = (2*12, rank) = (24, rank)
            # V index j*p + k where j is row of B (0 or 1), k is column of B (0..11)
            # So B[0, 2b] is at index 0*12 + 2b = 2b
            # B[0, 2b+1] is at index 0*12 + 2b+1 = 2b+1
            # B[1, 2b] is at index 1*12 + 2b = 12+2b
            # B[1, 2b+1] is at index 1*12 + 2b+1 = 12+2b+1
            
            # So for block b:
            # B00 -> index 2b
            # B10 -> index 12+2b
            # B01 -> index 2b+1
            # B11 -> index 12+2b+1
            
            v_index_map = {0: 2*block, 1: 12+2*block, 2: 2*block+1, 3: 12+2*block+1}
            for row, coeff in term['V']:
                V[v_index_map[row], col] = coeff
            
            # Fill W (depends on block)
            # W has shape (p*n, rank) = (12*2, rank) = (24, rank)
            # W index k*n + i where k is column of C (0..11), i is row of C (0 or 1)
            # C[0, 2b] is at index 2b*2 + 0 = 4b
            # C[1, 2b] is at index 2b*2 + 1 = 4b+1
            # C[0, 2b+1] is at index (2b+1)*2 + 0 = 4b+2
            # C[1, 2b+1] is at index (2b+1)*2 + 1 = 4b+3
            
            # So for block b:
            # C00 -> index 4b
            # C10 -> index 4b+1
            # C01 -> index 4b+2
            # C11 -> index 4b+3
            
            w_index_map = {0: 4*block, 1: 4*block+1, 2: 4*block+2, 3: 4*block+3}
            for row, coeff in term['W']:
                W[w_index_map[row], col] = coeff
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

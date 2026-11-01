import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition approach.
    # Split the 11 columns into blocks of size 2, except the last one which is size 1.
    # Blocks: 5 blocks of size 2, 1 block of size 1.
    # For a 2x2 * 2x2 block, we use Strassen's algorithm (rank 7).
    # For a 2x2 * 2x1 block, we use the trivial algorithm (rank 4).
    # Total rank = 5 * 7 + 1 * 4 = 35 + 4 = 39.
    
    rank = 5 * 7 + 4
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add Strassen decomposition for a 2x2 * 2x2 block
    # Indices for A (2x2): (0,0), (0,1), (1,0), (1,1) -> flat 0,1,2,3
    # Indices for B (2x2) starting at col_offset: (0,0), (0,1), (1,0), (1,1) -> flat col_offset + 0, 1, 2, 3
    # Indices for C (2x2) starting at row_offset: (0,0), (0,1), (1,0), (1,1) -> flat row_offset + 0, 1, 2, 3
    def add_strassen(col_offset, row_offset):
        nonlocal col
        # Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1
        U[0, col] = 1; U[3, col] = 1
        V[col_offset + 0, col] = 1; V[col_offset + 3, col] = 1
        W[row_offset + 0, col] = 1; W[row_offset + 3, col] = 1
        col += 1
        
        # M2
        U[2, col] = 1; U[3, col] = 1
        V[col_offset + 0, col] = 1
        W[row_offset + 2, col] = 1; W[row_offset + 3, col] = 1
        col += 1
        
        # M3
        U[0, col] = 1
        V[col_offset + 2, col] = 1; V[col_offset + 3, col] = -1
        W[row_offset + 1, col] = 1; W[row_offset + 3, col] = 1
        col += 1
        
        # M4
        U[3, col] = 1
        V[col_offset + 1, col] = 1; V[col_offset + 0, col] = -1
        W[row_offset + 0, col] = 1; W[row_offset + 2, col] = 1
        col += 1
        
        # M5
        U[0, col] = 1; U[1, col] = 1
        V[col_offset + 3, col] = 1
        W[row_offset + 0, col] = -1; W[row_offset + 1, col] = 1
        col += 1
        
        # M6
        U[2, col] = 1; U[0, col] = -1
        V[col_offset + 0, col] = 1; V[col_offset + 1, col] = 1
        W[row_offset + 3, col] = 1
        col += 1
        
        # M7
        U[1, col] = 1; U[3, col] = -1
        V[col_offset + 2, col] = 1; V[col_offset + 3, col] = 1
        W[row_offset + 0, col] = 1
        col += 1

    # Helper to add trivial decomposition for a 2x2 * 2x1 block
    # A (2x2): 0,1,2,3
    # B (2x1) starting at col_offset: (0,0), (1,0) -> flat col_offset + 0, 1
    # C (2x1) starting at row_offset: (0,0), (1,0) -> flat row_offset + 0, 1
    def add_trivial_2x2_2x1(col_offset, row_offset):
        nonlocal col
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        
        # Term 1: A00 * B00 -> C00
        U[0, col] = 1
        V[col_offset + 0, col] = 1
        W[row_offset + 0, col] = 1
        col += 1
        
        # Term 2: A01 * B10 -> C00
        U[1, col] = 1
        V[col_offset + 1, col] = 1
        W[row_offset + 0, col] = 1
        col += 1
        
        # Term 3: A10 * B00 -> C10
        U[2, col] = 1
        V[col_offset + 0, col] = 1
        W[row_offset + 1, col] = 1
        col += 1
        
        # Term 4: A11 * B10 -> C10
        U[3, col] = 1
        V[col_offset + 1, col] = 1
        W[row_offset + 1, col] = 1
        col += 1

    # Process blocks
    for i in range(5):
        # Block i corresponds to columns 2*i and 2*i+1 of B
        # And rows 2*i and 2*i+1 of C
        add_strassen(2 * i, 2 * i)
        
    # Last block corresponds to column 10 of B and row 10 of C
    add_trivial_2x2_2x1(10, 10)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

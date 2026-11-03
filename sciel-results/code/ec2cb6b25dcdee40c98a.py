import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Split indices into two blocks: 0 (size 2) and 1 (size 3)
    # Block 0: indices 0, 1
    # Block 1: indices 2, 3, 4
    
    # Total rank estimate:
    # A00 B00 -> C00: 2x2x2 (Strassen, 7)
    # A01 B10 -> C00: 2x3x2 (Schoolbook, 12)
    # A00 B01 -> C01: 2x2x3 (Schoolbook, 12)
    # A01 B11 -> C01: 2x3x3 (Schoolbook, 18)
    # A10 B00 -> C10: 3x2x2 (Schoolbook, 12)
    # A11 B10 -> C10: 3x3x2 (Schoolbook, 18)
    # A10 B01 -> C11: 3x2x3 (Schoolbook, 18)
    # A11 B11 -> C11: 3x3x3 (Schoolbook, 27)
    
    rank = 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Helper to add schoolbook decomposition for a rectangular block
    def add_schoolbook(row_start, row_end, col_start, col_end, out_row_start, out_row_end, out_col_start, out_col_end):
        nonlocal col
        for i in range(row_start, row_end):
            for k in range(out_col_start, out_col_end):
                # A[i, k] * B[k, j] -> C[i, j]
                # Here we are decomposing A_block * B_block -> C_block
                # A indices: i (row), k (col of A)
                # B indices: k (row of B), j (col of B)
                # C indices: i (row), j (col)
                
                # Wait, the standard schoolbook for C = A * B is:
                # C[i, j] = sum_k A[i, k] B[k, j]
                # Tensor term: A[i, k] B[k, j] -> C[i, j]
                
                # In our block context:
                # A_block is rows [row_start, row_end), cols [col_start, col_end)
                # B_block is rows [col_start, col_end), cols [out_col_start, out_col_end)
                # C_block is rows [row_start, row_end), cols [out_col_start, out_col_end)
                
                # We need to iterate over the "inner" dimension k
                # k ranges from col_start to col_end
                
                for k in range(col_start, col_end):
                    for j in range(out_col_start, out_col_end):
                        U[i*m + k, col] = 1
                        V[k*p + j, col] = 1
                        W[j*n + i, col] = 1
                        col += 1

    # 1. A00 B00 -> C00 (2x2x2)
    # Use Strassen
    # A00: rows 0,1 cols 0,1
    # B00: rows 0,1 cols 0,1
    # C00: rows 0,1 cols 0,1
    
    # Strassen implementation
    # M1 = (A11 + A22)(B11 + B22)
    U[0*m + 0, col] = 1; U[1*m + 1, col] = 1
    V[0*p + 0, col] = 1; V[1*p + 1, col] = 1
    W[0*n + 0, col] = 1; W[1*n + 1, col] = 1
    col += 1
    
    # M2 = (A21 + A22)B11
    U[1*m + 0, col] = 1; U[1*m + 1, col] = 1
    V[0*p + 0, col] = 1
    W[0*n + 1, col] = 1
    col += 1
    
    # M3 = A11(B12 - B22)
    U[0*m + 0, col] = 1
    V[0*p + 1, col] = 1; V[1*p + 1, col] = -1
    W[1*n + 0, col] = 1
    col += 1
    
    # M4 = A22(B21 - B11)
    U[1*m + 1, col] = 1
    V[1*p + 0, col] = 1; V[0*p + 0, col] = -1
    W[0*n + 1, col] = 1
    col += 1
    
    # M5 = (A11 + A12)B22
    U[0*m + 0, col] = 1; U[0*m + 1, col] = 1
    V[1*p + 1, col] = 1
    W[1*n + 0, col] = 1
    col += 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[1*m + 0, col] = 1; U[0*m + 0, col] = -1
    V[0*p + 0, col] = 1; V[0*p + 1, col] = 1
    W[0*n + 1, col] = 1
    col += 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[0*m + 1, col] = 1; U[1*m + 1, col] = -1
    V[1*p + 0, col] = 1; V[1*p + 1, col] = 1
    W[1*n + 0, col] = 1
    col += 1
    
    # 2. A01 B10 -> C00 (2x3x2)
    # A01: rows 0,1 cols 2,3,4
    # B10: rows 2,3,4 cols 0,1
    # C00: rows 0,1 cols 0,1
    add_schoolbook(0, 2, 2, 5, 0, 2, 0, 2)
    
    # 3. A00 B01 -> C01 (2x2x3)
    # A00: rows 0,1 cols 0,1
    # B01: rows 0,1 cols 2,3,4
    # C01: rows 0,1 cols 2,3,4
    add_schoolbook(0, 2, 0, 2, 0, 2, 2, 5)
    
    # 4. A01 B11 -> C01 (2x3x3)
    # A01: rows 0,1 cols 2,3,4
    # B11: rows 2,3,4 cols 2,3,4
    # C01: rows 0,1 cols 2,3,4
    add_schoolbook(0, 2, 2, 5, 0, 2, 2, 5)
    
    # 5. A10 B00 -> C10 (3x2x2)
    # A10: rows 2,3,4 cols 0,1
    # B00: rows 0,1 cols 0,1
    # C10: rows 2,3,4 cols 0,1
    add_schoolbook(2, 5, 0, 2, 2, 5, 0, 2)
    
    # 6. A11 B10 -> C10 (3x3x2)
    # A11: rows 2,3,4 cols 2,3,4
    # B10: rows 2,3,4 cols 0,1
    # C10: rows 2,3,4 cols 0,1
    add_schoolbook(2, 5, 2, 5, 2, 5, 0, 2)
    
    # 7. A10 B01 -> C11 (3x2x3)
    # A10: rows 2,3,4 cols 0,1
    # B01: rows 0,1 cols 2,3,4
    # C11: rows 2,3,4 cols 2,3,4
    add_schoolbook(2, 5, 0, 2, 2, 5, 2, 5)
    
    # 8. A11 B11 -> C11 (3x3x3)
    # A11: rows 2,3,4 cols 2,3,4
    # B11: rows 2,3,4 cols 2,3,4
    # C11: rows 2,3,4 cols 2,3,4
    add_schoolbook(2, 5, 2, 5, 2, 5, 2, 5)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

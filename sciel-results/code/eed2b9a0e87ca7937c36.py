import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 34.
    # Split p=6 into two blocks of p=3.
    # For each block, we use a rank-17 decomposition for 2x3x3 multiplication.
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, val=1):
        nonlocal col
        U[i * m + j][col] = val
        V[j * p + k][col] = val
        W[k * n + i][col] = val
        col += 1

    # Block 1: k in 0..2
    # A is 2x3, B is 3x3 (cols 0-2)
    # Split A into A1 (2x2) and A2 (2x1)
    # Split B into B1 (2x3) and B2 (1x3)
    # C = A1 B1 + A2 B2
    
    # A2 B2: 2x1 x 1x3 -> Rank 6
    # A2 rows: 0,1. Col 2.
    # B2 row: 2. Cols 0,1,2.
    for i in range(2):
        for k in range(3):
            add_term(i, 2, k)
            
    # A1 B1: 2x2 x 2x3
    # Split B1 into B11 (2x1) and B12 (2x2)
    # A1 B11: 2x2 x 2x1 -> Rank 4
    for i in range(2):
        for j in range(2):
            add_term(i, j, 0)
            
    # A1 B12: 2x2 x 2x2 -> Rank 7 (Strassen)
    # Strassen for 2x2 matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = A11(B01-B00)
    # M3 = (A00-A11)(B10+B11)
    # M4 = A00(B11-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)B00
    # M7 = (A01-A11)(B00+B10)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M5
    # C10 = M3 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices in this sub-block:
    # A1 rows 0,1. Cols 0,1.
    # B12 rows 0,1. Cols 1,2.
    # Output C12 cols 1,2.
    
    # We need to map Strassen terms to U, V, W.
    # Strassen terms are linear combinations.
    # We will add 7 terms.
    
    # M1: (A00+A11)(B00+B11)
    # U: A00, A11. V: B00, B11. W: C00, C01, C10, C11.
    # This is not a single rank-1 term. It's a sum.
    # The decomposition requires U, V, W to be rank-1.
    # So we must expand the Strassen algorithm into its linear components?
    # No, Strassen algorithm *is* a decomposition of rank 7.
    # The U, V, W matrices for Strassen 2x2x2 have specific entries.
    
    # Let's manually construct the 7 terms for A1 B12.
    # A1 elements: a00, a01, a10, a11 (indices i=0,1; j=0,1)
    # B12 elements: b00, b01, b10, b11 (indices j=0,1; k=1,2)
    # Note: B12 cols are 1,2 of the original B.
    
    # Term 1: M1
    U[0*m+0][col] = 1; U[1*m+1][col] = 1
    V[0*p+1][col] = 1; V[1*p+2][col] = 1
    W[1*n+0][col] = 1; W[2*n+0][col] = 1; W[1*n+1][col] = 1; W[2*n+1][col] = 1
    # Wait, W indices are k*n+i.
    # k=1, i=0 -> W[1*2+0] = W[2]
    # k=2, i=0 -> W[2*2+0] = W[4]
    # k=1, i=1 -> W[1*2+1] = W[3]
    # k=2, i=1 -> W[2*2+1] = W[5]
    
    # This is getting error-prone. I will use a known rank-7 decomposition for 2x2x2.
    # Standard Strassen U, V, W for 2x2x2:
    # U: 4x7, V: 4x7, W: 4x7.
    # I will define the 7 terms explicitly.
    
    def add_strassen_2x2x2(base_i, base_j, base_k):
        # base_i, base_j, base_k are offsets for A, B, C
        # A is 2x2, B is 2x2, C is 2x2
        # U indices: (base_i+i)*m + (base_j+j)
        # V indices: (base_j+j)*p + (base_k+k)
        # W indices: (base_k+k)*n + (base_i+i)
        
        # M1 = (A00+A11)(B00+B11)
        U[(base_i+0)*m + base_j+0][col] += 1
        U[(base_i+1)*m + base_j+1][col] += 1
        V[(base_j+0)*p + base_k+0][col] += 1
        V[(base_j+1)*p + base_k+1][col] += 1
        W[(base_k+0)*n + base_i+0][col] += 1
        W[(base_k+1)*n + base_i+0][col] += 1
        W[(base_k+0)*n + base_i+1][col] += 1
        W[(base_k+1)*n + base_i+1][col] += 1
        
        # M2 = A11(B01-B00)
        U[(base_i+1)*m + base_j+1][col+1] += 1
        V[(base_j+0)*p + base_k+1][col+1] += 1
        V[(base_j+0)*p + base_k+0][col+1] += -1
        W[(base_k+1)*n + base_i+0][col+1] += 1
        W[(base_k+1)*n + base_i+1][col+1] += 1
        
        # M3 = (A00-A11)(B10+B11)
        U[(base_i+0)*m + base_j+0][col+2] += 1
        U[(base_i+1)*m + base_j+1][col+2] += -1
        V[(base_j+1)*p + base_k+0][col+2] += 1
        V[(base_j+1)*p + base_k+1][col+2] += 1
        W[(base_k+0)*n + base_i+1][col+2] += 1
        W[(base_k+1)*n + base_i+1][col+2] += 1
        W[(base_k+0)*n + base_i+0][col+2] += 1
        W[(base_k+1)*n + base_i+0][col+2] += 1
        
        # M4 = A00(B11-B10)
        U[(base_i+0)*m + base_j+0][col+3] += 1
        V[(base_j+1)*p + base_k+1][col+3] += 1
        V[(base_j+1)*p + base_k+0][col+3] += -1
        W[(base_k+1)*n + base_i+0][col+3] += 1
        W[(base_k+0)*n + base_i+1][col+3] += 1
        
        # M5 = (A00+A01)B11
        U[(base_i+0)*m + base_j+0][col+4] += 1
        U[(base_i+0)*m + base_j+1][col+4] += 1
        V[(base_j+1)*p + base_k+1][col+4] += 1
        W[(base_k+1)*n + base_i+0][col+4] += -1
        W[(base_k+1)*n + base_i+1][col+4] += 1
        
        # M6 = (A10-A00)B00
        U[(base_i+1)*m + base_j+0][col+5] += 1
        U[(base_i+0)*m + base_j+0][col+5] += -1
        V[(base_j+0)*p + base_k+0][col+5] += 1
        W[(base_k+0)*n + base_i+1][col+5] += 1
        
        # M7 = (A01-A11)(B00+B10)
        U[(base_i+0)*m + base_j+1][col+6] += 1
        U[(base_i+1)*m + base_j+1][col+6] += -1
        V[(base_j+0)*p + base_k+0][col+6] += 1
        V[(base_j+1)*p + base_k+0][col+6] += 1
        W[(base_k+0)*n + base_i+0][col+6] += 1
        W[(base_k+0)*n + base_i+1][col+6] += -1

        col += 7

    # Apply Strassen for Block 1 (k=1,2)
    # A1 is i=0,1; j=0,1.
    # B12 is j=0,1; k=1,2.
    add_strassen_2x2x2(0, 0, 1)
    
    # Block 2: k in 3..5
    # Same structure.
    for i in range(2):
        for k in range(3, 6):
            add_term(i, 2, k)
            
    for i in range(2):
        for j in range(2):
            add_term(i, j, 3)
            
    add_strassen_2x2x2(0, 0, 3)

    return {"U": U, "V": V, "W": W}

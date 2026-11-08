import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # A: n×m, B: m×p, C: n×p
    # U: (n*m) × rank, V: (m*p) × rank, W: (p*n) × rank
    # W index: k*n + i
    
    # Strassen 2x2x2 decomposition:
    # For A (2x2), B (2x2), C = A*B (2x2):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A10)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4 * 4x6:
    # Split A (4x4) into 2x2 blocks: A00, A01 (top), A10, A11 (bottom)
    # Split B (4x6) into 2x2 blocks: B00, B01, B02 (top), B10, B11, B12 (bottom)
    # C (4x6) blocks: C00, C01, C02 (top), C10, C11, C12 (bottom)
    
    # Block structure:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # Each C block is a 2x2 matrix = sum of two 2x2*2x2 products.
    # We can apply Strassen to each 2x2*2x2 product.
    
    # Total 12 products of 2x2 matrices, each needing rank 7 = 84 total.
    
    # Let's enumerate all 12 (2x2 x 2x2) sub-problems:
    # (A_block, B_block, C_block)
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B02 (rows 0-1, cols 4-5),
    #           B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3), B12 (rows 2-3, cols 4-5)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3), C02 (rows 0-1, cols 4-5),
    #           C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3), C12 (rows 2-3, cols 4-5)
    
    # The 12 products:
    # C00: A00*B00, A01*B10
    # C01: A00*B01, A01*B11
    # C02: A00*B02, A01*B12
    # C10: A10*B00, A11*B10
    # C11: A10*B01, A11*B11
    # C12: A10*B02, A11*B12
    
    # For each 2x2*2x2 product, Strassen uses 7 rank-1 terms.
    # We need to map these to the global U, V, W indices.
    
    # Global indices:
    # U[i*m + j] for A[i,j]
    # V[j*p + k] for B[j,k]
    # W[k*n + i] for C[i,k]
    
    # Let's define the Strassen terms for a 2x2 block product.
    # For A_sub (2x2) with elements A_sub[a,b] and B_sub (2x2) with elements B_sub[c,d]:
    # In global coords: A_sub[a,b] = A[ar+a, ac+b], B_sub[c,d] = B[br+c, bc+d]
    # C_sub[x,y] = C[cr+x, cc+y]
    
    # Strassen M1 = (A00+A11)*(B00+B11)
    #   U contribution: A[ar,ac] and A[ar+1,ac+1]
    #   V contribution: B[br,bc] and B[br+1,bc+1]
    #   W contribution: C[cr,cc], C[cr,cc+1], C[cr+1,cc], C[cr+1,cc+1] with appropriate signs
    
    # Let me build this systematically.
    
    # Strassen coefficients for 2x2:
    # M1: A factors: (1,0,0,1) for (A00,A01,A10,A11); B factors: (1,0,0,1) for (B00,B01,B10,B11)
    #     C: C00+=1, C01+=0, C10+=0, C11+=1
    # M2: A: (0,0,1,1); B: (1,0,0,0)
    #     C: C00+=0, C01+=0, C10+=1, C11+=0
    # M3: A: (1,0,0,0); B: (0,1,0,-1)
    #     C: C00+=0, C01+=1, C10+=0, C11+=0
    # M4: A: (0,0,0,1); B: (0,0,1,-1)
    #     C: C00+=1, C01+=0, C10+=0, C11+=0
    # M5: A: (1,1,0,0); B: (0,0,0,1)
    #     C: C00+=-1, C01+=1, C10+=0, C11+=0
    # M6: A: (0,0,1,-1); B: (1,1,0,0)
    #     C: C00+=0, C01+=0, C10+=0, C11+=1
    # M7: A: (0,1,-1,0); B: (0,1,0,1)
    #     C: C00+=1, C01+=0, C10+=0, C11+=0
    
    # Wait, let me re-derive more carefully.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each M, the C contributions are:
    # M1: C00+=1, C01+=0, C10+=0, C11+=1
    # M2: C00+=0, C01+=0, C10+=1, C11+=-1
    # M3: C00+=0, C01+=1, C10+=0, C11+=1
    # M4: C00+=1, C01+=0, C10+=1, C11+=0
    # M5: C00+=-1, C01+=1, C10+=0, C11+=0
    # M6: C00+=0, C01+=0, C10+=0, C11+=1
    # M7: C00+=1, C01+=0, C10+=0, C11+=0
    
    # A factors for each M (coefficients of A00, A01, A10, A11):
    # M1: (1, 0, 0, 1)
    # M2: (0, 0, 1, 1)
    # M3: (1, 0, 0, 0)
    # M4: (0, 0, 0, 1)
    # M5: (1, 1, 0, 0)
    # M6: (0, 0, 1, -1)
    # M7: (0, 1, -1, 0)
    
    # B factors for each M (coefficients of B00, B01, B10, B11):
    # M1: (1, 0, 0, 1)
    # M2: (1, 0, 0, 0)
    # M3: (0, 1, 0, -1)
    # M4: (0, 0, 1, -1)
    # M5: (0, 0, 0, 1)
    # M6: (1, 1, 0, 0)
    # M7: (0, 1, 0, 1)
    
    # Now, the 12 sub-problems. Each involves a specific A 2x2 block, B 2x2 block, and contributes to a C 2x2 block.
    
    # A blocks (row_offset, col_offset) in the 4x4 A:
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    # B blocks (row_offset, col_offset) in the 4x6 B:
    # B00: (0,0), B01: (0,2), B02: (0,4), B10: (2,0), B11: (2,2), B12: (2,4)
    # C blocks (row_offset, col_offset) in the 4x6 C:
    # C00: (0,0), C01: (0,2), C02: (0,4), C10: (2,0), C11: (2,2), C12: (2,4)
    
    # The 12 products:
    products = [
        # (A_row, A_col, B_row, B_col, C_row, C_col)
        (0, 0, 0, 0, 0, 0),  # A00*B00 -> C00
        (0, 2, 2, 0, 0, 0),  # A01*B10 -> C00
        (0, 0, 0, 2, 0, 2),  # A00*B01 -> C01
        (0, 2, 2, 2, 0, 2),  # A01*B11 -> C01
        (0, 0, 0, 4, 0, 4),  # A00*B02 -> C02
        (0, 2, 2, 4, 0, 4),  # A01*B12 -> C02
        (2, 0, 0, 0, 2, 0),  # A10*B00 -> C10
        (2, 2, 2, 0, 2, 0),  # A11*B10 -> C10
        (2, 0, 0, 2, 2, 2),  # A10*B01 -> C11
        (2, 2, 2, 2, 2, 2),  # A11*B11 -> C11
        (2, 0, 0, 4, 2, 4),  # A10*B02 -> C12
        (2, 2, 2, 4, 2, 4),  # A11*B12 -> C12
    ]
    
    total_rank = 12 * 7  # 84
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen definitions
    strassen_A = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 1, -1], # M6
        [0, 1, -1, 0], # M7
    ]
    strassen_B = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 1, 0, 1],  # M7
    ]
    strassen_C = [
        [1, 0, 0, 1],   # M1: C00, C01, C10, C11
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]
    
    col = 0
    for (ar, ac, br, bc, cr, cc) in products:
        for s in range(7):  # 7 Strassen terms
            # A factors: strassen_A[s] = [coeff for A00, A01, A10, A11] in local 2x2
            # A00 global: (ar, ac), A01: (ar, ac+1), A10: (ar+1, ac), A11: (ar+1, ac+1)
            A_coeffs = strassen_A[s]
            B_coeffs = strassen_B[s]
            C_coeffs = strassen_C[s]
            
            # A elements: (row, col) in global A
            A_positions = [(ar, ac), (ar, ac+1), (ar+1, ac), (ar+1, ac+1)]
            B_positions = [(br, bc), (br, bc+1), (br+1, bc), (br+1, bc+1)]
            C_positions = [(cr, cc), (cr, cc+1), (cr+1, cc), (cr+1, cc+1)]
            
            for idx, (coeff, (r, c)) in enumerate(zip(A_coeffs, A_positions)):
                if coeff != 0:
                    U[r * m + c][col] = coeff
            
            for idx, (coeff, (r, c)) in enumerate(zip(B_coeffs, B_positions)):
                if coeff != 0:
                    V[r * p + c][col] = coeff
            
            for idx, (coeff, (r, c)) in enumerate(zip(C_coeffs, C_positions)):
                if coeff != 0:
                    # W index: k*n + i where i=r (row of C), k=c (col of C)
                    W[c * n + r][col] = coeff
            
            col += 1
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_reshaped = U_arr.reshape(n, m, total_rank)
    V_reshaped = V_arr.reshape(m, p, total_rank)
    W_reshaped = W_arr.reshape(p, n, total_rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    expected = np.ones((n, m, p))
    
    if not np.allclose(T, expected, atol=1e-9):
        # Fallback to schoolbook
        best_rank = n * m * p
        U = [[0] * best_rank for _ in range(n * m)]
        V = [[0] * best_rank for _ in range(m * p)]
        W = [[0] * best_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

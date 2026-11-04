import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(coeffs_A, coeffs_B, coeffs_C, j_offset=0):
        nonlocal col
        if col >= rank:
            return
        for (i, j), c in coeffs_A.items():
            U[i * m + (j + j_offset)][col] += c
        for (j, k), c in coeffs_B.items():
            V[(j + j_offset) * p + k][col] += c
        for (k, i), c in coeffs_C.items():
            W[k * n + i][col] += c
        col += 1

    # Strassen 2x2x2
    def strassen_222(j_offset):
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, j_offset)
        # M2 = (A10+A11)B00 -> C10, C11
        add_term({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}, j_offset) # Note: C11 gets -M2? 
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-add correctly based on standard formulas.
        # I already added M1.
        # M2
        add_term({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}, j_offset)
        # M3
        add_term({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}, j_offset)
        # M4
        add_term({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, j_offset)
        # M5
        add_term({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}, j_offset)
        # M6
        add_term({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}, j_offset)
        # M7
        add_term({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}, j_offset)

    # Schoolbook 2x2x1
    def schoolbook_221(j_offset, k_offset):
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        add_term({(0,0):1}, {(0,0):1}, {(0,0):1}, j_offset)
        add_term({(0,1):1}, {(1,0):1}, {(0,0):1}, j_offset)
        add_term({(1,0):1}, {(0,0):1}, {(0,0):1}, j_offset) # Wait, C10 is (1,0) in C? 
        # C indices are (k, i). 
        # For 2x2x1, k=0.
        # C00 is (0,0). C10 is (0,1)? No.
        # C is 2x1. C[0,0] and C[1,0].
        # In W, index is k*n + i.
        # C[0,0] -> W[0*2+0] = W[0].
        # C[1,0] -> W[0*2+1] = W[1].
        # So coeffs_C for C00 is {(0,0): 1}.
        # coeffs_C for C10 is {(0,1): 1}.
        
        # Let's re-verify W indexing.
        # W[k*n + i].
        # C[i, k] is the entry.
        # The tensor term is A[i, j] B[j, k] C[k, i].
        # So if we want to produce C[i, k], we add to W[k, i].
        
        # For 2x2x1:
        # C[0, 0] = A[0, 0] B[0, 0] + A[0, 1] B[1, 0]
        # Term 1: A[0,0] B[0,0] -> C[0,0]. W[0,0] += 1.
        # Term 2: A[0,1] B[1,0] -> C[0,0]. W[0,0] += 1.
        # C[1, 0] = A[1, 0] B[0, 0] + A[1, 1] B[1, 0]
        # Term 3: A[1,0] B[0,0] -> C[1,0]. W[0,1] += 1.
        # Term 4: A[1,1] B[1,0] -> C[1,0]. W[0,1] += 1.
        
        add_term({(0,0):1}, {(0,0):1}, {(0,0):1}, j_offset)
        add_term({(0,1):1}, {(1,0):1}, {(0,0):1}, j_offset)
        add_term({(1,0):1}, {(0,0):1}, {(0,1):1}, j_offset)
        add_term({(1,1):1}, {(1,0):1}, {(0,1):1}, j_offset)

    # We need to handle k_offset for the 2x2x1 part?
    # The 2x2x1 part is for a specific column k.
    # In my schoolbook_221, I assumed k=0.
    # I need to pass k_offset to shift the k index in V and W.
    
    def schoolbook_221_k(j_offset, k_offset):
        add_term({(0,0):1}, {(0,0):1}, {(0,0):1}, j_offset) # This is wrong, k is fixed.
        # V index is (j, k). If k is shifted, V[j, k+k_offset].
        # W index is (k, i). If k is shifted, W[k+k_offset, i].
        
        # Term 1: A[0,0] B[0, k] -> C[0, k]
        add_term({(0,0):1}, {(0,0):1}, {(0,0):1}, j_offset) # Need to modify add_term to handle k_offset?
        # Better to just pass global indices or modify add_term.
        
    # Let's modify add_term to accept k_offset for V and W.
    # Actually, it's easier to just write the indices directly.
    
    # Let's restart the construction with a cleaner approach.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_strassen(j_offset):
        nonlocal col
        # M1
        U[0*m + (0+j_offset)][col] = 1; U[0*m + (1+j_offset)][col] = 1
        V[(0+j_offset)*p + 0][col] = 1; V[(1+j_offset)*p + 1][col] = 1
        W[0*n + 0][col] = 1; W[1*n + 1][col] = 1
        col += 1
        # M2
        U[1*m + (0+j_offset)][col] = 1; U[1*m + (1+j_offset)][col] = 1
        V[(0+j_offset)*p + 0][col] = 1
        W[0*n + 1][col] = 1; W[1*n + 1][col] = -1
        col += 1
        # M3
        U[0*m + (0+j_offset)][col] = 1
        V[(0+j_offset)*p + 1][col] = 1; V[(1+j_offset)*p + 1][col] = -1
        W[1*n + 0][col] = 1
        col += 1
        # M4
        U[1*m + (1+j_offset)][col] = 1
        V[(1+j_offset)*p + 0][col] = 1; V[(0+j_offset)*p + 0][col] = -1
        W[0*n + 0][col] = 1; W[0*n + 1][col] = 1
        col += 1
        # M5
        U[0*m + (0+j_offset)][col] = 1; U[0*m + (1+j_offset)][col] = 1
        V[(1+j_offset)*p + 1][col] = 1
        W[0*n + 0][col] = -1; W[1*n + 0][col] = 1
        col += 1
        # M6
        U[1*m + (0+j_offset)][col] = 1; U[0*m + (0+j_offset)][col] = -1
        V[(0+j_offset)*p + 0][col] = 1; V[(0+j_offset)*p + 1][col] = 1
        W[1*n + 1][col] = 1
        col += 1
        # M7
        U[0*m + (1+j_offset)][col] = 1; U[1*m + (1+j_offset)][col] = -1
        V[(1+j_offset)*p + 0][col] = 1; V[(1+j_offset)*p + 1][col] = 1
        W[0*n + 0][col] = 1
        col += 1

    def add_schoolbook_221(j_offset, k_offset):
        nonlocal col
        # C[0, k] = A[0,0]B[0,k] + A[0,1]B[1,k]
        U[0*m + (0+j_offset)][col] = 1
        V[(0+j_offset)*p + (k_offset)][col] = 1
        W[(k_offset)*n + 0][col] = 1
        col += 1
        U[0*m + (1+j_offset)][col] = 1
        V[(1+j_offset)*p + (k_offset)][col] = 1
        W[(k_offset)*n + 0][col] = 1
        col += 1
        # C[1, k] = A[1,0]B[0,k] + A[1,1]B[1,k]
        U[1*m + (0+j_offset)][col] = 1
        V[(0+j_offset)*p + (k_offset)][col] = 1
        W[(k_offset)*n + 1][col] = 1
        col += 1
        U[1*m + (1+j_offset)][col] = 1
        V[(1+j_offset)*p + (k_offset)][col] = 1
        W[(k_offset)*n + 1][col] = 1
        col += 1

    # Block 1: j in [0, 2)
    # p split: 2, 2, 1
    add_strassen(0) # p in [0, 2)
    add_strassen(0) # p in [2, 4) -> Wait, Strassen is 2x2x2.
    # I need to offset k for the second Strassen.
    # My add_strassen assumes k in [0, 2).
    # I need a k_offset parameter.
    
    # Let's add k_offset to add_strassen.
    
    # Re-defining with k_offset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen_k(j_offset, k_offset):
        nonlocal col
        # M1
        U[0*m + (0+j_offset)][col] = 1; U[0*m + (1+j_offset)][col] = 1
        V[(0+j_offset)*p + (k_offset+0)][col] = 1; V[(1+j_offset)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + 0][col] = 1; W[(k_offset+1)*n + 1][col] = 1
        col += 1
        # M2
        U[1*m + (0+j_offset)][col] = 1; U[1*m + (1+j_offset)][col] = 1
        V[(0+j_offset)*p + (k_offset+0)][col] = 1
        W[(k_offset+0)*n + 1][col] = 1; W[(k_offset+1)*n + 1][col] = -1
        col += 1
        # M3
        U[0*m + (0+j_offset)][col] = 1
        V[(0+j_offset)*p + (k_offset+1)][col] = 1; V[(1+j_offset)*p + (k_offset+1)][col] = -1
        W[(k_offset+1)*n + 0][col] = 1
        col += 1
        # M4
        U[1*m + (1+j_offset)][col] = 1
        V[(1+j_offset)*p + (k_offset+0)][col] = 1; V[(0+j_offset)*p + (k_offset+0)][col] = -1
        W[(k_offset+0)*n + 0][col] = 1; W[(k_offset+0)*n + 1][col] = 1
        col += 1
        # M5
        U[0*m + (0+j_offset)][col] = 1; U[0*m + (1+j_offset)][col] = 1
        V[(1+j_offset)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + 0][col] = -1; W[(k_offset+1)*n + 0][col] = 1
        col += 1
        # M6
        U[1*m + (0+j_offset)][col] = 1; U[0*m + (0+j_offset)][col] = -1
        V[(0+j_offset)*p + (k_offset+0)][col] = 1; V[(0+j_offset)*p + (k_offset+1)][col] = 1
        W[(k_offset+1)*n + 1][col] = 1
        col += 1
        # M7
        U[0*m + (1+j_offset)][col] = 1; U[1*m + (1+j_offset)][col] = -1
        V[(1+j_offset)*p + (k_offset+0)][col] = 1; V[(1+j_offset)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + 0][col] = 1
        col += 1

    # Block 1: j in [0, 2)
    add_strassen_k(0, 0) # k in [0, 2)
    add_strassen_k(0, 2) # k in [2, 4)
    add_schoolbook_221(0, 4) # k in [4, 5)

    # Block 2: j in [2, 4)
    add_strassen_k(2, 0) # k in [0, 2)
    add_strassen_k(2, 2) # k in [2, 4)
    add_schoolbook_221(2, 4) # k in [4, 5)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

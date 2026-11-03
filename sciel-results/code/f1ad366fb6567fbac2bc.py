import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 68 decomposition
    R = 68
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(i, j, j2, k, i2, val=1):
        nonlocal col
        U[i * m + j][col] += val
        V[j2 * p + k][col] += val
        W[k * n + i2][col] += val
        col += 1

    # 1. 2x2x8 part (i in {0,1}, j in {0,1}, k in 0..7)
    # This consists of 4 independent 2x2x2 multiplications
    # Strassen coefficients for 2x2x2
    # M1..M7
    # U indices: i, j
    # V indices: j, k
    # W indices: k, i
    
    strassen_coeffs = [
        # (U_i, U_j, V_j, V_k, W_k, W_i, coeff)
        # M1 = (a11+a22)(b11+b22)
        (0, 0, 0, 0, 0, 0, 1),
        (0, 0, 1, 1, 0, 0, 1),
        (1, 1, 0, 0, 0, 0, 1),
        (1, 1, 1, 1, 0, 0, 1),
        # M2 = (a21+a22)b11
        (1, 0, 0, 0, 0, 1, 1),
        (1, 1, 0, 0, 0, 1, 1),
        # M3 = a11(b12-b22)
        (0, 0, 0, 1, 1, 0, 1),
        (0, 0, 1, 1, 1, 0, -1),
        # M4 = a22(b21-b11)
        (1, 1, 1, 0, 0, 1, 1),
        (1, 1, 0, 0, 0, 1, -1),
        # M5 = (a11+a12)b22
        (0, 0, 1, 1, 1, 0, 1),
        (0, 1, 1, 1, 1, 0, 1),
        # M6 = (a21-a11)(b11+b12)
        (1, 0, 0, 0, 0, 1, 1),
        (1, 0, 0, 1, 0, 1, 1),
        (0, 0, 0, 0, 0, 1, -1),
        (0, 0, 0, 1, 0, 1, -1),
        # M7 = (a12-a22)(b21+b22)
        (0, 1, 1, 0, 1, 0, 1),
        (0, 1, 1, 1, 1, 0, 1),
        (1, 1, 1, 0, 1, 0, -1),
        (1, 1, 1, 1, 1, 0, -1),
    ]
    
    # We need to map Strassen results to C11, C12, C21, C22
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Instead of adding terms one by one, we can define the 7 intermediate values
    # and then combine them in W? No, W is part of the decomposition.
    # The standard Strassen decomposition has rank 7.
    # The 7 terms are the M's.
    # U, V, W for the M's are:
    # M1: U=[1,1,1,1] (i,j), V=[1,1,1,1] (j,k), W=[1,0,0,1] (k,i) ? No.
    # Let's use the explicit rank-1 tensors for Strassen.
    
    # Strassen Rank 1 Tensors:
    # 1: (1,1,1,1) in A, (1,1,1,1) in B, (1,0,0,1) in C? No.
    # C11 gets 1, C22 gets 1.
    # So W vector for M1 has 1 at (k=0,i=0) and (k=1,i=1).
    
    # Let's define the 7 Strassen tensors for a single 2x2x2 block at offset k_off.
    def add_strassen_block(k_off):
        nonlocal col
        # M1
        U[0 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        U[0 * m + 1][col] += 1 # Wait, a12? No, a11+a22.
        # A = [[a00, a01], [a10, a11]]
        # M1 = (a00+a11)(b00+b11)
        # U: a00=1, a11=1.
        # V: b00=1, b11=1.
        # W: c00=1, c11=1.
        
        # Let's restart the Strassen definition with correct indices.
        # A: i=0,1; j=0,1
        # B: j=0,1; k=0,1
        # C: k=0,1; i=0,1
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[0][col] += 1
        U[3][col] += 1
        V[0][col] += 1
        V[3][col] += 1
        W[k_off * n + 0][col] += 1
        W[(k_off+1) * n + 1][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[3][col] += 1
        U[2][col] += 1 # A10
        V[0][col] += 1
        W[k_off * n + 1][col] += 1 # C10
        col += 1
        
        # M3 = A00(B01-B11)
        U[0][col] += 1
        V[1][col] += 1 # B01
        V[2][col] += -1 # B11
        W[(k_off+1) * n + 0][col] += 1 # C01
        col += 1
        
        # M4 = A11(B10-B00)
        U[3][col] += 1
        V[2][col] += 1 # B10
        V[0][col] += -1 # B00
        W[k_off * n + 1][col] += 1 # C10
        col += 1
        
        # M5 = (A00+A01)B11
        U[0][col] += 1
        U[1][col] += 1 # A01
        V[3][col] += 1 # B11
        W[(k_off+1) * n + 0][col] += 1 # C01
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][col] += 1 # A10
        U[0][col] += -1 # A00
        V[0][col] += 1 # B00
        V[1][col] += 1 # B01
        W[k_off * n + 1][col] += 1 # C10
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][col] += 1 # A01
        U[3][col] += -1 # A11
        V[2][col] += 1 # B10
        V[3][col] += 1 # B11
        W[(k_off+1) * n + 0][col] += 1 # C01
        col += 1

    # Apply Strassen to the 4 blocks of the 2x2x8 part
    # Note: The indices in U, V, W for the 2x2 part are:
    # i in {0,1}, j in {0,1} for U.
    # j in {0,1}, k in {0..7} for V.
    # k in {0..7}, i in {0,1} for W.
    
    # My add_strassen_block uses local indices 0,1 for i,j,k.
    # I need to map them to global indices.
    # U global: i*m + j. i in {0,1}, j in {0,1}.
    # V global: j*p + k. j in {0,1}, k in {k_off, k_off+1}.
    # W global: k*n + i. k in {k_off, k_off+1}, i in {0,1}.
    
    # I will rewrite add_strassen_block to take offsets.
    
    col = 0
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    def add_strassen_block_global(i_off, j_off, k_off):
        nonlocal col
        # i_off, j_off are 0 for the 2x2 block.
        # k_off is 0, 2, 4, 6.
        
        # M1
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+0)][col] += 1
        W[(k_off+1)*n + (i_off+1)][col] += 1
        col += 1
        
        # M2
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1
        
        # M3
        U[(i_off+0)*m + (j_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += -1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1
        
        # M4
        U[(i_off+1)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+0)][col] += -1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1
        
        # M5
        U[(i_off+0)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+1)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1
        
        # M6
        U[(i_off+1)*m + (j_off+0)][col] += 1
        U[(i_off+0)*m + (j_off+0)][col] += -1
        V[(j_off+0)*p + (k_off+0)][col] += 1
        V[(j_off+0)*p + (k_off+1)][col] += 1
        W[(k_off+0)*n + (i_off+1)][col] += 1
        col += 1
        
        # M7
        U[(i_off+0)*m + (j_off+1)][col] += 1
        U[(i_off+1)*m + (j_off+1)][col] += -1
        V[(j_off+1)*p + (k_off+0)][col] += 1
        V[(j_off+1)*p + (k_off+1)][col] += 1
        W[(k_off+1)*n + (i_off+0)][col] += 1
        col += 1

    # Add 4 Strassen blocks for 2x2x8
    for k_off in [0, 2, 4, 6]:
        add_strassen_block_global(0, 0, k_off)
        
    # 2. 2x1x8 part (i in {0,1}, j=2, k in 0..7)
    for i in [0, 1]:
        for k in range(8):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # 3. 1x2x8 part (i=2, j in {0,1}, k in 0..7)
    for j in [0, 1]:
        for k in range(8):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    # 4. 1x1x8 part (i=2, j=2, k in 0..7)
    for k in range(8):
        U[2*m + 2][col] = 1
        V[2*p + k][col] = 1
        W[k*n + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=6
    
    # We will build U, V, W with total rank 88.
    # U: (15, 88), V: (30, 88), W: (18, 88)
    
    U = [[0] * 88 for _ in range(n * m)]
    V = [[0] * 88 for _ in range(m * p)]
    W = [[0] * 88 for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Part 1: 3x3x6 (i in 0..2, j in 0..2, k in 0..5)
    # Split k into 0..2 and 3..5
    
    # Part 1a: 3x3x3 (i in 0..2, j in 0..2, k in 0..2) Schoolbook
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # u_idx: i*m + j = i*5 + j
                # v_idx: j*p + k = j*6 + k
                # w_idx: k*n + i = k*3 + i
                add_term(i * 5 + j, j * 6 + k, k * 3 + i)
                
    # Part 1b: 3x3x3 (i in 0..2, j in 0..2, k in 3..5) Schoolbook
    for i in range(3):
        for j in range(3):
            for k in range(3, 6):
                add_term(i * 5 + j, j * 6 + k, k * 3 + i)

    # Part 2: 3x2x6 (i in 0..2, j in 3..4, k in 0..5)
    # Split i into 0 and 1..2
    
    # Part 2a: 1x2x6 (i=0, j in 3..4, k in 0..5) Schoolbook
    for j in range(3, 5):
        for k in range(6):
            # i=0
            add_term(0 * 5 + j, j * 6 + k, k * 3 + 0)

    # Part 2b: 2x2x6 (i in 1..2, j in 3..4, k in 0..5)
    # Split k into 0..2 and 3..5
    
    # Part 2b1: 2x2x3 (i in 1..2, j in 3..4, k in 0..2)
    # Use Strassen for k=0,1 (2x2x2) and Schoolbook for k=2 (2x2x1)
    # Indices for this block:
    # i_local in 0..1 -> i_global = i_local + 1
    # j_local in 0..1 -> j_global = j_local + 3
    # k_local in 0..2 -> k_global = k_local
    
    # Strassen 2x2x2 for k=0,1
    # A is 2x2 (i_local, j_local), B is 2x2 (j_local, k_local)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map these to U, V, W terms.
    # Each M is a product of a linear combo of A and a linear combo of B.
    # We will add 7 terms.
    
    # M1
    # A: A00 + A11 -> (i=1, j=3) + (i=2, j=4)
    # B: B00 + B11 -> (j=3, k=0) + (j=4, k=1)
    # This expands to 4 terms in U,V,W? No, U and V are linear.
    # U[i, c] = coeff. V[j, c] = coeff.
    # We add one column c.
    # U[1*5+3] += 1, U[2*5+4] += 1
    # V[3*6+0] += 1, V[4*6+1] += 1
    # W[k*n+i] is determined by the product?
    # Wait, the tensor decomposition is sum of rank 1 tensors.
    # A term is u * v * w.
    # If we use Strassen, we are replacing 4 mults with 7 mults.
    # The 7 mults produce intermediate values M1..M7.
    # Then C is linear combo of M1..M7.
    # This corresponds to 7 rank-1 terms in the tensor?
    # No, Strassen is a bilinear algorithm.
    # It defines 7 scalar multiplications.
    # Each multiplication is $M_l = (\sum a_{ij} u_{ij}) (\sum b_{jk} v_{jk})$.
    # This is a rank-1 tensor in the space of A and B?
    # Yes, each $M_l$ corresponds to a rank-1 tensor $\sum_{i,j,k} u_{ij} v_{jk} w_{ik}^{(l)}$.
    # Actually, the output $C_{ik}$ is $\sum_l M_l W_{ik}^{(l)}$.
    # So we add 7 columns to U, V, W.
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # i_off=1, j_off=3, k_off=0
        # i in i_off..i_off+1, j in j_off..j_off+1, k in k_off..k_off+1
        
        # M1: (A00+A11)(B00+B11)
        # U: (i0,j0), (i1,j1)
        # V: (j0,k0), (j1,k1)
        # W: C00, C01, C10, C11 contributions
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1 (M1)
        U[i_off*5 + j_off][col] = 1
        U[(i_off+1)*5 + (j_off+1)][col] = 1
        V[j_off*6 + k_off][col] = 1
        V[(j_off+1)*6 + (k_off+1)][col] = 1
        # W for M1:
        # W[k0*3 + i0] += 1 (C00)
        # W[k1*3 + i0] -= 1 (C01? No, C01 uses M3,M5)
        # Let's check Strassen formulas carefully.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M1, W contributions:
        # W[k0, i0] += 1
        # W[k1, i1] += 1
        W[k_off*3 + i_off][col] = 1
        W[(k_off+1)*3 + (i_off+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[i_off*5 + (j_off+1)][col] = 1 # A10 is (i1, j0)? No.
        # A indices: i0=0, i1=1. j0=0, j1=1.
        # A10 is (i1, j0). A11 is (i1, j1).
        # So U[(i1, j0)] and U[(i1, j1)].
        U[i_off*5 + (j_off+1)][col] = 1 # Wait, i_off+1, j_off
        U[(i_off+1)*5 + (j_off+1)][col] = 1
        V[j_off*6 + k_off][col] = 1 # B00 is (j0, k0)
        # W for M2:
        # C10 += M2, C11 -= M2
        W[k_off*3 + (i_off+1)][col] = 1
        W[(k_off+1)*3 + (i_off+1)][col] = -1
        col += 1
        
        # M3: A00(B10-B11)
        U[i_off*5 + j_off][col] = 1 # A00
        V[(j_off+1)*6 + k_off][col] = 1 # B10 is (j1, k0)
        V[(j_off+1)*6 + (k_off+1)][col] = -1 # B11 is (j1, k1)
        # W for M3:
        # C01 += M3, C11 += M3
        W[(k_off+1)*3 + i_off][col] = 1
        W[(k_off+1)*3 + (i_off+1)][col] = 1
        col += 1
        
        # M4: A11(B01-B00)
        U[(i_off+1)*5 + (j_off+1)][col] = 1 # A11
        V[j_off*6 + (k_off+1)][col] = 1 # B01 is (j0, k1)
        V[j_off*6 + k_off][col] = -1 # B00 is (j0, k0)
        # W for M4:
        # C00 += M4, C10 += M4
        W[k_off*3 + i_off][col] = 1
        W[k_off*3 + (i_off+1)][col] = 1
        col += 1
        
        # M5: (A00+A01)B11
        U[i_off*5 + j_off][col] = 1 # A00
        U[i_off*5 + (j_off+1)][col] = 1 # A01
        V[(j_off+1)*6 + (k_off+1)][col] = 1 # B11
        # W for M5:
        # C00 -= M5, C01 += M5
        W[k_off*3 + i_off][col] = -1
        W[(k_off+1)*3 + i_off][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[(i_off+1)*5 + j_off][col] = 1 # A10
        U[i_off*5 + j_off][col] = -1 # A00
        V[j_off*6 + k_off][col] = 1 # B00
        V[j_off*6 + (k_off+1)][col] = 1 # B01
        # W for M6:
        # C11 += M6
        W[(k_off+1)*3 + (i_off+1)][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[i_off*5 + (j_off+1)][col] = 1 # A01
        U[(i_off+1)*5 + (j_off+1)][col] = -1 # A11
        V[(j_off+1)*6 + k_off][col] = 1 # B10
        V[(j_off+1)*6 + (k_off+1)][col] = 1 # B11
        # W for M7:
        # C00 += M7
        W[k_off*3 + i_off][col] = 1
        col += 1

    # Apply Strassen to Part 2b1 (i=1..2, j=3..4, k=0..1)
    add_strassen_block(1, 3, 0)
    
    # Schoolbook for k=2 in Part 2b1 (i=1..2, j=3..4, k=2)
    for i in range(1, 3):
        for j in range(3, 5):
            add_term(i * 5 + j, j * 6 + 2, 2 * 3 + i)
            
    # Part 2b2: 2x2x3 (i in 1..2, j in 3..4, k in 3..5)
    # Strassen for k=3,4
    add_strassen_block(1, 3, 3)
    # Schoolbook for k=5
    for i in range(1, 3):
        for j in range(3, 5):
            add_term(i * 5 + j, j * 6 + 5, 5 * 3 + i)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

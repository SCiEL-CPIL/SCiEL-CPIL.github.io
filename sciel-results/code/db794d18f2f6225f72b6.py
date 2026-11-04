import time
import sys

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 72.
    # Part 1: 4x4x4 part using Strassen (rank 56)
    # Part 2: 4x4x1 part using schoolbook (rank 16)
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # List of (A_block_row, A_block_col, B_block_row, B_block_col, C_block_row, C_block_col)
    # C_rc = sum_k A_rk B_kc
    # We decompose the 4x4x4 multiplication into 8 products of 2x2x2 blocks.
    # A is 4x4, B is 4x4 (first 4 cols), C is 4x4 (first 4 cols).
    # Blocks are 2x2.
    # A blocks: (0,0), (0,1), (1,0), (1,1) -> indices (r, c) in block space
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    # The 8 terms in the block matrix multiplication:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    products = [
        (0, 0, 0, 0, 0, 0), # A00 B00 -> C00
        (0, 1, 1, 0, 0, 0), # A01 B10 -> C00
        (0, 0, 0, 1, 0, 1), # A00 B01 -> C01
        (0, 1, 1, 1, 0, 1), # A01 B11 -> C01
        (1, 0, 0, 0, 1, 0), # A10 B00 -> C10
        (1, 1, 1, 0, 1, 0), # A11 B10 -> C10
        (1, 0, 0, 1, 1, 1), # A10 B01 -> C11
        (1, 1, 1, 1, 1, 1), # A11 B11 -> C11
    ]
    
    for (ar, ac, br, bc, cr, cc) in products:
        # Add Strassen for this 2x2x2 product
        # a_base = (ar*2, ac*2)
        # b_base = (br*2, bc*2)
        # c_base = (cr*2, cc*2)
        
        a_base = (ar*2, ac*2)
        b_base = (br*2, bc*2)
        c_base = (cr*2, cc*2)
        
        def a_idx(r, c): return (a_base[0]+r)*4 + (a_base[1]+c)
        def b_idx(r, c): return (b_base[0]+r)*5 + (b_base[1]+c)
        # W[k*n+i, r] where k is col of C, i is row of C?
        # Task says: "output factor follows the official trace convention W[k*n+i,r]"
        # Wait, standard convention for tensor T_{ijk} where A is n x m, B is m x p, C is n x p.
        # Usually U_{i, alpha}, V_{j, alpha}, W_{k, alpha} such that C_{ik} = sum_alpha U_{i, alpha} V_{j, alpha} W_{k, alpha} ?
        # No, matrix multiplication C = A B.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # Tensor decomposition: T_{ikj} = sum_alpha U_{i, alpha} V_{j, alpha} W_{k, alpha}?
        # Let's check the shapes in the prompt:
        # U: (n*m, rank) -> indices (i, j) flattened? Or (i) ?
        # V: (m*p, rank) -> indices (j, k) flattened?
        # W: (p*n, rank) -> indices (k, i) flattened?
        
        # If U is (n*m, rank), it likely corresponds to the A matrix entries.
        # If V is (m*p, rank), it likely corresponds to the B matrix entries.
        # If W is (p*n, rank), it likely corresponds to the C matrix entries.
        
        # Let's verify the index mapping for W.
        # "W[k*n+i, r]" implies the row index of W is `k*n + i`.
        # So for a given C entry C_{i,k} (row i, col k), the corresponding W row is `k*n + i`.
        # This matches the shape (p*n, rank) where p is number of cols in C, n is number of rows in C.
        # So W index = k * n + i.
        
        def c_idx(r, c):
            # r is row of C block, c is col of C block
            # Global row i = c_base[0] + r
            # Global col k = c_base[1] + c
            return (c_base[1]+c)*4 + (c_base[0]+r)
        
        # Strassen factors for 2x2x2
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
        
        # M1: (A00+A11)(B00+B11)
        U[a_idx(0,0)][col] += 1
        U[a_idx(1,1)][col] += 1
        V[b_idx(0,0)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] += 1
        W[c_idx(1,1)][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[a_idx(1,0)][col] += 1
        U[a_idx(1,1)][col] += 1
        V[b_idx(0,0)][col] += 1
        W[c_idx(1,0)][col] += 1
        W[c_idx(1,1)][col] -= 1
        col += 1
        
        # M3: A00(B01-B11)
        U[a_idx(0,0)][col] += 1
        V[b_idx(0,1)][col] += 1
        V[b_idx(1,1)][col] -= 1
        W[c_idx(0,1)][col] += 1
        W[c_idx(1,1)][col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[a_idx(1,1)][col] += 1
        V[b_idx(1,0)][col] += 1
        V[b_idx(0,0)][col] -= 1
        W[c_idx(0,0)][col] += 1
        W[c_idx(1,0)][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[a_idx(0,0)][col] += 1
        U[a_idx(0,1)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] -= 1
        W[c_idx(0,1)][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[a_idx(1,0)][col] += 1
        U[a_idx(0,0)][col] -= 1
        V[b_idx(0,0)][col] += 1
        V[b_idx(0,1)][col] += 1
        W[c_idx(1,1)][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[a_idx(0,1)][col] += 1
        U[a_idx(1,1)][col] -= 1
        V[b_idx(1,0)][col] += 1
        V[b_idx(1,1)][col] += 1
        W[c_idx(0,0)][col] += 1
        col += 1

    # Part 2: 4x4x1 (rank 16)
    # A (4x4) * B_2 (4x1) -> C_2 (4x1)
    # B_2 is column 4 of B.
    # C_2 is column 4 of C.
    
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1 # B index: row j, col 4
            W[4*4 + i][col] = 1 # C index: k=4, i. W[k*n+i] = W[4*4+i]
            col += 1

    return {"U": U, "V": V, "W": W}

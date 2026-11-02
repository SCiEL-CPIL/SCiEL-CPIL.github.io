import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by combining smaller decompositions.
    # The final U, V, W will be lists of lists.
    # We need to map the local indices of sub-problems to global indices.
    
    # Sub-problem 1: 1 x 4 x 7 (Rank 1)
    # Sub-problem 2: 2 x 2 x 7 (Rank 25)
    # Sub-problem 3: 2 x 2 x 7 (Rank 25)
    
    # Total Rank = 1 + 25 + 25 = 51
    
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add a decomposition
    def add_decomp(u_local, v_local, w_local, i_offset, j_offset, k_offset, u_rows, v_rows, w_rows):
        nonlocal current_rank
        r = len(u_local[0])
        for r_idx in range(r):
            for i in range(u_rows):
                U[(i_offset + i) * m + (j_offset + 0)][current_rank + r_idx] += u_local[i][r_idx] # Note: j index in U is local
            # Wait, U shape is (n*m, rank). Index is i*m + j.
            # In sub-problem, i is local row, j is local col.
            # Global i = i_offset + i_local
            # Global j = j_offset + j_local
            # So U[ (i_offset+i_local)*m + (j_offset+j_local) ][col]
            
            # Let's rewrite the helper properly.
            
    # Let's just build it directly.
    
    # 1. Top row of A (1 x 4) multiplied by B (4 x 7)
    # This corresponds to i=0, j=0..3, k=0..6
    # Rank 1 decomposition:
    # U[0*m + j][0] = 1 for all j? No.
    # T[0, j, k] = 1.
    # U[0*m + j][r] = 1
    # V[j*p + k][r] = 1
    # W[k*n + 0][r] = 1
    # This works for r=0.
    
    for j in range(4):
        U[0 * m + j][current_rank] = 1
    for j in range(4):
        for k in range(7):
            V[j * p + k][current_rank] = 1
    for k in range(7):
        W[k * n + 0][current_rank] = 1
    current_rank += 1
    
    # 2. Bottom 2 rows of A (2 x 4) multiplied by B (4 x 7)
    # Split into two 2 x 2 x 7 blocks.
    # Block A: A[1:3, 0:2] * B[0:2, :]
    # Block B: A[1:3, 2:4] * B[2:4, :]
    
    def add_2x2x7(i_off, j_off, k_off):
        nonlocal current_rank
        # 2 x 2 x 7 split into 2x2x2, 2x2x2, 2x2x3
        # 2x2x2 uses Strassen (Rank 7)
        # 2x2x3 uses Strassen (2x2x2) + Schoolbook (2x2x1) (Rank 11)
        
        # Sub-block 1: 2x2x2 (j_off..j_off+1, k_off..k_off+1)
        add_2x2x2(i_off, j_off, k_off)
        
        # Sub-block 2: 2x2x2 (j_off..j_off+1, k_off+2..k_off+3)
        add_2x2x2(i_off, j_off, k_off + 2)
        
        # Sub-block 3: 2x2x3 (j_off..j_off+1, k_off+4..k_off+6)
        # Split into 2x2x2 (k_off+4..k_off+5) and 2x2x1 (k_off+6)
        add_2x2x2(i_off, j_off, k_off + 4)
        add_2x2x1(i_off, j_off, k_off + 6)
        
    def add_2x2x2(i_off, j_off, k_off):
        nonlocal current_rank
        # Strassen
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices:
        # A: (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
        # B: (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
        # C: (i_off, k_off), (i_off, k_off+1), (i_off+1, k_off), (i_off+1, k_off+1)
        
        terms = [
            # M1
            { 'U': [(0,1,1), (1,1,1), (0,0,1), (1,0,1)], 'V': [(0,0,1), (0,1,1), (1,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1), (0,1,-1), (1,0,1)] },
            # This notation is confusing. Let's use explicit loops.
        ]
        
        # Strassen coefficients
        # r=0: M1
        # r=1: M2
        # r=2: M3
        # r=3: M4
        # r=4: M5
        # r=5: M6
        # r=6: M7
        
        # U entries (i, j)
        # V entries (j, k)
        # W entries (k, i)
        
        # M1: (A00+A11)(B00+B11)
        # U: A00(1), A11(1)
        # V: B00(1), B11(1)
        # W: C00(1), C11(1), C01(-1), C10(1) ? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's define the contributions of each Mr to the U, V, W tensors.
        
        # M1
        U[i_off*m + j_off][current_rank+0] += 1
        U[(i_off+1)*m + (j_off+1)][current_rank+0] += 1
        V[j_off*p + k_off][current_rank+0] += 1
        V[(j_off+1)*p + (k_off+1)][current_rank+0] += 1
        W[k_off*n + i_off][current_rank+0] += 1
        W[(k_off+1)*n + (i_off+1)][current_rank+0] += 1
        W[(k_off+1)*n + i_off][current_rank+0] -= 1
        W[k_off*n + (i_off+1)][current_rank+0] += 1
        
        # M2
        U[(i_off+1)*m + j_off][current_rank+1] += 1
        U[(i_off+1)*m + (j_off+1)][current_rank+1] += 1
        V[j_off*p + k_off][current_rank+1] += 1
        W[k_off*n + (i_off+1)][current_rank+1] += 1
        W[(k_off+1)*n + (i_off+1)][current_rank+1] -= 1
        
        # M3
        U[i_off*m + j_off][current_rank+2] += 1
        V[j_off*p + (k_off+1)][current_rank+2] += 1
        V[(j_off+1)*p + (k_off+1)][current_rank+2] -= 1
        W[(k_off+1)*n + i_off][current_rank+2] += 1
        W[k_off*n + (i_off+1)][current_rank+2] += 1
        
        # M4
        U[(i_off+1)*m + (j_off+1)][current_rank+3] += 1
        V[(j_off+1)*p + k_off][current_rank+3] += 1
        V[j_off*p + k_off][current_rank+3] -= 1
        W[k_off*n + i_off][current_rank+3] += 1
        W[k_off*n + (i_off+1)][current_rank+3] += 1
        
        # M5
        U[i_off*m + j_off][current_rank+4] += 1
        U[i_off*m + (j_off+1)][current_rank+4] += 1
        V[(j_off+1)*p + (k_off+1)][current_rank+4] += 1
        W[k_off*n + i_off][current_rank+4] -= 1
        W[(k_off+1)*n + i_off][current_rank+4] += 1
        
        # M6
        U[(i_off+1)*m + j_off][current_rank+5] += 1
        U[i_off*m + j_off][current_rank+5] -= 1
        V[j_off*p + k_off][current_rank+5] += 1
        V[j_off*p + (k_off+1)][current_rank+5] += 1
        W[(k_off+1)*n + (i_off+1)][current_rank+5] += 1
        
        # M7
        U[i_off*m + (j_off+1)][current_rank+6] += 1
        U[(i_off+1)*m + (j_off+1)][current_rank+6] -= 1
        V[(j_off+1)*p + k_off][current_rank+6] += 1
        V[(j_off+1)*p + (k_off+1)][current_rank+6] += 1
        W[k_off*n + i_off][current_rank+6] += 1
        
        current_rank += 7
        
    def add_2x2x1(i_off, j_off, k_off):
        nonlocal current_rank
        # Schoolbook for 2x2x1
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        # Rank 4
        
        # r0: A00 B00
        U[i_off*m + j_off][current_rank+0] += 1
        V[j_off*p + k_off][current_rank+0] += 1
        W[k_off*n + i_off][current_rank+0] += 1
        
        # r1: A01 B10
        U[i_off*m + (j_off+1)][current_rank+1] += 1
        V[(j_off+1)*p + k_off][current_rank+1] += 1
        W[k_off*n + i_off][current_rank+1] += 1
        
        # r2: A10 B00
        U[(i_off+1)*m + j_off][current_rank+2] += 1
        V[j_off*p + k_off][current_rank+2] += 1
        W[k_off*n + (i_off+1)][current_rank+2] += 1
        
        # r3: A11 B10
        U[(i_off+1)*m + (j_off+1)][current_rank+3] += 1
        V[(j_off+1)*p + k_off][current_rank+3] += 1
        W[k_off*n + (i_off+1)][current_rank+3] += 1
        
        current_rank += 4

    # Call the blocks
    # Block 1: i=1,2; j=0,1; k=0..6
    add_2x2x7(1, 0, 0)
    # Block 2: i=1,2; j=2,3; k=0..6
    add_2x2x7(1, 2, 0)
    
    return {"U": U, "V": V, "W": W}

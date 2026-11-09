import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 42.
    # Structure:
    # A (2x3) = [A_L (2x2) | A_R (2x1)]
    # B (3x9) = [B_T (2x9); B_B (1x9)]
    # C = A_L B_T + A_R B_B
    
    # Part 1: A_L (2x2) * B_T (2x9)
    # B_T is split into 3 blocks of 2x3: B_T1, B_T2, B_T3
    # Each 2x2 * 2x3 uses rank 11 (7 for 2x2*2x2 + 4 for 2x2*2x1)
    # Total rank for Part 1: 3 * 11 = 33
    
    # Part 2: A_R (2x1) * B_B (1x9)
    # Rank 9 (schoolbook)
    # Total rank: 33 + 9 = 42
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # Maps local indices to global indices
    def add_strassen_222(i_offset, j_offset, k_offset):
        nonlocal col
        # i: 0,1; j: 0,1; k: 0,1
        # Global U index: (i_offset+i)*m + (j_offset+j)
        # Global V index: (j_offset+j)*p + (k_offset+k)
        # Global W index: (k_offset+k)*n + (i_offset+i)
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11, A22. V: B11, B22. W: C11, C22 (with signs)
        # This is complex to map directly. 
        # Easier to just define the 7 vectors explicitly for local 2x2x2.
        
        # Local U (4x7), V (4x7), W (4x7)
        # Indices: 0:00, 1:01, 2:10, 3:11
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        u[0, 0] = 1; u[3, 0] = 1
        v[0, 0] = 1; v[3, 0] = 1
        w[0, 0] = 1; w[3, 0] = 1
        
        # M2: (A10+A11)B00 -> C10+=1
        u[2, 1] = 1; u[3, 1] = 1
        v[0, 1] = 1
        w[0, 1] = 1 # C10 is w[0*2+1] = w[2]? No.
        # W index: k*n + i.
        # C10: k=0, i=1 -> index 1.
        # C11: k=1, i=1 -> index 3.
        # C01: k=1, i=0 -> index 2.
        # C00: k=0, i=0 -> index 0.
        
        # Let's re-verify W indices for 2x2 output.
        # C = [[C00, C01], [C10, C11]]
        # W[k*n+i]. n=2.
        # C00: k=0, i=0 -> 0
        # C01: k=1, i=0 -> 2
        # C10: k=0, i=1 -> 1
        # C11: k=1, i=1 -> 3
        
        # M1: C00+=1, C11+=1
        w[0, 0] = 1; w[3, 0] = 1
        
        # M2: (A10+A11)B00 -> C10+=1
        u[2, 1] = 1; u[3, 1] = 1
        v[0, 1] = 1
        w[1, 1] = 1
        
        # M3: A00(B01-B11) -> C01+=1
        u[0, 2] = 1
        v[1, 2] = 1; v[3, 2] = -1
        w[2, 2] = 1
        
        # M4: A11(B10-B00) -> C10+=1
        u[3, 3] = 1
        v[2, 3] = 1; v[0, 3] = -1
        w[1, 3] = 1
        
        # M5: (A00+A01)B11 -> C01+=1, C11+=1
        u[0, 4] = 1; u[1, 4] = 1
        v[3, 4] = 1
        w[2, 4] = 1; w[3, 4] = 1
        
        # M6: (A10-A00)(B00+B01) -> C11+=1
        u[2, 5] = 1; u[0, 5] = -1
        v[0, 5] = 1; v[1, 5] = 1
        w[3, 5] = 1
        
        # M7: (A01-A11)(B10+B11) -> C00+=1
        u[1, 6] = 1; u[3, 6] = -1
        v[2, 6] = 1; v[3, 6] = 1
        w[0, 6] = 1
        
        # Map to global
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    if u[i*2+j, r] != 0:
                        U[(i_offset+i)*m + (j_offset+j), col+r] += u[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    if v[j*2+k, r] != 0:
                        V[(j_offset+j)*p + (k_offset+k), col+r] += v[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    if w[k*2+i, r] != 0:
                        W[(k_offset+k)*n + (i_offset+i), col+r] += w[k*2+i, r]
        col += 7

    # Helper to add 2x2x1 block (rank 4)
    def add_221(i_offset, j_offset, k_offset):
        nonlocal col
        # A (2x2) * b (2x1) -> c (2x1)
        # u: 4x4, v: 4x4 (k=0 only), w: 4x4 (k=0 only)
        u = np.zeros((4, 4))
        v = np.zeros((4, 4))
        w = np.zeros((4, 4))
        
        # A00*b0 -> C0
        u[0, 0] = 1; v[0, 0] = 1; w[0, 0] = 1
        # A01*b1 -> C0
        u[1, 1] = 1; v[1, 1] = 1; w[0, 1] = 1
        # A10*b0 -> C1
        u[2, 2] = 1; v[0, 2] = 1; w[1, 2] = 1
        # A11*b1 -> C1
        u[3, 3] = 1; v[1, 3] = 1; w[1, 3] = 1
        
        for r in range(4):
            for i in range(2):
                for j in range(2):
                    if u[i*2+j, r] != 0:
                        U[(i_offset+i)*m + (j_offset+j), col+r] += u[i*2+j, r]
            for j in range(2):
                # k is fixed at 0
                if v[j*2+0, r] != 0:
                    V[(j_offset+j)*p + (k_offset+0), col+r] += v[j*2+0, r]
            # k is fixed at 0
            for i in range(2):
                if w[0*2+i, r] != 0:
                    W[(k_offset+0)*n + (i_offset+i), col+r] += w[0*2+i, r]
        col += 4

    # Helper to add 2x1x9 block (rank 9)
    def add_219(i_offset, j_offset, k_offset):
        nonlocal col
        # A (2x1) * B (1x9) -> C (2x9)
        # u: 2x9, v: 9x9 (j=0 only), w: 18x9
        # Actually V index: j*p+k. j=0.
        # W index: k*n+i.
        for i in range(2):
            for k in range(9):
                U[(i_offset+i)*m + (j_offset+0), col] = 1
                V[(j_offset+0)*p + (k_offset+k), col] = 1
                W[(k_offset+k)*n + (i_offset+i), col] = 1
                col += 1

    # Part 1: A_L (2x2) * B_T (2x9)
    # B_T is rows 0,1 of B.
    # Split B_T into 3 blocks of 2x3.
    # Block 1: cols 0-2 of B_T.
    #   Split 2x3 into 2x2 (cols 0-1) + 2x1 (col 2).
    add_strassen_222(0, 0, 0) # i=0, j=0, k=0..1
    add_221(0, 0, 2)        # i=0, j=0, k=2
    
    # Block 2: cols 3-5 of B_T.
    add_strassen_222(0, 0, 3)
    add_221(0, 0, 5)
    
    # Block 3: cols 6-8 of B_T.
    add_strassen_222(0, 0, 6)
    add_221(0, 0, 8)

    # Part 2: A_R (2x1) * B_B (1x9)
    # A_R is col 2 of A.
    # B_B is row 2 of B.
    add_219(0, 2, 0)

    # Convert to lists as required by interface?
    # The prompt example uses lists of lists.
    # "factor matrices of shapes ... with integer/half-integer entries"
    # I will return numpy arrays or lists. The example uses lists.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

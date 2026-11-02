import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition for 3x3x3 (rank 26) and 3x3x2 (rank 17)
    # Total rank 43
    
    # Helper to add a term to U, V, W lists
    # U, V, W will be lists of dicts or lists of lists
    # We'll use lists of [0]*size for efficiency
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_map, v_map, w_map):
        # u_map: {(i,j): coeff}
        # v_map: {(j,k): coeff}
        # w_map: {(k,i): coeff}
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        
        for (i, j), c in u_map.items():
            if c != 0:
                u_col[i * m + j] = c
                
        for (j, k), c in v_map.items():
            if c != 0:
                v_col[j * p + k] = c
                
        for (k, i), c in w_map.items():
            if c != 0:
                w_col[k * n + i] = c
                
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def add_schoolbook_block(i_start, i_end, j_start, j_end, k_start, k_end):
        # Adds schoolbook terms for sub-matrix A[i_start:i_end, j_start:j_end] * B[j_start:j_end, k_start:k_end]
        # Note: This is for the case where the block is square in the middle index j
        # But here j is the contraction index.
        # A is n x m. B is m x p.
        # We are summing over j.
        # The block is defined by ranges of i, j, k.
        # For each j in j_start:j_end, we have A[i,j] and B[j,k].
        # This creates a term for each (i, j, k).
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u_map = {(i, j): 1.0}
                    v_map = {(j, k): 1.0}
                    w_map = {(k, i): 1.0}
                    add_term(u_map, v_map, w_map)

    def add_strassen_2x2x2(i0, j0, k0):
        # Adds Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        # Indices:
        # i0, i0+1
        # j0, j0+1
        # k0, k0+1
        
        # M1 = (A00+A11)(B00+B11)
        add_term(
            {(i0, j0): 1.0, (i0+1, j0+1): 1.0},
            {(j0, k0): 1.0, (j0+1, k0+1): 1.0},
            {(k0, i0): 1.0, (k0+1, i0+1): 1.0}
        )
        
        # M2 = (A10+A11)B00
        add_term(
            {(i0+1, j0): 1.0, (i0+1, j0+1): 1.0},
            {(j0, k0): 1.0},
            {(k0, i0+1): 1.0} # C21 gets M2? No, C21 = M2+M4. C21 is i=1, k=0.
            # Wait, standard Strassen:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            # Here i=0 is row 1, i=1 is row 2.
            # k=0 is col 1, k=1 is col 2.
            # C21 corresponds to i=1, k=0. Index k*n+i = 0*2+1 = 1.
        )
        
        # M3 = A00(B01-B11)
        add_term(
            {(i0, j0): 1.0},
            {(j0, k0+1): 1.0, (j0+1, k0+1): -1.0},
            {(k0+1, i0): 1.0} # C12 is i=0, k=1. Index 1*2+0 = 2.
        )
        
        # M4 = A11(B10-B00)
        add_term(
            {(i0+1, j0+1): 1.0},
            {(j0+1, k0): 1.0, (j0, k0): -1.0},
            {(k0, i0+1): 1.0, (k0+1, i0+1): 1.0} # C21 (i=1,k=0) and C11 (i=0,k=0)?
            # C11 = ... + M4. C21 = ... + M4.
            # C11: i=0, k=0. Index 0.
            # C21: i=1, k=0. Index 1.
        )
        
        # M5 = (A00+A01)B11
        add_term(
            {(i0, j0): 1.0, (i0, j0+1): 1.0},
            {(j0+1, k0+1): 1.0},
            {(k0+1, i0): -1.0, (k0+1, i0+1): 1.0} # C11 (-M5), C22 (M5)
            # C11: i=0, k=0. Index 0.
            # C22: i=1, k=1. Index 3.
        )
        
        # M6 = (A10-A00)(B00+B01)
        add_term(
            {(i0+1, j0): 1.0, (i0, j0): -1.0},
            {(j0, k0): 1.0, (j0, k0+1): 1.0},
            {(k0+1, i0+1): 1.0} # C22 gets M6. i=1, k=1. Index 3.
        )
        
        # M7 = (A01-A11)(B10+B11)
        add_term(
            {(i0, j0+1): 1.0, (i0+1, j0+1): -1.0},
            {(j0+1, k0): 1.0, (j0+1, k0+1): 1.0},
            {(k0, i0): 1.0, (k0, i0+1): 1.0} # C11 (M7), C21? No.
            # C11 = ... + M7.
            # C11: i=0, k=0. Index 0.
            # Does M7 go to C21? No.
            # C11 = M1 + M4 - M5 + M7.
            # C12 = M3 + M5.
            # C21 = M2 + M4.
            # C22 = M1 - M2 + M3 + M6.
            # M7 only in C11.
        )

    # Construct 3x3x3 part (k=0,1,2)
    # Blocks:
    # A[0:2, 0:2] * B[0:2, 0:2] -> Strassen (i0=0, j0=0, k0=0)
    add_strassen_2x2x2(0, 0, 0)
    
    # A[0:2, 2] * B[2, 0:2] -> Schoolbook 2x1x2 (i0=0, i1=2, j=2, k0=0, k1=2)
    add_schoolbook_block(0, 2, 2, 3, 0, 2)
    
    # A[0:2, 0:2] * B[0:2, 2] -> Schoolbook 2x2x1 (i0=0, i1=2, j0=0, j1=2, k=2)
    add_schoolbook_block(0, 2, 0, 2, 2, 3)
    
    # A[0:2, 2] * B[2, 2] -> Schoolbook 2x1x1
    add_schoolbook_block(0, 2, 2, 3, 2, 3)
    
    # A[2, 0:2] * B[0:2, 0:2] -> Schoolbook 1x2x2
    add_schoolbook_block(2, 3, 0, 2, 0, 2)
    
    # A[2, 2] * B[2, 0:2] -> Schoolbook 1x1x2
    add_schoolbook_block(2, 3, 2, 3, 0, 2)
    
    # A[2, 0:2] * B[0:2, 2] -> Schoolbook 1x2x1
    add_schoolbook_block(2, 3, 0, 2, 2, 3)
    
    # A[2, 2] * B[2, 2] -> Schoolbook 1x1x1
    add_schoolbook_block(2, 3, 2, 3, 2, 3)
    
    # Construct 3x3x2 part (k=3,4)
    # This is A[0:3, 0:3] * B[0:3, 3:5]
    # Local k indices 0, 1 map to global 3, 4.
    
    # A[0:2, 0:2] * B[0:2, 3:5] -> Strassen (i0=0, j0=0, k0=3)
    add_strassen_2x2x2(0, 0, 3)
    
    # A[0:2, 2] * B[2, 3:5] -> Schoolbook 2x1x2
    add_schoolbook_block(0, 2, 2, 3, 3, 5)
    
    # A[2, 0:2] * B[0:2, 3:5] -> Schoolbook 1x2x2
    add_schoolbook_block(2, 3, 0, 2, 3, 5)
    
    # A[2, 2] * B[2, 3:5] -> Schoolbook 1x1x2
    add_schoolbook_block(2, 3, 2, 3, 3, 5)
    
    # Convert to numpy arrays
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}

import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define block partitions
    # A: 3x5 -> Rows [2, 1], Cols [2, 2, 1]
    # B: 5x6 -> Rows [2, 2, 1], Cols [2, 2, 2]
    
    a_row_splits = [2, 1]
    a_col_splits = [2, 2, 1]
    b_row_splits = [2, 2, 1]
    b_col_splits = [2, 2, 2]
    
    # Calculate total rank
    # We will count as we build
    rank = 0
    U = []
    V = []
    W = []
    
    def add_column(u_vec, v_vec, w_vec):
        nonlocal rank
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        rank += 1

    # Helper to get global indices
    # A index: i*m + j
    # B index: j*p + k
    # C index: k*n + i
    
    # We iterate over blocks
    # A blocks: (a_r, a_c)
    # B blocks: (b_r, b_c)
    # Note: a_c index corresponds to b_r index
    
    a_row_starts = []
    curr = 0
    for s in a_row_splits:
        a_row_starts.append(curr)
        curr += s
        
    a_col_starts = []
    curr = 0
    for s in a_col_splits:
        a_col_starts.append(curr)
        curr += s
        
    b_row_starts = []
    curr = 0
    for s in b_row_splits:
        b_row_starts.append(curr)
        curr += s
        
    b_col_starts = []
    curr = 0
    for s in b_col_splits:
        b_col_starts.append(curr)
        curr += s
        
    # Precompute sizes
    a_row_sizes = a_row_splits
    a_col_sizes = a_col_splits
    b_row_sizes = b_row_splits
    b_col_sizes = b_col_splits
    
    # Iterate through all block pairs (a_block, b_block)
    # a_block is (ar, ac), b_block is (br, bc)
    # Condition: ac index == br index (since A cols match B rows)
    
    for ar_idx in range(len(a_row_splits)):
        for ac_idx in range(len(a_col_splits)):
            br_idx = ac_idx
            for bc_idx in range(len(b_col_splits)):
                
                a_r_start = a_row_starts[ar_idx]
                a_c_start = a_col_starts[ac_idx]
                b_r_start = b_row_starts[br_idx]
                b_c_start = b_col_starts[bc_idx]
                
                a_r_size = a_row_sizes[ar_idx]
                a_c_size = a_col_sizes[ac_idx]
                b_r_size = b_row_sizes[br_idx]
                b_c_size = b_col_sizes[bc_idx]
                
                # Check if we can use Strassen
                if a_r_size == 2 and a_c_size == 2 and b_r_size == 2 and b_c_size == 2:
                    # Strassen
                    # Local indices 0..1
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)B00
                    # M3 = A00(B01-B11)
                    # M4 = A11(B10-B00)
                    # M5 = (A00+A01)B11
                    # M6 = (A10-A00)(B00+B01)
                    # M7 = (A01-A11)(B10+B11)
                    
                    # We need to map local (i,j,k) to global
                    # U global: (a_r_start+i)*m + (a_c_start+j)
                    # V global: (a_c_start+j)*p + (b_c_start+k)
                    # W global: (b_c_start+k)*n + (a_r_start+i)
                    
                    def strassen_col(u_coeffs, v_coeffs, w_coeffs):
                        # u_coeffs: list of (i, j, val)
                        # v_coeffs: list of (j, k, val)
                        # w_coeffs: list of (k, i, val)
                        u_vec = [0.0] * (n * m)
                        v_vec = [0.0] * (m * p)
                        w_vec = [0.0] * (p * n)
                        
                        for (li, lj, val) in u_coeffs:
                            gi = a_r_start + li
                            gj = a_c_start + lj
                            u_vec[gi * m + gj] = val
                            
                        for (lj, lk, val) in v_coeffs:
                            gj = a_c_start + lj
                            gk = b_c_start + lk
                            v_vec[gj * p + gk] = val
                            
                        for (lk, li, val) in w_coeffs:
                            gk = b_c_start + lk
                            gi = a_r_start + li
                            w_vec[gk * n + gi] = val
                            
                        add_column(u_vec, v_vec, w_vec)
                    
                    # M1
                    strassen_col(
                        [(0,0,1), (1,1,1)],
                        [(0,0,1), (1,1,1)],
                        [(0,0,1), (1,1,1)]
                    )
                    # M2
                    strassen_col(
                        [(1,0,1), (1,1,1)],
                        [(0,0,1)],
                        [(0,1,1)] # Wait, M2 contributes to C21?
                        # Formula: C21 = M2 + M4. C is 2x2.
                        # C21 is row 1, col 0.
                        # So W should have 1 at (k=0, i=1).
                        # Local k=0, i=1.
                    )
                    # Correction on W indices for Strassen:
                    # C11 (0,0): M1, M4, -M5, M7
                    # C12 (0,1): M3, M5
                    # C21 (1,0): M2, M4
                    # C22 (1,1): M1, -M2, M3, M6
                    
                    # Let's re-define W coeffs carefully.
                    
                    # M1: C11, C22
                    # W: (0,0,1), (1,1,1)
                    
                    # M2: C21
                    # W: (0,1,1) -> k=0, i=1
                    
                    # M3: C12, C22
                    # W: (1,0,1) -> k=1, i=0; (1,1,1) -> k=1, i=1
                    
                    # M4: C11, C21
                    # W: (0,0,1) -> k=0, i=0; (0,1,1) -> k=0, i=1
                    
                    # M5: C12, -C11
                    # W: (1,0,1) -> k=1, i=0; (0,0,-1) -> k=0, i=0
                    
                    # M6: C22
                    # W: (1,1,1) -> k=1, i=1
                    
                    # M7: C11
                    # W: (0,0,1) -> k=0, i=0

                    # Re-doing the columns with correct W assignments.
                    # I already added M1 above.
                    
                    # M2
                    strassen_col(
                        [(1,0,1), (1,1,1)],
                        [(0,0,1)],
                        [(0,1,1)]
                    )
                    
                    # M3
                    strassen_col(
                        [(0,0,1)],
                        [(0,1,1), (1,1,-1)],
                        [(1,0,1), (1,1,1)]
                    )
                    
                    # M4
                    strassen_col(
                        [(1,1,1)],
                        [(1,0,1), (0,0,-1)],
                        [(0,0,1), (0,1,1)]
                    )
                    
                    # M5
                    strassen_col(
                        [(0,0,1), (0,1,1)],
                        [(1,1,1)],
                        [(1,0,1), (0,0,-1)]
                    )
                    
                    # M6
                    strassen_col(
                        [(1,0,1), (0,0,-1)],
                        [(0,0,1), (0,1,1)],
                        [(1,1,1)]
                    )
                    
                    # M7
                    strassen_col(
                        [(0,1,1), (1,1,-1)],
                        [(1,0,1), (1,1,1)],
                        [(0,0,1)]
                    )
                    
                else:
                    # Schoolbook
                    for li in range(a_r_size):
                        for lj in range(a_c_size):
                            for lk in range(b_c_size):
                                u_vec = [0.0] * (n * m)
                                v_vec = [0.0] * (m * p)
                                w_vec = [0.0] * (p * n)
                                
                                gi = a_r_start + li
                                gj = a_c_start + lj
                                gk = b_c_start + lk
                                
                                u_vec[gi * m + gj] = 1.0
                                v_vec[gj * p + gk] = 1.0
                                w_vec[gk * n + gi] = 1.0
                                
                                add_column(u_vec, v_vec, w_vec)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

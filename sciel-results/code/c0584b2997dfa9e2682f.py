import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We will build lists of terms (u_row, v_row, w_row, coeff)
    terms = []
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_2x2x2(u_off, v_off, w_off, coeff=1):
        # u_off: index of A[0,0] in U
        # v_off: index of B[0,0] in V
        # w_off: index of C[0,0] in W
        
        # Strassen vectors
        u_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        v_vecs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        w_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        # Map indices
        # A is 2x2. Rows 0,1. Cols 0,1.
        # U index for A[i,j] is u_off + i*2 + j? 
        # Wait, U rows are global i*m + j.
        # If we pass u_off as the global index of A[0,0],
        # then A[0,1] is u_off + 1.
        # A[1,0] is u_off + m.
        # A[1,1] is u_off + m + 1.
        
        # So for a 2x2 block starting at global (i0, j0):
        # u_idx[0,0] = i0*m + j0
        # u_idx[0,1] = i0*m + j0 + 1
        # u_idx[1,0] = (i0+1)*m + j0
        # u_idx[1,1] = (i0+1)*m + j0 + 1
        
        # We need to pass i0, j0, k0, l0 or similar.
        # Let's change signature to pass global indices of top-left corner.
        
    # Revised helper
    def add_strassen_block(i0, j0, k0, l0, coeff=1):
        # i0: row in A
        # j0: col in A
        # k0: row in B
        # l0: col in B
        # Result is in C at row i0, col l0
        
        u_base = i0 * m + j0
        v_base = k0 * p + l0
        w_base = l0 * n + i0
        
        u_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        v_vecs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        w_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        # Indices for A
        u_idx = [
            [u_base, u_base + 1, u_base + m, u_base + m + 1]
        ]
        # Flatten u_idx to correspond to u_vecs
        # u_vecs[c] has 4 entries.
        # Entry 0: A[0,0] -> u_base
        # Entry 1: A[0,1] -> u_base + 1
        # Entry 2: A[1,0] -> u_base + m
        # Entry 3: A[1,1] -> u_base + m + 1
        
        v_idx = [
            [v_base, v_base + 1, v_base + p, v_base + p + 1]
        ]
        
        w_idx = [
            [w_base, w_base + 1, w_base + n, w_base + n + 1]
        ]
        
        for c in range(7):
            for e in range(4):
                cu = u_vecs[c][e]
                cv = v_vecs[c][e]
                cw = w_vecs[c][e]
                
                # We need to add a term for each non-zero component?
                # No, the term is a product of sums.
                # The decomposition is sum of rank-1 tensors.
                # Each rank-1 tensor is defined by vectors u_c, v_c, w_c.
                # So we add one term per c.
                # The vectors u_c, v_c, w_c are sparse.
                # We can just add the term with the full vectors?
                # The interface expects U, V, W matrices.
                # So we add to U[u_idx, c_global] += cu * coeff
                # etc.
                
                # We need a global column index for this term.
                # We will collect terms in a list first.
                pass

    # Let's collect terms as (u_row, val_u, v_row, val_v, w_row, val_w)
    # Actually, it's easier to just build the matrices directly if we know the rank.
    # Rank is 78.
    
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_row, v_row, w_row, coeff):
        nonlocal col
        if col >= rank:
            return
        U[u_row][col] += coeff
        V[v_row][col] += coeff
        W[w_row][col] += coeff
        col += 1
        
    # This is not quite right. A Strassen term is one rank-1 tensor.
    # But the vectors have multiple non-zero entries.
    # So one "term" in the rank decomposition corresponds to one column in U, V, W.
    # But the entries in that column can be non-zero in multiple rows.
    
    # So we should have a function add_strassen_term(u_idx_map, v_idx_map, w_idx_map, coeff)
    # where u_idx_map is a list of (row, val).
    
    def add_rank1(u_entries, v_entries, w_entries, coeff):
        nonlocal col
        if col >= rank:
            return
        for r, v in u_entries:
            U[r][col] += v * coeff
        for r, v in v_entries:
            V[r][col] += v * coeff
        for r, v in w_entries:
            W[r][col] += v * coeff
        col += 1
        
    def add_strassen(i0, j0, k0, l0, coeff=1):
        u_base = i0 * m + j0
        v_base = k0 * p + l0
        w_base = l0 * n + i0
        
        u_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        v_vecs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        w_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        u_rows = [u_base, u_base + 1, u_base + m, u_base + m + 1]
        v_rows = [v_base, v_base + 1, v_base + p, v_base + p + 1]
        w_rows = [w_base, w_base + 1, w_base + n, w_base + n + 1]
        
        for c in range(7):
            u_ent = [(u_rows[e], u_vecs[c][e]) for e in range(4) if u_vecs[c][e] != 0]
            v_ent = [(v_rows[e], v_vecs[c][e]) for e in range(4) if v_vecs[c][e] != 0]
            w_ent = [(w_rows[e], w_vecs[c][e]) for e in range(4) if w_vecs[c][e] != 0]
            add_rank1(u_ent, v_ent, w_ent, coeff)

    def add_schoolbook_1x2x2(i0, j0, k0, l0, coeff=1):
        # A is 1x2, B is 2x2, C is 1x2
        # Rank 4
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        # Terms:
        # 1: A[0,0]B[0,0] -> C[0,0]
        # 2: A[0,1]B[1,0] -> C[0,0]
        # 3: A[0,0]B[0,1] -> C[0,1]
        # 4: A[0,1]B[1,1] -> C[0,1]
        
        u0 = i0 * m + j0
        u1 = i0 * m + j0 + 1
        
        v00 = k0 * p + l0
        v01 = k0 * p + l0 + 1
        v10 = (k0 + 1) * p + l0
        v11 = (k0 + 1) * p + l0 + 1
        
        w0 = l0 * n + i0
        w1 = (l0 + 1) * n + i0
        
        add_rank1([(u0, 1)], [(v00, 1)], [(w0, 1)], coeff)
        add_rank1([(u1, 1)], [(v10, 1)], [(w0, 1)], coeff)
        add_rank1([(u0, 1)], [(v01, 1)], [(w1, 1)], coeff)
        add_rank1([(u1, 1)], [(v11, 1)], [(w1, 1)], coeff)

    def add_schoolbook_1x2x3(i0, j0, k0, l0, coeff=1):
        # A is 1x2, B is 2x3, C is 1x3
        # Rank 6
        for col in range(3):
            add_schoolbook_1x2x2(i0, j0, k0, l0 + col, coeff)

    # C_TL: Rows 0,1. Cols 0,1,2,3.
    # Blocks:
    # A[0:2, 0:2] B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0, 0)
    # A[0:2, 2:4] B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 2, 2, 0)
    # A[0:2, 0:2] B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 0, 0, 2)
    # A[0:2, 2:4] B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 2, 2, 2)
    
    # C_TR: Rows 0,1. Cols 4,5,6.
    # A[0:2, 0:2] B[0:2, 4:6] -> C[0:2, 4:6] (2x2x2)
    add_strassen(0, 0, 0, 4)
    # A[0:2, 0:2] B[0:2, 6:7] -> C[0:2, 6:7] (2x2x1)
    # 2x2x1 rank 4
    # A[0,0]B[0,6] + A[0,1]B[1,6] -> C[0,6]
    # A[1,0]B[0,6] + A[1,1]B[1,6] -> C[1,6]
    # Terms:
    add_rank1([(0, 1)], [(6, 1)], [(6*3+0, 1)], 1) # A[0,0]B[0,6] -> C[0,6]
    add_rank1([(1, 1)], [(13, 1)], [(6*3+0, 1)], 1) # A[0,1]B[1,6] -> C[0,6]
    add_rank1([(4, 1)], [(6, 1)], [(6*3+1, 1)], 1) # A[1,0]B[0,6] -> C[1,6]
    add_rank1([(5, 1)], [(13, 1)], [(6*3+1, 1)], 1) # A[1,1]B[1,6] -> C[1,6]
    
    # A[0:2, 2:4] B[2:4, 4:6] -> C[0:2, 4:6] (2x2x2)
    add_strassen(0, 2, 2, 4)
    # A[0:2, 2:4] B[2:4, 6:7] -> C[0:2, 6:7] (2x2x1)
    add_rank1([(2, 1)], [(14, 1)], [(6*3+0, 1)], 1) # A[0,2]B[2,6] -> C[0,6]
    add_rank1([(3, 1)], [(21, 1)], [(6*3+0, 1)], 1) # A[0,3]B[3,6] -> C[0,6]
    add_rank1([(6, 1)], [(14, 1)], [(6*3+1, 1)], 1) # A[1,2]B[2,6] -> C[1,6]
    add_rank1([(7, 1)], [(21, 1)], [(6*3+1, 1)], 1) # A[1,3]B[3,6] -> C[1,6]

    # C_BL: Row 2. Cols 0,1,2,3.
    # A[2, 0:2] B[0:2, 0:2] -> C[2, 0:2] (1x2x2)
    add_schoolbook_1x2x2(2, 0, 0, 0)
    # A[2, 2:4] B[2:4, 0:2] -> C[2, 0:2] (1x2x2)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    # A[2, 0:2] B[0:2, 2:4] -> C[2, 2:4] (1x2x2)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    # A[2, 2:4] B[2:4, 2:4] -> C[2, 2:4] (1x2x2)
    add_schoolbook_1x2x2(2, 2, 2, 2)

    # C_BR: Row 2. Cols 4,5,6.
    # A[2, 0:2] B[0:2, 4:7] -> C[2, 4:7] (1x2x3)
    add_schoolbook_1x2x3(2, 0, 0, 4)
    # A[2, 2:4] B[2:4, 4:7] -> C[2, 4:7] (1x2x3)
    add_schoolbook_1x2x3(2, 2, 2, 4)

    # Check col
    # 28 + 7 + 4 + 7 + 4 + 16 + 12 = 78
    
    return {"U": U, "V": V, "W": W}


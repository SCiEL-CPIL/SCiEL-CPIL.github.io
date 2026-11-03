import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    U = [[0] * (n * m * p) for _ in range(nm)]
    V = [[0] * (n * m * p) for _ in range(mp)]
    W = [[0] * (n * m * p) for _ in range(pn)]
    
    rank = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal rank
        if rank >= n * m * p:
            return
        U[u_idx][rank] = u_val
        V[v_idx][rank] = v_val
        W[w_idx][rank] = w_val
        rank += 1

    # Helper to add schoolbook decomposition for a block
    # A_block is n_b x m_b, B_block is m_b x p_b
    # Global offsets: a_row, a_col for A; b_row, b_col for B; c_row, c_col for C
    def add_schoolblock(n_b, m_b, p_b, a_row, a_col, b_row, b_col, c_row, c_col):
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u_idx = (a_row + i) * m + (a_col + j)
                    v_idx = (b_row + j) * p + (b_col + k)
                    w_idx = (c_row + i) * p + (c_col + k) # Wait, W index is k*n + i?
                    # Task says: W[k*n+i, r]. So W index is (k * n + i).
                    # My W matrix is of size (p*n, rank).
                    # So W row index is k * n + i.
                    w_idx = k * n + i
                    # But wait, the C matrix is n x p.
                    # C[i, k] = sum_j A[i, j] B[j, k].
                    # The W matrix in the decomposition corresponds to the output matrix C.
                    # The index in W is for the entry of C.
                    # C has dimensions n x p.
                    # The task says W is (p*n, rank).
                    # And "output factor follows the official trace convention W[k*n+i,r]".
                    # This implies the index in W is k*n + i, where i is row (0..n-1) and k is col (0..p-1).
                    # So W[k*n + i] corresponds to C[i, k].
                    # In my add_term, w_idx should be the index into the W matrix.
                    # So w_idx = k * n + i.
                    add_term(u_idx, v_idx, w_idx, 1, 1, 1)

    # Helper to add Strassen decomposition for 2x2x2 block
    def add_strassenblock(a_row, a_col, b_row, b_col, c_row, c_col):
        # n_b=2, m_b=2, p_b=2
        # Local indices i,j in {0,1} for A, j,k in {0,1} for B, i,k in {0,1} for C.
        
        # U factors (2x2 matrices flattened)
        # L1: A11+A22 -> [[1,0],[0,1]]
        # L2: A21+A22 -> [[0,0],[1,1]]
        # L3: A11 -> [[1,0],[0,0]]
        # L4: A22 -> [[0,0],[0,1]]
        # L5: A11+A12 -> [[1,1],[0,0]]
        # L6: A21-A11 -> [[-1,0],[1,0]]
        # L7: A12-A22 -> [[0,1],[0,-1]]
        
        # V factors
        # L1: B11+B22 -> [[1,0],[0,1]]
        # L2: B11 -> [[1,0],[0,0]]
        # L3: B12-B22 -> [[0,1],[0,-1]]
        # L4: B21-B11 -> [[-1,0],[1,0]]
        # L5: B22 -> [[0,0],[0,1]]
        # L6: B11+B12 -> [[1,1],[0,0]]
        # L7: B21+B22 -> [[0,1],[1,1]]
        
        # W factors (2x2 matrices, index k*n+i)
        # W1: [[1,0],[0,1]] -> C11=1, C22=1. Indices: 0*2+0=0, 1*2+1=3.
        # W2: [[0,0],[1,-1]] -> C21=1, C22=-1. Indices: 0*2+1=1, 1*2+1=3.
        # W3: [[0,1],[0,1]] -> C12=1, C22=1. Indices: 1*2+0=2, 1*2+1=3.
        # W4: [[1,0],[1,0]] -> C11=1, C21=1. Indices: 0*2+0=0, 0*2+1=1.
        # W5: [[-1,1],[0,0]] -> C11=-1, C12=1. Indices: 0*2+0=0, 1*2+0=2.
        # W6: [[0,0],[0,1]] -> C22=1. Index: 1*2+1=3.
        # W7: [[1,0],[0,0]] -> C11=1. Index: 0*2+0=0.
        
        terms = [
            # (U_vals, V_vals, W_vals)
            # U_vals is dict {(i,j): val}, V_vals is dict {(j,k): val}, W_vals is dict {(i,k): val}
            # But we need to map to global indices.
            # It's easier to just list the non-zero entries.
            
            # Term 1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # Term 2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            # Term 3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            # Term 4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # Term 5
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # Term 6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # Term 7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
        ]
        
        for u_entries, v_entries, w_entries in terms:
            for (ui, uj, uv) in u_entries:
                u_idx = (a_row + ui) * m + (a_col + uj)
                add_term(u_idx, -1, -1, uv, 0, 0) # Placeholder, need to add all at once
            # This approach is flawed because add_term increments rank.
            # We need to add a full term (U, V, W) at once.
            
        # Let's rewrite add_term to accept the full set of entries for one rank step.
        
    # Redefine add_term
    def add_full_term(u_entries, v_entries, w_entries):
        nonlocal rank
        if rank >= n * m * p:
            return
        for (idx, val) in u_entries:
            U[idx][rank] += val
        for (idx, val) in v_entries:
            V[idx][rank] += val
        for (idx, val) in w_entries:
            W[idx][rank] += val
        rank += 1

    # Redefine add_schoolblock
    def add_schoolblock(n_b, m_b, p_b, a_row, a_col, b_row, b_col, c_row, c_col):
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    u_idx = (a_row + i) * m + (a_col + j)
                    v_idx = (b_row + j) * p + (b_col + k)
                    w_idx = k * n + i
                    add_full_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    # Redefine add_strassenblock
    def add_strassenblock(a_row, a_col, b_row, b_col, c_row, c_col):
        # n_b=2, m_b=2, p_b=2
        # Map local (i,j) to global u_idx
        # Map local (j,k) to global v_idx
        # Map local (i,k) to global w_idx
        
        def g_u(i, j): return (a_row + i) * m + (a_col + j)
        def g_v(j, k): return (b_row + j) * p + (b_col + k)
        def g_w(i, k): return k * n + i
        
        terms = [
            # T1
            ([(g_u(0,0), 1), (g_u(1,1), 1)], [(g_v(0,0), 1), (g_v(1,1), 1)], [(g_w(0,0), 1), (g_w(1,1), 1)]),
            # T2
            ([(g_u(1,0), 1), (g_u(1,1), 1)], [(g_v(0,0), 1)], [(g_w(1,0), 1), (g_w(1,1), -1)]),
            # T3
            ([(g_u(0,0), 1)], [(g_v(0,1), 1), (g_v(1,1), -1)], [(g_w(0,1), 1), (g_w(1,1), 1)]),
            # T4
            ([(g_u(1,1), 1)], [(g_v(1,0), 1), (g_v(0,0), -1)], [(g_w(0,0), 1), (g_w(1,0), 1)]),
            # T5
            ([(g_u(0,0), 1), (g_u(0,1), 1)], [(g_v(1,1), 1)], [(g_w(0,0), -1), (g_w(0,1), 1)]),
            # T6
            ([(g_u(1,0), 1), (g_u(0,0), -1)], [(g_v(0,0), 1), (g_v(0,1), 1)], [(g_w(1,1), 1)]),
            # T7
            ([(g_u(0,1), 1), (g_u(1,1), -1)], [(g_v(1,0), 1), (g_v(1,1), 1)], [(g_w(0,0), 1)]),
        ]
        
        for u_entries, v_entries, w_entries in terms:
            add_full_term(u_entries, v_entries, w_entries)

    # Define block splits
    # A: 4x5. Rows: [0,1], [2,3]. Cols: [0,1], [2,3,4]
    a_blocks = [
        (0, 0, 2, 2), # A00: rows 0-1, cols 0-1
        (0, 2, 2, 3), # A01: rows 0-1, cols 2-4
        (2, 0, 2, 2), # A10: rows 2-3, cols 0-1
        (2, 2, 2, 3), # A11: rows 2-3, cols 2-4
    ]
    
    # B: 5x6. Rows: [0,1], [2,3,4]. Cols: [0,1], [2,3], [4,5]
    b_blocks = [
        (0, 0, 2, 2), # B00: rows 0-1, cols 0-1
        (0, 2, 2, 2), # B01: rows 0-1, cols 2-3
        (0, 4, 2, 2), # B02: rows 0-1, cols 4-5
        (2, 0, 3, 2), # B10: rows 2-4, cols 0-1
        (2, 2, 3, 2), # B11: rows 2-4, cols 2-3
        (2, 4, 3, 2), # B12: rows 2-4, cols 4-5
    ]
    
    # C: 4x6. Rows: [0,1], [2,3]. Cols: [0,1], [2,3], [4,5]
    c_blocks = [
        (0, 0, 2, 2), # C00
        (0, 2, 2, 2), # C01
        (0, 4, 2, 2), # C02
        (2, 0, 2, 2), # C10
        (2, 2, 2, 2), # C11
        (2, 4, 2, 2), # C12
    ]
    
    # Iterate over all combinations
    # A[I, J] * B[J, K] -> C[I, K]
    # I in {0, 1}, J in {0, 1}, K in {0, 1, 2}
    
    for I in range(2):
        for J in range(2):
            for K in range(3):
                a_row, a_col, n_b, m_b = a_blocks[I*2 + J]
                b_row, b_col, m_b2, p_b = b_blocks[J*3 + K]
                c_row, c_col, n_c, p_c = c_blocks[I*3 + K]
                
                # Check dimensions match
                if m_b != m_b2:
                    continue
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    add_strassenblock(a_row, a_col, b_row, b_col, c_row, c_col)
                else:
                    add_schoolblock(n_b, m_b, p_b, a_row, a_col, b_row, b_col, c_row, c_col)
    
    # Trim to actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

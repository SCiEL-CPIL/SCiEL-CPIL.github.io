"""Matrix multiplication 4x4x9 tensor decomposition using block decomposition."""

def _make_2x2_strassen():
    """Return (U, V, W) for 2x2x2 Strassen decomposition.
    Index conventions:
      U[i*2+j, r] for row i, col j of A
      V[j*2+k, r] for row j, col k of B
      W[k*2+i, r] for row k, col i of C (trace convention: W[k*n+i])
    """
    # 7 Strassen terms. Each term has:
    #   u: length 4 (2x2 A factors), v: length 4 (2x2 B factors), w: length 4 (2x2 C factors)
    # Strassen formulas:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B10-B11)
    #   M4 = A11*(B01-B11)
    #   M5 = (A00+A10)*B11
    #   M6 = (A01-A00)*(B00+B10)
    #   M7 = (A10-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Flattened indices for 2x2: row i, col j -> i*2+j
    # A: (00, 01, 10, 11) -> indices 0, 1, 2, 3
    # B: (00, 01, 10, 11) -> indices 0, 1, 2, 3
    # C: trace convention W[k*n+i] where n=2, so W[k*2+i]
    #   C00 at k=0,i=0 -> 0; C01 at k=0,i=1 -> 1; C10 at k=1,i=0 -> 2; C11 at k=1,i=1 -> 3
    
    # u[t][a_idx], v[t][b_idx], w[t][c_idx] for t in range(7)
    U = [[0.0]*4 for _ in range(7)]
    V = [[0.0]*4 for _ in range(7)]
    W = [[0.0]*4 for _ in range(7)]
    
    # M1: u = [1,0,0,1], v = [1,0,0,1], w = [1,0,0,1] (contributes + to C00, + to C11)
    # Actually let's be more careful.
    # M1 = (A00+A11)*(B00+B11)
    #   u: A00=1, A11=1 -> [1,0,0,1]
    #   v: B00=1, B11=1 -> [1,0,0,1]
    #   This product appears in C00 (coeff +1) and C11 (coeff +1)
    #   So w: C00=+1, C11=+1 -> [1,0,0,1]
    
    # M2 = (A10+A11)*B00
    #   u: A10=1, A11=1 -> [0,0,1,1]
    #   v: B00=1 -> [1,0,0,0]
    #   Appears in C10 (+1), C11 (-1)
    #   w: C10=+1, C11=-1 -> [0,0,1,-1]
    
    # M3 = A00*(B10-B11)
    #   u: A00=1 -> [1,0,0,0]
    #   v: B10=1, B11=-1 -> [0,0,1,-1]
    #   Appears in C01 (+1), C11 (+1)
    #   w: C01=+1, C11=+1 -> [0,1,0,1]
    
    # M4 = A11*(B01-B11)
    #   u: A11=1 -> [0,0,0,1]
    #   v: B01=1, B11=-1 -> [0,1,0,-1]
    #   Appears in C00 (+1), C10 (+1)
    #   w: C00=+1, C10=+1 -> [1,0,1,0]
    
    # M5 = (A00+A10)*B11
    #   u: A00=1, A10=1 -> [1,0,1,0]
    #   v: B11=1 -> [0,0,0,1]
    #   Appears in C00 (-1), C01 (+1)
    #   w: C00=-1, C01=+1 -> [-1,1,0,0]
    
    # M6 = (A01-A00)*(B00+B10)
    #   u: A01=1, A00=-1 -> [-1,1,0,0]
    #   v: B00=1, B10=1 -> [1,0,1,0]
    #   Appears in C11 (+1)
    #   w: C11=+1 -> [0,0,0,1]
    
    # M7 = (A10-A11)*(B01+B11)
    #   u: A10=1, A11=-1 -> [0,0,1,-1]
    #   v: B01=1, B11=1 -> [0,1,0,1]
    #   Appears in C00 (+1)
    #   w: C00=+1 -> [1,0,0,0]
    
    for t, (u_vals, v_vals, w_vals) in enumerate([
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # M2
        ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),   # M3
        ([0,0,0,1], [0,1,0,-1], [1,0,1,0]),   # M4
        ([1,0,1,0], [0,0,0,1], [-1,1,0,0]),   # M5
        ([-1,1,0,0], [1,0,1,0], [0,0,0,1]),   # M6
        ([0,0,1,-1], [0,1,0,1], [1,0,0,0]),   # M7
    ]):
        for idx in range(4):
            U[t][idx] = u_vals[idx]
            V[t][idx] = v_vals[idx]
            W[t][idx] = w_vals[idx]
    
    return U, V, W


def _make_2x2_identity_terms(n_blocks_a, n_blocks_b):
    """Create identity-like terms for multiplying block-diagonal structures.
    For multiplying a n_blocks_a x n_blocks_b "block matrix" where each block is 2x2,
    we need diagonal terms that pass through the 2x2 structure."""
    U = []
    V = []
    W = []
    for i in range(n_blocks_a):
        for j in range(n_blocks_b):
            # u: only A block (i,j) active, with identity 2x2
            # But we only need this for the "identity" part where block (i,j) maps to output (i,j)
            u = [0.0]*4
            v = [0.0]*4
            w = [0.0]*4
            u[0] = 1  # A00
            v[0] = 1  # B00
            # W index: k*2+i, where the output block position is (j,i) in block space
            # Within the 2x2 block, we need C = A*B, which is 4 terms for identity pass-through
            # Actually this gets complicated. Let's use a different approach.
            pass
    return U, V, W


def construct():
    """Construct a low-rank tensor decomposition for 4x4 x 4x9 matrix multiplication.
    
    Uses block decomposition: split the 4x9 matrix B into 3 blocks of 4x3 (column-wise).
    Each 4x4 * 4x3 product can be handled with a specialized decomposition.
    
    Alternative: Use Strassen for 4x4x4 blocks and identity for the rest.
    
    Actually, let's use a cleaner approach:
    - Split B (4x9) into columns: B = [B1, B2, B3] where each Bi is 4x3.
    - C = A * B = [A*B1, A*B2, A*B3]
    - For each 4x4 * 4x3 product, we can use a decomposition.
    
    Better approach: Use the fact that 4x4x4 has rank 49 (Strassen-based) and 4x4x1 has rank 16.
    4x9 = 4x4 + 4x4 + 4x1 (splitting 9 = 4+4+1)
    Rank = 49 + 49 + 16 = 114.
    
    Even better: 9 = 3+3+3, so 4x3 blocks.
    4x4x3 can be done with rank 32 (or so).
    Let's think: 4x4x3. The schoolbook rank is 4*4*3 = 48.
    Using Strassen on 4x4 gives 49 terms for 4x4x4. For 4x4x3, we can use 3/4 of the B factors.
    
    Actually, let me just implement the 49+49+16 = 114 approach cleanly.
    """
    import numpy as np
    
    n, m, p = 4, 4, 9
    
    # We'll decompose into three parts:
    # Part 1: columns 0-3 of B (4x4 block) -> rank 49 using Strassen 4x4x4
    # Part 2: columns 4-7 of B (4x4 block) -> rank 49 using Strassen 4x4x4  
    # Part 3: column 8 of B (4x1 block) -> rank 16 using schoolbook 4x4x1
    
    total_rank = 49 + 49 + 16
    U = np.zeros((n*m, total_rank))  # 16 x rank
    V = np.zeros((m*p, total_rank))  # 36 x rank
    W = np.zeros((p*n, total_rank))  # 36 x rank
    
    # --- Part 1: 4x4x4 Strassen decomposition for columns 0-3 ---
    # Build 4x4x4 using 4 2x2x2 Strassen decompositions (one per 2x2 block pair)
    # Actually, the standard way to get rank 49 for 4x4x4:
    # Split into 2x2 blocks: A has 4 blocks (AA, AB; BA, BB), B has 4 blocks, C has 4 blocks.
    # For each of the 7 Strassen products, we need 2x2 2x2 Strassen decompositions.
    # That gives 7*7 = 49 terms.
    
    U2, V2, W2 = _make_2x2_strassen()
    
    # Block structure for 4x4:
    # A blocks: top-left (rows 0-1, cols 0-1), top-right (rows 0-1, cols 2-3),
    #           bottom-left (rows 2-3, cols 0-1), bottom-right (rows 2-3, cols 2-3)
    # In flattened U[i*m+j]: 
    #   top-left: i in 0-1, j in 0-1 -> indices 0-3
    #   top-right: i in 0-1, j in 2-3 -> indices 4-7
    #   bottom-left: i in 2-3, j in 0-1 -> indices 8-11
    #   bottom-right: i in 2-3, j in 2-3 -> indices 12-15
    
    # B blocks: similarly for V[j*p+k] where p=4 for this subproblem
    #   top-left: j in 0-1, k in 0-1 -> indices 0-3
    #   top-right: j in 0-1, k in 2-3 -> indices 4-7
    #   bottom-left: j in 2-3, k in 0-1 -> indices 8-11
    #   bottom-right: j in 2-3, k in 2-3 -> indices 12-15
    
    # C blocks: W[k*n+i] with n=4
    #   top-left: k in 0-1, i in 0-1 -> indices 0-3
    #   top-right: k in 0-1, i in 2-3 -> indices 4-7
    #   bottom-left: k in 2-3, i in 0-1 -> indices 8-11
    #   bottom-right: k in 2-3, i in 2-3 -> indices 12-15
    
    # Strassen 4x4x4 formula:
    #   M1 = (A_tl + A_br) * (B_tl + B_br)
    #   M2 = (A_bl + A_br) * B_tl
    #   M3 = A_tl * (B_bl - B_br)
    #   M4 = A_br * (B_tr - B_br)
    #   M5 = (A_tl + A_bl) * B_br
    #   M6 = (A_tr - A_tl) * (B_tl + B_bl)
    #   M7 = (A_bl - A_br) * (B_tr + B_br)
    #   
    #   C_tl = M1 + M4 - M5 + M7
    #   C_tr = M3 + M5
    #   C_bl = M2 + M4
    #   C_br = M1 - M2 + M3 + M6
    
    # Each Mt is a 2x2x2 product, decomposed with rank 7 using the 2x2 Strassen above.
    # For Mt, the A-block and B-block are each 2x2 matrices, and the result contributes
    # to one or more C-blocks.
    
    # Let's enumerate all 49 terms:
    # Outer term t (0-6), inner Strassen term t' (0-6)
    # For outer term t, the A-block combination gives a 2x2 matrix, similarly for B-block.
    # The inner Strassen decomposes this 2x2x2 product into 7 rank-1 terms.
    # Each inner term maps to a contribution in the C-block space.
    
    col_offset = 0
    for t_outer in range(7):
        # Determine the A-block coefficients for this outer term
        # t_outer 0: A_tl=1, A_br=1, others 0
        # t_outer 1: A_bl=1, A_br=1, others 0
        # t_outer 2: A_tl=1, others 0
        # t_outer 3: A_br=1, others 0
        # t_outer 4: A_tl=1, A_bl=1, others 0
        # t_outer 5: A_tr=1, A_tl=-1, others 0
        # t_outer 6: A_bl=1, A_br=-1, others 0
        
        # A-block signs: dict from block_name to sign
        a_blocks = {0: {'tl': 1, 'br': 1}, 1: {'bl': 1, 'br': 1}, 2: {'tl': 1},
                     3: {'br': 1}, 4: {'tl': 1, 'bl': 1}, 5: {'tr': 1, 'tl': -1},
                     6: {'bl': 1, 'br': -1}}[t_outer]
        
        # B-block signs
        b_blocks = {0: {'tl': 1, 'br': 1}, 1: {'tl': 1}, 2: {'bl': 1, 'br': -1},
                     3: {'tr': 1, 'br': -1}, 4: {'br': 1}, 5: {'tl': 1, 'bl': 1},
                     6: {'tr': 1, 'br': 1}}[t_outer]
        
        # C-block contributions for this outer term
        c_blocks = {0: {'tl': 1, 'br': 1}, 1: {'bl': 1, 'br': -1}, 2: {'tr': 1, 'br': 1},
                     3: {'tl': 1, 'bl': 1}, 4: {'tl': -1, 'tr': 1}, 5: {'br': 1},
                     6: {'tl': 1}}[t_outer]
        
        for t_inner in range(7):
            u_vec = U2[t_inner]  # length 4
            v_vec = V2[t_inner]  # length 4
            w_vec = W2[t_inner]  # length 4
            
            # Build global U vector (length 16)
            U_local = np.zeros(16)
            # Map 2x2 block indices to global indices
            # tl: rows 0-1, cols 0-1 -> U[0:4]
            # tr: rows 0-1, cols 2-3 -> U[4:8]
            # bl: rows 2-3, cols 0-1 -> U[8:12]
            # br: rows 2-3, cols 2-3 -> U[12:16]
            block_map_a = {'tl': (0, 1, 0, 1), 'tr': (0, 1, 2, 3),
                           'bl': (2, 3, 0, 1), 'br': (2, 3, 2, 3)}
            for blk_name, sign in a_blocks.items():
                r1, r2, c1, c2 = block_map_a[blk_name]
                # Within the 2x2 block, index = row*2 + col, mapping to:
                # local 0 -> (r1, c1), local 1 -> (r1, c2), local 2 -> (r2, c1), local 3 -> (r2, c2)
                local_indices = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
                for li, (rr, cc) in enumerate(local_indices):
                    U_local[rr * 4 + cc] = u_vec[li] * sign
            
            # Build global V vector (length 36 for full problem, but 16 for this sub-block)
            V_local = np.zeros(16)  # Will be placed into correct columns later
            block_map_b = {'tl': (0, 1, 0, 1), 'tr': (0, 1, 2, 3),
                           'bl': (2, 3, 0, 1), 'br': (2, 3, 2, 3)}
            for blk_name, sign in b_blocks.items():
                r1, r2, c1, c2 = block_map_b[blk_name]
                local_indices = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
                for li, (rr, cc) in enumerate(local_indices):
                    V_local[rr * 4 + cc] = v_vec[li] * sign
            
            # Build global W vector (length 16 for 4x4 C)
            W_local = np.zeros(16)
            # C block map: tl -> (rows 0-1, cols 0-1) in C
            # W[k*n+i] so for C row k, col i: W[k*4+i]
            block_map_c = {'tl': (0, 1, 0, 1), 'tr': (0, 1, 2, 3),
                           'bl': (2, 3, 0, 1), 'br': (2, 3, 2, 3)}
            for blk_name, sign in c_blocks.items():
                r1, r2, c1, c2 = block_map_c[blk_name]
                local_indices = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
                for li, (rr, cc) in enumerate(local_indices):
                    W_local[rr * 4 + cc] = w_vec[li] * sign
            
            # Place into global arrays
            U[:, col_offset] = U_local
            V_local_full = np.zeros(36)
            V_local_full[0:16] = V_local  # Columns 0-3 of B
            V[:, col_offset] = V_local_full
            W_local_full = np.zeros(36)
            W_local_full[0:16] = W_local  # Columns 0-3 of C
            W[:, col_offset] = W_local_full
            col_offset += 1
    
    # --- Part 2: 4x4x4 Strassen decomposition for columns 4-7 ---
    for t_outer in range(7):
        a_blocks = {0: {'tl': 1, 'br': 1}, 1: {'bl': 1, 'br': 1}, 2: {'tl': 1},
                     3: {'br': 1}, 4: {'tl': 1, 'bl': 1}, 5: {'tr': 1, 'tl': -1},
                     6: {'bl': 1, 'br': -1}}[t_outer]
        b_blocks = {0: {'tl': 1, 'br': 1}, 1: {'tl': 1}, 2: {'bl': 1, 'br': -1},
                     3: {'tr': 1, 'br': -1}, 4: {'br': 1}, 5: {'tl': 1, 'bl': 1},
                     6: {'tr': 1, 'br': 1}}[t_outer]
        c_blocks = {0: {'tl': 1, 'br': 1}, 1: {'bl': 1, 'br': -1}, 2: {'tr': 1, 'br': 1},
                     3: {'tl': 1, 'bl': 1}, 4: {'tl': -1, 'tr': 1}, 5: {'br': 1},
                     6: {'tl': 1}}[t_outer]
        
        for t_inner in range(7):
            u_vec = U2[t_inner]
            v_vec = V2[t_inner]
            w_vec = W2[t_inner]
            
            U_local = np.zeros(16)
            block_map_a = {'tl': (0, 1, 0, 1), 'tr': (0, 1, 2, 3),
                           'bl': (2, 3, 0, 1), 'br': (2, 3, 2, 3)}
            for blk_name, sign in a_blocks.items():
                r1, r2, c1, c2 = block_map_a[blk_name]
                local_indices = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
                for li, (rr, cc) in enumerate(local_indices):
                    U_local[rr * 4 + cc] = u_vec[li] * sign
            
            V_local = np.zeros(16)
            block_map_b = {'tl': (0, 1, 0, 1), 'tr': (0, 1, 2, 3),
                           'bl': (2, 3, 0, 1), 'br': (2, 3, 2, 3)}
            for blk_name, sign in b_blocks.items():
                r1, r2, c1, c2 = block_map_b[blk_name]
                local_indices = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
                for li, (rr, cc) in enumerate(local_indices):
                    V_local[rr * 4 + cc] = v_vec[li] * sign
            
            W_local = np.zeros(16)
            block_map_c = {'tl': (0, 1, 0, 1), 'tr': (0, 1, 2, 3),
                           'bl': (2, 3, 0, 1), 'br': (2, 3, 2, 3)}
            for blk_name, sign in c_blocks.items():
                r1, r2, c1, c2 = block_map_c[blk_name]
                local_indices = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
                for li, (rr, cc) in enumerate(local_indices):
                    W_local[rr * 4 + cc] = w_vec[li] * sign
            
            U[:, col_offset] = U_local
            V_local_full = np.zeros(36)
            V_local_full[16:32] = V_local  # Columns 4-7 of B: j*4+k, j in 0-3, k in 0-3 -> offset 16
            # Wait, V index is j*p+k where p=9. So for column block 4-7:
            # B row j (0-3), column 4+k (k=0-3): index = j*9 + (4+k)
            V_block = np.zeros(36)
            for blk_name, sign in b_blocks.items():
                r1, r2, c1, c2 = block_map_b[blk_name]
                local_indices = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
                for li, (rr, cc) in enumerate(local_indices):
                    global_idx = rr * 9 + (4 + cc)
                    V_block[global_idx] = v_vec[li] * sign
            V[:, col_offset] = V_block
            
            # C columns 4-7: W[k*4+i], k in 0-3, i in 0-3, but offset by 4 in column
            # W global index: k*9 + i (since n=4, p=9, W is p*n = 9*4 = 36)
            # Wait, W shape is (p*n, rank) = (36, rank). W[k*n+i] = W[k*4+i].
            # For C block with column offset 4: C[k, 4+i]
            # W global index = k*4 + (4+i)? No.
            # W is indexed as W[k*n + i] where k is column index (0-8), i is row index (0-3).
            # So W[k*4 + i] for column k, row i.
            # For columns 4-7: k in 4-7, i in 0-3 -> W[k*4+i]
            W_block = np.zeros(36)
            for blk_name, sign in c_blocks.items():
                r1, r2, c1, c2 = block_map_c[blk_name]
                local_indices = [(r1, c1), (r1, c2), (r2, c1), (r2, c2)]
                for li, (rr, cc) in enumerate(local_indices):
                    # rr is the row in C (0-3), cc is the column offset within block (0-3)
                    # Global column = 4 + cc, global row = rr
                    global_idx = (4 + cc) * 4 + rr
                    W_block[global_idx] = w_vec[li] * sign
            W[:, col_offset] = W_block
            col_offset += 1
    
    # --- Part 3: 4x4x1 schoolbook for column 8 ---
    # Rank 16: for each (i, j), U[i*4+j]=1, V[j*9+8]=1, W[8*4+i]=1
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col_offset] = 1
            V[j * 9 + 8, col_offset] = 1
            W[8 * 4 + i, col_offset] = 1
            col_offset += 1
    
    # Convert to lists of lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

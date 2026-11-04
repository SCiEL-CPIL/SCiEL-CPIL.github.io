import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition: 5 = 2 + 3
    # Blocks:
    # A11: 2x2, A12: 2x3
    # A21: 3x2, A22: 3x3
    # Same for B and C.
    
    # Offsets
    # i indices: 0,1 for block 0; 2,3,4 for block 1
    # j indices: 0,1 for block 0; 2,3,4 for block 1
    # k indices: 0,1 for block 0; 2,3,4 for block 1
    
    blocks = [
        # (row_block, col_block, size_i, size_j, size_k)
        # C_rc = A_r* B_*c
        # A_r* is row r, B_*c is col c
        # A11 (2x2) * B11 (2x2) -> C11 (2x2)
        (0, 0, 2, 2, 2),
        # A12 (2x3) * B21 (3x2) -> C11 (2x2)
        (0, 1, 2, 3, 2),
        # A11 (2x2) * B12 (2x3) -> C12 (2x3)
        (0, 0, 2, 2, 3),
        # A12 (2x3) * B22 (3x3) -> C12 (2x3)
        (0, 1, 2, 3, 3),
        # A21 (3x2) * B11 (2x2) -> C21 (3x2)
        (1, 0, 3, 2, 2),
        # A22 (3x3) * B21 (3x2) -> C21 (3x2)
        (1, 1, 3, 3, 2),
        # A21 (3x2) * B12 (2x3) -> C22 (3x3)
        (1, 0, 3, 2, 3),
        # A22 (3x3) * B22 (3x3) -> C22 (3x3)
        (1, 1, 3, 3, 3),
    ]
    
    # Helper to get offsets
    def get_offset(block_idx, size, total_size):
        # Block 0 is first 'size' elements
        # Block 1 is next elements
        # This is for 1D indices
        if block_idx == 0:
            return 0
        else:
            # Sum of sizes of previous blocks
            # In our case, only 2 blocks: 2 and 3
            # If block_idx is 1, offset is 2
            # We need to know the size of the previous block.
            # For i: block 0 is 2, block 1 is 3. Offset for 1 is 2.
            # For j: block 0 is 2, block 1 is 3. Offset for 1 is 2.
            # For k: block 0 is 2, block 1 is 3. Offset for 1 is 2.
            # This logic is specific to the 2+3 split.
            return 2

    U_list = []
    V_list = []
    W_list = []
    
    # Precompute Strassen decomposition for 2x2x2
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to generate U, V, W for a specific block multiplication.
    # For 2x2x2, we use Strassen.
    # For others, we use Schoolbook.
    
    def get_strassen_222():
        # Returns U, V, W for 2x2x2 multiplication using Strassen (rank 7)
        # Local indices i,j,k in 0..1
        # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        
        # Define the 7 terms
        # Each term is (u_vec, v_vec, w_vec)
        # u_vec is length 4 (2x2), v_vec is length 4, w_vec is length 4
        
        terms = []
        
        # M1 = (A11+A22)(B11+B22)
        # A part: A11 + A22 -> [1, 0, 0, 1] (flat: 00, 01, 10, 11)
        # B part: B11 + B22 -> [1, 0, 0, 1]
        # C part: contributes to C11, C22
        # C11 += 1, C22 += 1
        # w_vec: C11 is index 0 (k=0, i=0), C22 is index 3 (k=1, i=1)
        # W[k*2+i]
        # C11: k=0, i=0 -> 0
        # C22: k=1, i=1 -> 3
        # w = [1, 0, 0, 1]
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        
        # M2 = (A21+A22)B11
        # A part: A21 + A22 -> [0, 0, 1, 1]
        # B part: B11 -> [1, 0, 0, 0]
        # C part: C21 += 1
        # C21: k=0, i=1 -> 1
        # w = [0, 1, 0, 0]
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]))
        
        # M3 = A11(B12-B22)
        # A part: A11 -> [1, 0, 0, 0]
        # B part: B12 - B22 -> [0, 1, 0, -1]
        # C part: C12 += 1
        # C12: k=1, i=0 -> 2
        # w = [0, 0, 1, 0]
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]))
        
        # M4 = A22(B21-B11)
        # A part: A22 -> [0, 0, 0, 1]
        # B part: B21 - B11 -> [-1, 0, 1, 0]
        # C part: C11 += 1, C21 += 1
        # C11: 0, C21: 1
        # w = [1, 1, 0, 0]
        terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]))
        
        # M5 = (A11+A12)B22
        # A part: A11 + A12 -> [1, 1, 0, 0]
        # B part: B22 -> [0, 0, 0, 1]
        # C part: C11 -= 1, C12 += 1
        # C11: 0, C12: 2
        # w = [-1, 0, 1, 0]
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]))
        
        # M6 = (A21-A11)(B11+B12)
        # A part: A21 - A11 -> [-1, 0, 1, 0]
        # B part: B11 + B12 -> [1, 1, 0, 0]
        # C part: C22 += 1
        # C22: 3
        # w = [0, 0, 0, 1]
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        
        # M7 = (A12-A22)(B21+B22)
        # A part: A12 - A22 -> [0, 1, 0, -1]
        # B part: B21 + B22 -> [0, 0, 1, 1]
        # C part: C11 += 1
        # C11: 0
        # w = [1, 0, 0, 0]
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        for r in range(7):
            u, v, w = terms[r]
            for idx in range(4):
                U[idx][r] = u[idx]
                V[idx][r] = v[idx]
                W[idx][r] = w[idx]
                
        return U, V, W

    def get_schoolbook(n, m, p):
        # Returns U, V, W for n x m x p multiplication using Schoolbook (rank n*m*p)
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W

    for (rb, cb, ni, nj, nk) in blocks:
        # ni: size of A row block (rows of C)
        # nj: size of A col block / B row block (middle index)
        # nk: size of B col block (cols of C)
        
        # Get local decomposition
        if ni == 2 and nj == 2 and nk == 2:
            u_loc, v_loc, w_loc = get_strassen_222()
        else:
            u_loc, v_loc, w_loc = get_schoolbook(ni, nj, nk)
            
        # Map to global indices
        # i_global = offset_i(rb) + i_loc
        # j_global = offset_j(cb) + j_loc  <-- WAIT.
        # The middle index j is the column of A and row of B.
        # A is n x m. B is m x p.
        # In our block definition:
        # A_rb, cb is ni x nj.
        # B_cb, ck is nj x nk.
        # So the middle index is nj.
        # For A, j ranges 0..nj-1. Global j = offset_j(cb) + j_loc.
        # For B, j ranges 0..nj-1. Global j = offset_j(cb) + j_loc.
        # Note: The block index for the middle dimension is the column of A / row of B.
        # In our loop, cb is the column block of A, which is the row block of B?
        # Let's check the block list:
        # (rb, cb, ni, nj, nk)
        # A_rb, cb: ni x nj.
        # B_cb, ? : The next block in B must have row index corresponding to A's column index.
        # So B's row block index is cb.
        # B's column block index is... determined by nk?
        # In the list:
        # (0, 0, 2, 2, 2): A11 * B11. A col 0, B row 0.
        # (0, 1, 2, 3, 2): A12 * B21. A col 1, B row 1.
        # So yes, the middle index block is cb.
        
        # Offsets
        # i offset: if rb==0, 0; else 2
        # j offset: if cb==0, 0; else 2
        # k offset: if nk is part of block 0 or 1?
        # The output C is ni x nk.
        # C's row index is i (from A's row block rb).
        # C's col index is k (from B's col block).
        # What is B's col block?
        # In (0,0,2,2,2), B is 2x2, so col block 0.
        # In (0,1,2,3,2), B is 3x2. A is 2x3.
        # A col 1 (size 3). B row 1 (size 3). B col ? (size 2).
        # B col block must be 0 (size 2).
        # So we need to know the column block of B.
        # It is not explicitly in the tuple.
        # But nk determines the size.
        # If nk == 2, it's block 0. If nk == 3, it's block 1.
        # Let's verify.
        # (0,0,2,2,2): nk=2 -> B col 0.
        # (0,1,2,3,2): nk=2 -> B col 0.
        # (0,0,2,2,3): nk=3 -> B col 1.
        # (0,1,2,3,3): nk=3 -> B col 1.
        # (1,0,3,2,2): nk=2 -> B col 0.
        # (1,1,3,3,2): nk=2 -> B col 0.
        # (1,0,3,2,3): nk=3 -> B col 1.
        # (1,1,3,3,3): nk=3 -> B col 1.
        
        # So:
        # i_off = 0 if rb==0 else 2
        # j_off = 0 if cb==0 else 2
        # k_off = 0 if nk==2 else 2
        
        i_off = 0 if rb == 0 else 2
        j_off = 0 if cb == 0 else 2
        k_off = 0 if nk == 2 else 2
        
        # Append to global lists
        rank_loc = len(u_loc[0])
        
        # U: size (ni*nj) x rank_loc
        # Global U index: (i_global * 5 + j_global)
        for r in range(rank_loc):
            for i_loc in range(ni):
                for j_loc in range(nj):
                    i_global = i_off + i_loc
                    j_global = j_off + j_loc
                    u_global_idx = i_global * 5 + j_global
                    U_list.append([0] * rank_loc) # Placeholder, we need to build full matrix
                    # Actually, it's easier to build the full matrices at the end.
                    # Let's store the local blocks and offsets.
                    
        # This approach of appending row by row is messy.
        # Let's collect the local U, V, W and offsets, then construct the final matrices.
        
        # Store: (u_loc, v_loc, w_loc, i_off, j_off, k_off, ni, nj, nk)
        # We'll do the mapping in a second pass.
        
    # Re-do the collection properly
    U_final = []
    V_final = []
    W_final = []
    
    # We need to know the total rank first to initialize?
    # Or we can just append columns.
    # U is 25 x R.
    
    total_rank = 0
    blocks_data = []
    for (rb, cb, ni, nj, nk) in blocks:
        if ni == 2 and nj == 2 and nk == 2:
            u_loc, v_loc, w_loc = get_strassen_222()
        else:
            u_loc, v_loc, w_loc = get_schoolbook(ni, nj, nk)
            
        i_off = 0 if rb == 0 else 2
        j_off = 0 if cb == 0 else 2
        k_off = 0 if nk == 2 else 2
        
        blocks_data.append((u_loc, v_loc, w_loc, i_off, j_off, k_off, ni, nj, nk))
        total_rank += len(u_loc[0])
        
    # Initialize final matrices
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    col_offset = 0
    for (u_loc, v_loc, w_loc, i_off, j_off, k_off, ni, nj, nk) in blocks_data:
        rank_loc = len(u_loc[0])
        
        # Fill U
        for r in range(rank_loc):
            for i_loc in range(ni):
                for j_loc in range(nj):
                    i_global = i_off + i_loc
                    j_global = j_off + j_loc
                    u_idx = i_global * 5 + j_global
                    U[u_idx][col_offset + r] = u_loc[i_loc * nj + j_loc][r]
                    
        # Fill V
        # V local index: j_loc * nk + k_loc?
        # No, V is m x p.
        # In local block, V is nj x nk.
        # Index: j_loc * nk + k_loc.
        # Global: j_global * 5 + k_global.
        for r in range(rank_loc):
            for j_loc in range(nj):
                for k_loc in range(nk):
                    j_global = j_off + j_loc
                    k_global = k_off + k_loc
                    v_idx = j_global * 5 + k_global
                    V[v_idx][col_offset + r] = v_loc[j_loc * nk + k_loc][r]
                    
        # Fill W
        # W local index: k_loc * ni + i_loc.
        # Global: k_global * 5 + i_global.
        for r in range(rank_loc):
            for k_loc in range(nk):
                for i_loc in range(ni):
                    k_global = k_off + k_loc
                    i_global = i_off + i_loc
                    w_idx = k_global * 5 + i_global
                    W[w_idx][col_offset + r] = w_loc[k_loc * ni + i_loc][r]
                    
        col_offset += rank_loc
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

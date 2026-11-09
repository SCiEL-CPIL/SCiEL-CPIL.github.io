import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will decompose the multiplication into blocks.
    # Blocks of type 2x2x2 using Strassen (rank 7)
    # Block of type 4x1x6 using schoolbook (rank 24)
    
    # Total rank calculation
    # 2 row-blocks (0-1, 2-3) x 2 col-blocks (0-1, 2-3) x 3 k-blocks (0-1, 2-3, 4-5) = 12 blocks of 2x2x2
    # 1 block of 4x1x6 (all rows, col 4, all k)
    
    rank_222 = 7
    num_222_blocks = 2 * 2 * 3
    rank_416 = 4 * 6
    total_rank = num_222_blocks * rank_222 + rank_416
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    current_col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # row_offset: starting row index in 0..3
    # col_offset: starting col index in 0..5
    # k_offset: starting k index in 0..5
    def add_strassen_block(row_off, col_off, k_off, col_start):
        nonlocal current_col
        # The 7 terms of Strassen decomposition
        # Each term is a tuple of (u_coefs, v_coefs, w_coefs)
        # u_coefs: dict or list of (local_row, val)
        # v_coefs: dict or list of (local_col, val)
        # w_coefs: dict or list of (local_result_pos, val)
        
        # Local indices for 2x2:
        # U rows: 0,1
        # V cols: 0,1 (indices into the 2x2 block of B)
        # W results: 00, 01, 10, 11 -> indices 0,1,2,3 in local 2x2 C
        
        terms = [
            # 1. M1 = a00 * (b00 + b11)
            ({0: 1}, {0: 1, 1: 1}, {0: 1, 3: 1}), # c00, c11
            # 2. M2 = (a10 + a11) * b00
            ({1: 1, 2: 1}, {0: 1}, {2: 1, 3: -1}), # c10, c11
            # 3. M3 = a11 * (b10 - b00)
            ({2: 1}, {1: 1, 0: -1}, {1: 1, 3: 1}), # c01, c11
            # 4. M4 = a00 * (b01 + b11)
            ({0: 1}, {1: 1, 1: 1}, {1: 1, 2: 1}), # c01, c10 -- Wait, check c indices.
            # Let's re-verify Strassen indices carefully.
            # C = [[c00, c01], [c10, c11]]
            # M1 = a00(b00+b11) -> c00, c11
            # M2 = (a10+a11)b00 -> c10, c11
            # M3 = a11(b10-b00) -> c01, c11
            # M4 = a00(b01+b11) -> c01, c10? No.
            # Standard Strassen:
            # M1 = a00 (b00 + b11)
            # M2 = (a10 + a11) b00
            # M3 = a11 (b10 - b00)
            # M4 = a00 (b01 + b11)
            # M5 = (a00 + a01) b11
            # M6 = (a10 - a00) (b00 + b01)
            # M7 = (a01 + a11) (b10 + b11)
            #
            # c00 = M1 + M4 - M5 - M7
            # c01 = M3 + M5
            # c10 = M2 + M4
            # c11 = M1 - M2 + M3 + M6
            
            # Let's map W coefficients (1 for +1, -1 for -1)
            # Term 1 (M1): U={0:1}, V={0:1, 1:1}, W={0:1, 3:1}
            # Term 2 (M2): U={1:1, 2:1}, V={0:1}, W={2:1, 3:-1}
            # Term 3 (M3): U={2:1}, V={1:1, 0:-1}, W={1:1, 3:1}
            # Term 4 (M4): U={0:1}, V={1:1, 1:1}, W={1:1, 2:1}
            # Term 5 (M5): U={0:1, 1:1}, V={1:1}, W={0:-1, 1:1}
            # Term 6 (M6): U={1:1, 0:-1}, V={0:1, 1:1}, W={3:1}
            # Term 7 (M7): U={1:1, 2:1}, V={1:1, 1:1}, W={0:-1, 3:1}
        ]
        
        # Correcting Term 4 W: c01 and c10 are indices 1 and 2.
        # c01 = M3 + M5. So M4 does NOT contribute to c01?
        # My formula for c01 was M3+M5.
        # My formula for c10 was M2+M4.
        # So M4 contributes to c10 (index 2) and c00?
        # c00 = M1 + M4 - M5 - M7. Yes, M4 contributes to c00 (index 0).
        # So Term 4 W: {0:1, 2:1}
        
        # Let's re-list Terms with correct W mappings:
        terms = [
            # 1: M1
            ({0: 1}, {0: 1, 1: 1}, {0: 1, 3: 1}),
            # 2: M2
            ({1: 1, 2: 1}, {0: 1}, {2: 1, 3: -1}),
            # 3: M3
            ({2: 1}, {1: 1, 0: -1}, {1: 1, 3: 1}),
            # 4: M4
            ({0: 1}, {1: 1, 1: 1}, {0: 1, 2: 1}),
            # 5: M5
            ({0: 1, 1: 1}, {1: 1}, {0: -1, 1: 1}),
            # 6: M6
            ({1: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            # 7: M7
            ({1: 1, 2: 1}, {1: 1, 1: 1}, {0: -1, 3: 1}),
        ]
        
        for u_loc, v_loc, w_loc in terms:
            # U: rows in block are row_off, row_off+1
            # Local row 0 -> global row_off, Local row 1 -> global row_off+1
            # U index is i*m + j. Here j is col_off, col_off+1.
            # Wait, U is (n*m, rank). Index is i*m + j.
            # In our block, i is fixed to row_off, row_off+1. j is fixed to col_off, col_off+1.
            # So for a U coefficient at local row r_loc (0 or 1), the global row i = row_off + r_loc.
            # The global column j in U is col_off + 0? No, U depends on i and j.
            # In the tensor decomposition, the U factor for the block at (i-range, j-range) 
            # has non-zeros only at U[i*m + j, c].
            
            for r_loc, val_u in u_loc.items():
                i_global = row_off + r_loc
                j_global = col_off # This is wrong. U index is i*m+j.
                # The U matrix maps input A[i,j] to U.
                # For a term involving A[i,j], the U entry is at (i*m + j).
                # In Strassen, the linear forms are combinations of A entries.
                # E.g., M2 involves a10 and a11.
                # a10 is at i=1, j=0 (local). a11 is at i=1, j=1 (local).
                # So for local row 1 (i=1), we have contributions from j=0 and j=1?
                # No, the keys in u_loc dict I defined earlier were {1:1, 2:1}.
                # That implied local indices 1 and 2. But local rows are only 0 and 1.
                # My u_loc keys were mixed up.
                # Let's redefine u_loc keys as (local_i, local_j) pairs?
                # Yes, U index is flattened (i, j).
                # So u_loc should be a dict: {(i_loc, j_loc): val}
                
            # I need to rebuild the terms with (i_loc, j_loc) keys.
            
            pass
            
        # Re-implementing terms with correct keys
        terms = [
            # 1: M1 = a00 (b00 + b11)
            ([( (0,0), 1 )], [((0,0), 1), ((0,1), 1)], [((0,0), 1), ((1,1), 1)]),
            # 2: M2 = (a10 + a11) b00
            ([( (1,0), 1 ), ((1,1), 1 )], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
            # 3: M3 = a11 (b10 - b00)
            ([( (1,1), 1 )], [((1,0), 1), ((0,0), -1)], [((0,1), 1), ((1,1), 1)]),
            # 4: M4 = a00 (b01 + b11)
            ([( (0,0), 1 )], [((0,1), 1), ((1,1), 1)], [((0,0), 1), ((1,0), 1)]),
            # 5: M5 = (a00 + a01) b11
            ([( (0,0), 1 ), ((0,1), 1 )], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
            # 6: M6 = (a10 - a00) (b00 + b01)
            ([( (1,0), 1 ), ((0,0), -1 )], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
            # 7: M7 = (a01 + a11) (b10 + b11)
            ([( (0,1), 1 ), ((1,1), 1 )], [((1,0), 1), ((1,1), 1)], [((0,0), -1), ((1,1), 1)]),
        ]
        
        for u_list, v_list, w_list in terms:
            col_idx = current_col
            for (i_loc, j_loc), val_u in u_list:
                i_g = row_off + i_loc
                j_g = col_off + j_loc
                U[i_g * m + j_g, col_idx] += val_u
                
            for (k_loc, p_loc), val_v in v_list:
                # V index is j*p + k.
                # j is local col of B, which corresponds to j_loc in A?
                # In Strassen, B is 2x2. B[i,j] where i is row of B (K-dim), j is col of B (P-dim).
                # In our problem, B is 5x6.
                # The block is at K=k_off..k_off+1, P=p_off..p_off+1.
                # So local k (row of B) is 0,1. Local p (col of B) is 0,1.
                # V index is j*p + k? No, V maps B[j,k] to V.
                # B row is j (middle dim), B col is k (last dim)?
                # Standard: A(n,m), B(m,p).
                # B[j,k] where j is 0..m-1, k is 0..p-1.
                # V index is j*p + k.
                # In the block, j is the local row of B? No, j is the global middle index.
                # The block uses B[j_off + j_loc, k_off + k_loc].
                # So for V, the index is (j_off + j_loc) * p + (k_off + k_loc).
                # My v_list keys were (0,0), (0,1), (1,0), (1,1).
                # Let's assume key (r, c) in v_list means B_local[r, c].
                # B_local row r corresponds to global j = col_off + r?
                # No, B_local row corresponds to the middle dimension j.
                # B_local col corresponds to the last dimension k.
                # So j_g = col_off + r.
                # k_g = k_off + c.
                j_g = col_off + r
                k_g = k_off + c
                V[j_g * p + k_g, col_idx] += val_v
                
            for (i_loc, k_loc), val_w in w_list:
                # W index is k*n + i.
                # W maps to C[i, k].
                # C_local[i_loc, k_loc].
                # i_g = row_off + i_loc.
                # k_g = k_off + k_loc.
                i_g = row_off + i_loc
                k_g = k_off + k_loc
                W[k_g * n + i_g, col_idx] += val_w
                
            current_col += 1
            # Ensure we didn't exceed rank
            if current_col > total_rank:
                raise ValueError("Rank exceeded")

    # Add the 12 Strassen blocks
    for row_off in [0, 2]:
        for col_off in [0, 2]:
            for k_off in [0, 2, 4]:
                add_strassen_block(row_off, col_off, k_off, current_col)
                
    # Add the 4x1x6 schoolbook block
    # This covers rows 0-3, col 4, k 0-5.
    for i in range(4):
        for k in range(6):
            col_idx = current_col
            U[i * m + 4, col_idx] = 1
            V[4 * p + k, col_idx] = 1
            W[k * n + i, col_idx] = 1
            current_col += 1
            
    if current_col != total_rank:
        raise ValueError("Column count mismatch")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

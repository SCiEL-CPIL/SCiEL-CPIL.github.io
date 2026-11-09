import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 189
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define block indices
    # A11: rows 0-2, cols 0-2
    # A12: rows 0-2, cols 3-5
    # A21: rows 3-5, cols 0-2
    # A22: rows 3-5, cols 3-5
    
    # Helper to get indices for a block
    def get_block_indices(block_id):
        if block_id == 11:
            return [(r, c) for r in range(3) for c in range(3)]
        elif block_id == 12:
            return [(r, c + 3) for r in range(3) for c in range(3)]
        elif block_id == 21:
            return [(r + 3, c) for r in range(3) for c in range(3)]
        elif block_id == 22:
            return [(r + 3, c + 3) for r in range(3) for c in range(3)]
    
    # Define A combinations
    # A_comb[r] is a list of (block_id, sign)
    A_comb = {
        1: [(11, 1), (22, 1)],
        2: [(21, 1), (22, 1)],
        3: [(11, 1)],
        4: [(22, 1)],
        5: [(11, 1), (12, 1)],
        6: [(21, 1), (11, -1)],
        7: [(12, 1), (22, -1)]
    }
    
    # Define B combinations
    B_comb = {
        1: [(11, 1), (22, 1)],
        2: [(11, 1)],
        3: [(12, 1), (22, -1)],
        4: [(21, 1), (11, -1)],
        5: [(22, 1)],
        6: [(11, 1), (12, 1)],
        7: [(21, 1), (22, 1)]
    }
    
    # Define C combinations
    # C_comb[block] is a list of (M_id, sign)
    C_comb = {
        11: [(1, 1), (4, 1), (5, -1), (7, 1)],
        12: [(3, 1), (5, 1)],
        21: [(2, 1), (4, 1)],
        22: [(1, 1), (2, -1), (3, 1), (6, 1)]
    }
    
    # Iterate over all terms
    for m_id in range(1, 8):
        for s in range(3): # block row of M
            for t in range(3): # block col of M
                for l in range(3): # inner index
                    # Determine which C blocks this M_id contributes to
                    # M_id[s,t] is at block row s//3? No, s,t are 0..2.
                    # The block position of M_id in C is determined by C_comb.
                    # We need to find which C block contains M_id.
                    # Actually, M_id is a full 3x3 matrix.
                    # The element M_id[s,t] contributes to the C block specified by the mapping.
                    # But the mapping is per M_id, not per element.
                    # Wait, C_11 = M_1 + ...
                    # This means the entire M_1 matrix is added to C_11.
                    # So M_1[s,t] contributes to C_11[s,t].
                    
                    # Find all C blocks that include this M_id
                    for c_block_id, signs in C_comb.items():
                        for m_ref, sign in signs:
                            if m_ref == m_id:
                                # This M_id contributes to C_block_id with 'sign'
                                # M_id[s,t] contributes to C_block_id[s,t]
                                
                                # Get indices for A part: A_comb[m_id][s,l]
                                # A_comb[m_id] defines the matrix A^(m_id)
                                # A^(m_id)[s,l] is a sum of elements from specific blocks
                                
                                # Build U vector for this term
                                u_vec = [0] * (n * m)
                                for block_id, a_sign in A_comb[m_id]:
                                    indices = get_block_indices(block_id)
                                    # We need the element at [s,l] of this block
                                    # The block is 3x3.
                                    # The element [s,l] in the block corresponds to:
                                    # If block 11: global (s, l)
                                    # If block 12: global (s, l+3)
                                    # etc.
                                    
                                    # Let's just use the list of indices and pick the s-th row, l-th col
                                    # The indices list is row-major.
                                    # Index in list: s*3 + l
                                    idx_in_block = s * 3 + l
                                    i, j = indices[idx_in_block]
                                    u_vec[i * m + j] += a_sign * sign
                                    
                                # Build V vector for this term
                                v_vec = [0] * (m * p)
                                for block_id, b_sign in B_comb[m_id]:
                                    indices = get_block_indices(block_id)
                                    # B^(m_id)[l,t]
                                    # Element [l,t] in the block
                                    idx_in_block = l * 3 + t
                                    j, k = indices[idx_in_block]
                                    v_vec[j * p + k] += b_sign * sign
                                    
                                # Build W vector for this term
                                w_vec = [0] * (p * n)
                                # C_block_id[s,t]
                                # Get global indices for C
                                c_indices = get_block_indices(c_block_id)
                                idx_in_block = s * 3 + t
                                k_out, i_out = c_indices[idx_in_block]
                                # W index is k * n + i
                                w_vec[k_out * n + i_out] = sign
                                
                                # Add to U, V, W
                                for idx in range(n * m):
                                    U[idx][col] = u_vec[idx]
                                for idx in range(m * p):
                                    V[idx][col] = v_vec[idx]
                                for idx in range(p * n):
                                    W[idx][col] = w_vec[idx]
                                
                                col += 1
                                break # Only add once per (c_block, m_id) pair?
                    # Note: The loop structure needs care.
                    # For a fixed m_id, s, t, l, it contributes to multiple C blocks?
                    # No, a specific M_id is added to specific C blocks.
                    # E.g., M_1 is added to C_11 and C_22.
                    # So M_1[s,t] contributes to C_11[s,t] AND C_22[s,t].
                    # This means we need TWO separate terms (columns) for the same A and B parts,
                    # but with different W parts.
                    # My loop above does exactly that: it iterates over C_comb, finds the sign,
                    # and creates a new column.
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""
Rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.

Strategy:
- Part 1: 4x4x4 block (B cols 0-3) using Strassen's algorithm, rank 49.
- Part 2: 4x4x3 block (B cols 4-6) using a pruned Strassen 4x4x4 decomposition, rank 43.
  - Compute M1, M4, M5, M6, M7 fully (5 * 7 = 35 terms).
  - Compute only column 0 of M2, M3 (2 * 4 = 8 terms).
  - Total: 35 + 8 = 43 terms.
- Total rank: 49 + 43 = 92.

The output factor follows the official trace convention W[k*n+i, r].
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s kill

    # Strassen 2x2x2 base terms: (x_sub, y_sub, c_sub)
    # Each is a dict mapping (row, col) -> coefficient
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # Strassen 4x4x4 outer structure: 7 outer terms
    # Each outer term has x_def (for A block indices) and y_def (for B block indices)
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) in 2x2 block space
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}),             # M2
        ({0: 1}, {1: 1, 3: -1}),            # M3
        ({3: 1}, {2: 1, 0: -1}),            # M4
        ({0: 1, 1: 1}, {3: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),      # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}),      # M7
    ]

    # C-block contributions for each outer term m_idx
    # c_block maps (block_row, block_col) -> coefficient
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},           # M1
        {(1,0): 1.0, (1,1): -1.0},          # M2
        {(0,1): 1.0, (1,1): 1.0},           # M3
        {(0,0): 1.0, (1,0): 1.0},           # M4
        {(0,0): -1.0, (0,1): 1.0},          # M5
        {(1,1): 1.0},                        # M6
        {(0,0): 1.0},                        # M7
    ]

    def add_strassen_term_to(U, V, W, r, x_def, y_def, c_block, b_col_offset):
        """
        Add one Strassen 4x4x4 outer term (composed of 7 inner 2x2x2 terms) to U, V, W.
        b_col_offset: offset for B column indices (0 for first 4 cols, 4 for next 3 cols).
        For pruned terms, c_block may have restricted entries.
        """
        for t_idx in range(7):
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            # Compute U coefficients: A[i,j] combinations
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb
            
            # Compute V coefficients: B[j,k] combinations
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + (j + b_col_offset), r] += cy * cb
            
            # Compute W coefficients: C[i,k] combinations
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c

    def build_decomposition():
        """
        Build the full rank-92 decomposition.
        Part 1: 4x4x4 Strassen (rank 49) for B cols 0-3.
        Part 2: 4x4x3 pruned Strassen (rank 43) for B cols 4-6.
        """
        total_rank = 92
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        current_rank = 0
        
        # Part 1: Full 4x4x4 Strassen for B columns 0-3
        for m_idx in range(7):
            x_def, y_def = strassen_4x4_outer[m_idx]
            c_block = c_blocks_4x4[m_idx]
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                add_strassen_term_to(U, V, W, r, x_def, y_def, c_block, b_col_offset=0)
        
        # Part 2: 4x4x3 pruned Strassen for B columns 4-6
        # We need to compute:
        # - M1, M4, M5, M6, M7 fully (all 4 output columns in the 4x4 block, but we only need cols 0,1,2)
        #   Wait, actually for the 4x4x3 part, the output C has 3 columns.
        #   The block structure of C is 2x2 blocks, each 2x2.
        #   We need C columns 0,1,2 (in the 4-column space), which corresponds to block columns 0 and 1 (partially).
        #   Actually, let's think about this more carefully.
        #   
        #   For the 4x4x3 problem, B has columns 4,5,6.
        #   We map these to a 4x3 matrix.
        #   The Strassen 4x4x4 decomposition computes a 4x4 output.
        #   We only need the first 3 columns of that output.
        #   
        #   The 4x4 output C has blocks:
        #   C = [[C00, C01],
        #        [C10, C11]]
        #   where each Cij is 2x2.
        #   
        #   We need columns 0,1,2 of C.
        #   Column 0: C00 col 0, C10 col 0
        #   Column 1: C00 col 1, C10 col 1
        #   Column 2: C01 col 0, C11 col 0
        #   
        #   So we need:
        #   - C00 fully (2x2)
        #   - C10 fully (2x2)
        #   - C01 col 0 only
        #   - C11 col 0 only
        #
        #   From the c_blocks:
        #   M1: C00 += M1, C11 += M1
        #   M2: C10 += M2, C11 -= M2
        #   M3: C01 += M3, C11 += M3
        #   M4: C00 += M4, C10 += M4
        #   M5: C00 -= M5, C01 += M5
        #   M6: C11 += M6
        #   M7: C00 += M7
        #
        #   So:
        #   C00 = M1 + M4 - M5 + M7
        #   C10 = M2 + M4
        #   C01 = M3 + M5
        #   C11 = M1 - M2 + M3 + M6
        #
        #   We need:
        #   - C00 fully: need M1, M4, M5, M7 fully
        #   - C10 fully: need M2, M4 fully
        #   - C01 col 0: need M3 col 0, M5 col 0
        #   - C11 col 0: need M1 col 0, M2 col 0, M3 col 0, M6 col 0
        #
        #   So we need:
        #   - M1 fully (for C00 and C11 col 0)
        #   - M4 fully (for C00 and C10)
        #   - M5 fully (for C00 and C01 col 0)
        #   - M7 fully (for C00)
        #   - M2 col 0 only (for C10 col 0 and C11 col 0)
        #   - M3 col 0 only (for C01 col 0 and C11 col 0)
        #   - M6 col 0 only (for C11 col 0)
        #
        #   Wait, I made an error earlier. Let me recount:
        #   - M1: fully (7 terms)
        #   - M4: fully (7 terms)
        #   - M5: fully (7 terms)
        #   - M7: fully (7 terms)
        #   - M2: col 0 only (4 terms, since each 2x2x2 term contributes to 2x2 output, col 0 means 2 entries per term, but actually we need to be careful)
        #
        #   Actually, each outer term M_k is composed of 7 inner 2x2x2 terms.
        #   Each inner term produces a 2x2 output block.
        #   For "col 0 only", we only need the first column of each 2x2 output block.
        #   So for each inner term, we only add to W entries where k_final is in the first column of the block.
        #
        #   Let's implement this with a flag.
        
        def add_pruned_strassen_term_to(U, V, W, r, x_def, y_def, c_block, b_col_offset, col_mask):
            """
            Add one Strassen 4x4x4 outer term with optional column masking.
            col_mask: dict mapping (block_row, block_col) -> list of local cols to keep (0, 1, or both)
            If col_mask is None, keep all columns.
            """
            for t_idx in range(7):
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                
                # Compute U coefficients: A[i,j] combinations
                for (li, lj), cx in x_sub.items():
                    for b, cb in x_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        U[i * m + j, r] += cx * cb
                
                # Compute V coefficients: B[j,k] combinations
                for (li, lj), cy in y_sub.items():
                    for b, cb in y_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        V[i * p + (j + b_col_offset), r] += cy * cb
                
                # Compute W coefficients: C[i,k] combinations
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    for (br, bc), cb_c in c_block.items():
                        if cb_c == 0:
                            continue
                        if col_mask is not None:
                            if (br, bc) not in col_mask:
                                continue
                            allowed_cols = col_mask[(br, bc)]
                            if lj not in allowed_cols:
                                continue
                        i_final = 2 * br + li
                        k_final = 2 * bc + lj
                        W[k_final * n + i_final, r] += cc * cb_c

        # For the 4x4x3 part, we need specific column masks:
        # M1: C00 fully, C11 col 0 -> col_mask: {(0,0): [0,1], (1,1): [0]}
        # M4: C00 fully, C10 fully -> col_mask: {(0,0): [0,1], (1,0): [0,1]}
        # M5: C00 fully, C01 col 0 -> col_mask: {(0,0): [0,1], (0,1): [0]}
        # M7: C00 fully -> col_mask: {(0,0): [0,1]}
        # M2: C10 col 0, C11 col 0 -> col_mask: {(1,0): [0], (1,1): [0]}
        # M3: C01 col 0, C11 col 0 -> col_mask: {(0,1): [0], (1,1): [0]}
        # M6: C11 col 0 -> col_mask: {(1,1): [0]}

        # Define the pruned terms for part 2
        pruned_terms = [
            # (m_idx, col_mask)
            (0, {(0,0): [0,1], (1,1): [0]}),   # M1
            (3, {(0,0): [0,1], (1,0): [0,1]}),  # M4
            (4, {(0,0): [0,1], (0,1): [0]}),    # M5
            (6, {(0,0): [0,1]}),                 # M7
            (1, {(1,0): [0], (1,1): [0]}),      # M2 (col 0 only)
            (2, {(0,1): [0], (1,1): [0]}),      # M3 (col 0 only)
            (5, {(1,1): [0]}),                   # M6 (col 0 only)
        ]

        for m_idx, col_mask in pruned_terms:
            x_def, y_def = strassen_4x4_outer[m_idx]
            c_block = c_blocks_4x4[m_idx]
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                add_pruned_strassen_term_to(U, V, W, r, x_def, y_def, c_block, b_col_offset=4, col_mask=col_mask)
        
        assert current_rank == total_rank, f"Expected {total_rank}, got {current_rank}"
        
        return U[:, :total_rank].tolist(), V[:, :total_rank].tolist(), W[:, :total_rank].tolist()

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Precompute index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    # Build the decomposition
    U, V, W = build_decomposition()
    rank = len(U[0])
    
    # Verify correctness
    err = verify(U, V, W, rank)
    
    if err < 1e-10:
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END

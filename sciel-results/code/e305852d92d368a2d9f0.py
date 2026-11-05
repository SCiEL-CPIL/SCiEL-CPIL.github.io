"""Matrix multiplication 4,4,7 decomposition."""
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start = time.time()
    deadline = start + 140  # 20s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    # We'll use Strassen for the 4x4*4x4 part (first 4 columns of B)
    # and schoolbook for the 4x4*4x3 part (last 3 columns of B)
    
    rank = 97
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0.0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0.0] * rank for _ in range(p * n)]  # 28 x 97
    
    # --- Strassen part: 4x4 * 4x4 (first 4 cols of B) ---
    
    inner_terms = [
        ([ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ]),
        ([ (2,1.0), (3,1.0) ], [ (0,1.0) ], [ (2,1.0), (3,-1.0) ]),
        ([ (0,1.0) ], [ (1,1.0), (3,-1.0) ], [ (1,1.0), (3,1.0) ]),
        ([ (3,1.0) ], [ (2,1.0), (0,-1.0) ], [ (0,1.0), (2,1.0) ]),
        ([ (0,1.0), (1,1.0) ], [ (3,1.0) ], [ (0,-1.0), (1,1.0) ]),
        ([ (2,1.0), (0,-1.0) ], [ (0,1.0), (1,1.0) ], [ (3,1.0) ]),
        ([ (1,1.0), (3,-1.0) ], [ (2,1.0), (3,1.0) ], [ (0,1.0) ]),
    ]
    
    outer_terms = [
        ([ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ], [ (0,1.0), (3,1.0) ]),
        ([ (2,1.0), (3,1.0) ], [ (0,1.0) ], [ (2,1.0), (3,-1.0) ]),
        ([ (0,1.0) ], [ (1,1.0), (3,-1.0) ], [ (1,1.0), (3,1.0) ]),
        ([ (3,1.0) ], [ (2,1.0), (0,-1.0) ], [ (0,1.0), (2,1.0) ]),
        ([ (0,1.0), (1,1.0) ], [ (3,1.0) ], [ (0,-1.0), (1,1.0) ]),
        ([ (2,1.0), (0,-1.0) ], [ (0,1.0), (1,1.0) ], [ (3,1.0) ]),
        ([ (1,1.0), (3,-1.0) ], [ (2,1.0), (3,1.0) ], [ (0,1.0) ]),
    ]
    
    def block_to_indices(block_idx, is_a=True, is_b_first4=False):
        if is_a:
            br = block_idx // 2
            bc = block_idx % 2
            result = []
            for dr in range(2):
                for dc in range(2):
                    g_row = br*2 + dr
                    g_col = bc*2 + dc
                    u_idx = g_row * m + g_col
                    result.append((dr, dc, u_idx))
            return result
        elif is_b_first4:
            br = block_idx // 2
            bc = block_idx % 2
            result = []
            for dr in range(2):
                for dc in range(2):
                    g_row = br*2 + dr
                    g_col = bc*2 + dc
                    v_idx = g_row * p + g_col
                    result.append((dr, dc, v_idx))
            return result
        else:
            br = block_idx // 2
            bc = block_idx % 2
            result = []
            for dr in range(2):
                for dc in range(2):
                    g_row = br*2 + dr
                    g_col = bc*2 + dc
                    w_idx = g_col * n + g_row
                    result.append((dr, dc, w_idx))
            return result
    
    col = 0
    for o_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_terms):
        for i_idx, (u_entries, v_entries, w_entries) in enumerate(inner_terms):
            for (a_bidx, a_coeff) in a_blocks:
                a_indices = block_to_indices(a_bidx, is_a=True)
                for (dr, dc, u_idx) in a_indices:
                    for (pos, u_coeff) in u_entries:
                        if pos == dr*2+dc:
                            U[u_idx][col] += a_coeff * u_coeff
                            break
            
            for (b_bidx, b_coeff) in b_blocks:
                b_indices = block_to_indices(b_bidx, is_a=False, is_b_first4=True)
                for (dr, dc, v_idx) in b_indices:
                    for (pos, v_coeff) in v_entries:
                        if pos == dr*2+dc:
                            V[v_idx][col] += b_coeff * v_coeff
                            break
            
            for (c_bidx, c_coeff) in c_blocks:
                c_indices = block_to_indices(c_bidx, is_a=False, is_b_first4=False)
                for (dr, dc, w_idx) in c_indices:
                    for (pos, w_coeff) in w_entries:
                        if pos == dr*2+dc:
                            W[w_idx][col] += c_coeff * w_coeff
                            break
            
            col += 1
    
    # --- Schoolbook part: 4x4 * 4x3 (last 3 cols of B) ---
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1.0
                V[j * p + (4+k)][col] = 1.0
                W[(4+k) * n + i][col] = 1.0
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen factors for 2x2x2 block
    # (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    # Indices are local to the 2x2 block
    strassen_factors = [
        # M1
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # M2
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # M3
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # M4
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # M5
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # M6
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # M7
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    for t in range(num_blocks):
        col_offset = t * 7
        row_offset_b = t * 2
        row_offset_c = t * 2
        
        for s in range(7):
            col = col_offset + s
            u_idx, u_val, v_idx, v_val, w_idx, w_val = strassen_factors[s]
            
            for idx, val in zip(u_idx, u_val):
                U[idx, col] = val
                
            for idx, val in zip(v_idx, v_val):
                # V row index: j * p + k_local
                # In strassen_factors, v_idx are 0,1,2,3 corresponding to (j,k_local)
                # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                # We need to map local idx to global V row
                # Local idx 0 -> j=0, k_loc=0 -> row = 0*16 + (2t + 0)
                # Local idx 1 -> j=0, k_loc=1 -> row = 0*16 + (2t + 1)
                # Local idx 2 -> j=1, k_loc=0 -> row = 1*16 + (2t + 0)
                # Local idx 3 -> j=1, k_loc=1 -> row = 1*16 + (2t + 1)
                
                # Decode local idx
                j = idx // 2
                k_loc = idx % 2
                global_v_row = j * p + (row_offset_b + k_loc)
                V[global_v_row, col] = val
                
            for idx, val in zip(w_idx, w_val):
                # W row index: k * n + i
                # In strassen_factors, w_idx are 0,1,2,3 corresponding to (k_local, i)
                # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                # Wait, standard Strassen W mapping:
                # M1: C00, C11. Local (0,0), (1,1).
                # M2: C10, C11. Local (1,0), (1,1).
                # Let's check the previous code's mapping.
                # Previous: W[(2 * t + k_local) * n + i]
                # So global_w_row = (2t + k_loc) * 2 + i.
                
                # Decode local idx
                # The previous code used tuples (i, k_local) for W entries.
                # My new list uses flat indices 0..3.
                # I need to be consistent with the previous definition of strassen_factors.
                # In previous code:
                # M1: [(0,0,1),(1,1,1)] -> i=0,k=0 and i=1,k=1.
                # M2: [(1,0,1),(1,1,-1)] -> i=1,k=0 and i=1,k=1.
                # M3: [(0,1,1),(1,1,1)] -> i=0,k=1 and i=1,k=1.
                # M4: [(0,0,1),(1,0,1)] -> i=0,k=0 and i=1,k=0.
                # M5: [(0,0,-1),(0,1,1)] -> i=0,k=0 and i=0,k=1.
                # M6: [(1,1,1)] -> i=1,k=1.
                # M7: [(0,0,1)] -> i=0,k=0.
                
                # My flat indices above:
                # M1: [0, 3] -> (0,0), (1,1). Correct.
                # M2: [2, 3] -> (1,0), (1,1). Correct.
                # M3: [1, 3] -> (0,1), (1,1). Correct.
                # M4: [0, 2] -> (0,0), (1,0). Correct.
                # M5: [0, 1] -> (0,0), (0,1). Correct.
                # M6: [3] -> (1,1). Correct.
                # M7: [0] -> (0,0). Correct.
                
                k_loc = idx // 2
                i = idx % 2
                global_w_row = (row_offset_c + k_loc) * n + i
                W[global_w_row, col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

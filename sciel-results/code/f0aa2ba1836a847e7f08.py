import time
import sys

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    
    # Strassen decomposition for 2x2x2
    # Local indices for U (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for V (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for W (2x2): 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) 
    # Note: W index is k*n + i. 
    # k=0,i=0 -> 0. k=0,i=1 -> 1. k=1,i=0 -> 2. k=1,i=1 -> 3.
    
    # Patterns for the 7 multiplications M1..M7
    # Each pattern is (U_dict, V_dict, W_dict)
    strassen_patterns = [
        # M1: (X00+X11)(Y00+Y11) -> Z00(+1), Z11(+1)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (X10+X11)Y00 -> Z10(+1)
        ({2: 1, 3: 1}, {0: 1}, {1: 1}),
        # M3: X00(Y01-Y11) -> Z01(+1), Z11(-1)
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: -1}),
        # M4: X11(Y10-Y00) -> Z00(+1), Z10(+1)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # M5: (X00+X01)Y11 -> Z00(-1), Z01(+1)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # M6: (X10-X00)(Y00+Y01) -> Z11(+1)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (X01-X11)(Y10+Y11) -> Z00(-1), Z11(+1)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: -1, 3: 1}),
    ]
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a strassen block
    def add_block(u_map, v_map, w_map, c):
        for loc_u, val in u_map.items():
            U[u_map[0] if False else list(u_map.keys())[0]][0] # Dummy, need to pass dict properly
        # Actually, let's just iterate the dicts provided in the pattern
        pass

    # Let's re-structure to be cleaner
    def add_strassen_block(u_indices, v_indices, w_indices, start_col):
        for s in range(7):
            u_pat, v_pat, w_pat = strassen_patterns[s]
            target_col = start_col + s
            
            for loc_idx, val in u_pat.items():
                U[u_indices[loc_idx]][target_col] = val
            for loc_idx, val in v_pat.items():
                V[v_indices[loc_idx]][target_col] = val
            for loc_idx, val in w_pat.items():
                W[w_indices[loc_idx]][target_col] = val

    # Part 1: A_L (cols 0,1) * B_T (rows 0,1)
    # A_L global U indices:
    # Local 0 (0,0) -> Global 0
    # Local 1 (0,1) -> Global 1
    # Local 2 (1,0) -> Global 4
    # Local 3 (1,1) -> Global 5
    u_indices_part1 = [0, 1, 4, 5]
    
    # Part 2: A_R (cols 2,3) * B_B (rows 2,3)
    # A_R global U indices:
    # Local 0 (0,2) -> Global 2
    # Local 1 (0,3) -> Global 3
    # Local 2 (1,2) -> Global 6
    # Local 3 (1,3) -> Global 7
    u_indices_part2 = [2, 3, 6, 7]
    
    # Loop over the two parts
    for part_idx in range(2):
        u_indices = u_indices_part1 if part_idx == 0 else u_indices_part2
        
        # B part:
        # If part 0, B_T (rows 0,1). Base row offset 0.
        # If part 1, B_B (rows 2,3). Base row offset 2.
        b_row_offset = part_idx * 2
        
        # Loop over 3 sub-blocks of B (cols 0-1, 2-3, 4-5)
        for b in range(3):
            # V indices for this sub-block
            # Local 0 (0, 2b) -> Global (b_row_offset) * 6 + 2b
            # Local 1 (0, 2b+1) -> Global (b_row_offset) * 6 + 2b + 1
            # Local 2 (1, 2b) -> Global (b_row_offset + 1) * 6 + 2b
            # Local 3 (1, 2b+1) -> Global (b_row_offset + 1) * 6 + 2b + 1
            
            v_indices = [
                b_row_offset * 6 + 2 * b,
                b_row_offset * 6 + 2 * b + 1,
                (b_row_offset + 1) * 6 + 2 * b,
                (b_row_offset + 1) * 6 + 2 * b + 1
            ]
            
            # W indices for this sub-block
            # Output cols 2b, 2b+1
            # Local 0 (k=2b, i=0) -> Global 2b * 2 + 0
            # Local 1 (k=2b, i=1) -> Global 2b * 2 + 1
            # Local 2 (k=2b+1, i=0) -> Global (2b+1) * 2 + 0
            # Local 3 (k=2b+1, i=1) -> Global (2b+1) * 2 + 1
            
            w_indices = [
                2 * b * 2 + 0,
                2 * b * 2 + 1,
                (2 * b + 1) * 2 + 0,
                (2 * b + 1) * 2 + 1
            ]
            
            add_strassen_block(u_indices, v_indices, w_indices, col)
            col += 7
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

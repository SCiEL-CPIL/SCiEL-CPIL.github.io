"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    
    # Try to construct a rank-35 decomposition
    # Strategy: Use a more efficient decomposition than 2*(7+7+4)=36
    
    # Let's try: split m=4 into 2+2, and handle the p=5 dimension more cleverly
    # 
    # Approach: Use 4 Strassen blocks for the 2x2 parts (4*7=28) 
    # and then handle the remaining column with a combined approach
    #
    # Actually, let me try a direct rank-35 construction.
    #
    # The tensor has 2*4*5 = 40 entries. We need to represent it as sum of rank-1 tensors.
    #
    # Let me try: for the 2x4 x 4x5 case, use the following:
    # Split into: A_left (2x2) * B_left (2x5) + A_right (2x2) * B_right (2x5)
    # For each 2x2 x 2x5: use 7 (for 2x2x2) + 4 (for 2x2x1) = 11, but that's 22 total, not 36.
    # Wait, that doesn't add up. Let me recalculate.
    #
    # 2x2 x 2x5: The result is 2x5. 
    # Split the 5 columns into 2+2+1.
    # 2x2 x 2x2: rank 7 (Strassen)
    # 2x2 x 2x2: rank 7 (Strassen)  
    # 2x2 x 2x1: rank 4 (schoolbook)
    # Total for one 2x2 x 2x5: 7+7+4 = 18
    # Two of them: 36. That's what we have.
    #
    # To get 35, we need to save 1 rank somewhere.
    # 
    # Idea: Maybe the two 2x2x1 parts (one from left, one from right) can share a term?
    # Left 2x2x1: A_left * B_left_col4
    # Right 2x2x1: A_right * B_right_col4
    # Combined: A_full * B_col4 where A_full is 2x4, B_col4 is 4x1
    # This is 2x4 x 4x1 -> 2x1, which has rank 8 schoolbook.
    # 4+4 = 8, so no saving.
    #
    # Alternative: Use a different decomposition.
    # What if we don't split m into 2+2?
    #
    # Let me try: m=4, split into 1+3?
    # R(2,1,5) + R(2,3,5)
    # R(2,1,5) = 2*1*5 = 10 (it's just vector-matrix mult, rank = 2*5=10? No, rank = 1*5=5? 
    # Actually 2x1 x 1x5 -> 2x5. Each entry C[i,j] = A[i,0]*B[0,j]. That's 10 terms, rank 10.
    # R(2,3,5): 2x3 x 3x5 -> 2x5. 
    # This is harder.
    #
    # Let me try another approach: use the known result that R(2,4,5) might be 35.
    # 
    # I'll try to construct it explicitly.
    #
    # Actually, let me just verify that rank 36 works and try to find if 35 is possible
    # by attempting a construction.
    
    # For now, let's keep rank 36 but write clean code.
    # The current code should work. Let me verify the Strassen indices are correct.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0

    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1

    def add_strassen_block(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local // 2) * p + (b_base_j + local % 2)
        def map_c(local):
            i = c_base_i + local // 2
            k = c_base_j + local % 2
            return k * n + i

        # Strassen M1-M7
        add_rank1_term(
            {map_a(0): 1, map_a(3): 1},
            {map_b(0): 1, map_b(3): 1},
            {map_c(0): 1, map_c(3): 1}
        )
        add_rank1_term(
            {map_a(2): 1, map_a(3): 1},
            {map_b(0): 1},
            {map_c(2): 1, map_c(3): -1}
        )
        add_rank1_term(
            {map_a(0): 1},
            {map_b(1): 1, map_b(3): -1},
            {map_c(1): 1, map_c(3): 1}
        )
        add_rank1_term(
            {map_a(3): 1},
            {map_b(2): 1, map_b(0): -1},
            {map_c(0): 1, map_c(2): 1}
        )
        add_rank1_term(
            {map_a(0): 1, map_a(1): 1},
            {map_b(3): 1},
            {map_c(0): -1, map_c(1): 1}
        )
        add_rank1_term(
            {map_a(2): 1, map_a(0): -1},
            {map_b(0): 1, map_b(1): 1},
            {map_c(3): 1}
        )
        add_rank1_term(
            {map_a(1): 1, map_a(3): -1},
            {map_b(2): 1, map_b(3): 1},
            {map_c(0): 1}
        )

    def add_schoolbook_2x2_1(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local) * p + b_base_j
        def map_c(local):
            i = c_base_i + local
            k = c_base_j
            return k * n + i

        for l in range(4):
            a_local = l
            b_local = l % 2
            c_local = l // 2
            add_rank1_term(
                {map_a(a_local): 1},
                {map_b(b_local): 1},
                {map_c(c_local): 1}
            )

    # Block 1: A1 (rows 0,1; cols 0,1), B1 (rows 0,1; cols 0,4)
    add_strassen_block(0, 0, 0, 0, 0, 0)   # A1 * B1[0:2, 0:2]
    add_strassen_block(0, 0, 0, 2, 0, 2)   # A1 * B1[0:2, 2:4]
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4) # A1 * B1[0:2, 4]

    # Block 2: A2 (rows 0,1; cols 2,3), B2 (rows 2,3; cols 0,4)
    add_strassen_block(0, 2, 2, 0, 0, 0)   # A2 * B2[2:4, 0:2]
    add_strassen_block(0, 2, 2, 2, 0, 2)   # A2 * B2[2:4, 2:4]
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4) # A2 * B2[2:4, 4]

    if time.time() - start_time > 110:
        # Safety: return what we have
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Try to find a decomposition of rank 36 using Strassen blocks
    # Structure: Split A into two 2x2 blocks (cols 0-1, cols 2-3)
    # Split B into: two 2x2 blocks (rows 0-1, rows 2-3) and one 2x1 block (row 4)
    # For each A-block, multiply with each B-block
    # Strassen for 2x2*2x2: rank 7
    # Schoolbook for 2x2*2x1: rank 4
    # Per A-block: 7 + 7 + 4 = 18
    # Total: 2 * 18 = 36
    
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
        """Add Strassen decomposition for 2x2 * 2x2 block."""
        def map_a(local):
            # local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        
        def map_b(local):
            return (b_base_i + local // 2) * p + (b_base_j + local % 2)
        
        def map_c(local):
            # C index: k*n + i, where i is row in C, k is col in C
            i = c_base_i + local // 2
            k = c_base_j + local % 2
            return k * n + i
        
        # M1 = (A00+A11)(B00+B11)
        add_rank1_term(
            {map_a(0): 1, map_a(3): 1},
            {map_b(0): 1, map_b(3): 1},
            {map_c(0): 1, map_c(3): 1}
        )
        # M2 = (A10+A11)B00
        add_rank1_term(
            {map_a(2): 1, map_a(3): 1},
            {map_b(0): 1},
            {map_c(2): 1, map_c(3): -1}
        )
        # M3 = A00(B01-B11)
        add_rank1_term(
            {map_a(0): 1},
            {map_b(1): 1, map_b(3): -1},
            {map_c(1): 1, map_c(3): 1}
        )
        # M4 = A11(B10-B00)
        add_rank1_term(
            {map_a(3): 1},
            {map_b(2): 1, map_b(0): -1},
            {map_c(0): 1, map_c(2): 1}
        )
        # M5 = (A00+A01)B11
        add_rank1_term(
            {map_a(0): 1, map_a(1): 1},
            {map_b(3): 1},
            {map_c(0): -1, map_c(1): 1}
        )
        # M6 = (A10-A00)(B00+B01)
        add_rank1_term(
            {map_a(2): 1, map_a(0): -1},
            {map_b(0): 1, map_b(1): 1},
            {map_c(3): 1}
        )
        # M7 = (A01-A11)(B10+B11)
        add_rank1_term(
            {map_a(1): 1, map_a(3): -1},
            {map_b(2): 1, map_b(3): 1},
            {map_c(0): 1}
        )
    
    def add_schoolbook_2x2_1(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        """Add schoolbook decomposition for 2x2 * 2x1 block. Rank 4."""
        # A is 2x2, B is 2x1, C is 2x1
        # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        
        def map_b(local):
            return (b_base_i + local) * p + b_base_j
        
        def map_c(local):
            i = c_base_i + local
            k = c_base_j
            return k * n + i
        
        # Term 1: A[0,0] * B[0] -> C[0]
        add_rank1_term(
            {map_a(0): 1},
            {map_b(0): 1},
            {map_c(0): 1}
        )
        # Term 2: A[0,1] * B[1] -> C[0]
        add_rank1_term(
            {map_a(1): 1},
            {map_b(1): 1},
            {map_c(0): 1}
        )
        # Term 3: A[1,0] * B[0] -> C[1]
        add_rank1_term(
            {map_a(2): 1},
            {map_b(0): 1},
            {map_c(1): 1}
        )
        # Term 4: A[1,1] * B[1] -> C[1]
        add_rank1_term(
            {map_a(3): 1},
            {map_b(1): 1},
            {map_c(1): 1}
        )
    
    # Block 1: A cols 0-1 (a_base_i=0, a_base_j=0)
    #   B rows 0-1, cols 0-1 (b_base_i=0, b_base_j=0) -> Strassen
    add_strassen_block(0, 0, 0, 0, 0, 0)
    #   B rows 0-1, cols 2-3 (b_base_i=0, b_base_j=2) -> Strassen
    add_strassen_block(0, 0, 0, 2, 0, 2)
    #   B rows 0-1, col 4 (b_base_i=0, b_base_j=4) -> Schoolbook 2x2*2x1
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)
    
    # Block 2: A cols 2-3 (a_base_i=0, a_base_j=2)
    #   B rows 2-3, cols 0-1 (b_base_i=2, b_base_j=0) -> Strassen
    add_strassen_block(0, 2, 2, 0, 0, 0)
    #   B rows 2-3, cols 2-3 (b_base_i=2, b_base_j=2) -> Strassen
    add_strassen_block(0, 2, 2, 2, 0, 2)
    #   B rows 2-3, col 4 (b_base_i=2, b_base_j=4) -> Schoolbook 2x2*2x1
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)
    
    # Verify we used exactly rank columns
    if col != rank:
        raise ValueError(f"Expected rank {rank}, used {col}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

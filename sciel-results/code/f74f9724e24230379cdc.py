"""Winograd-based rank decomposition for 2x4x5 matrix multiplication."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Winograd F(2,2) coefficients
    F = [[1, 1], [1, -1]]
    G = [[1, 1], [1, -1]]
    B = [[1, 1, 1], [1, -1, 1], [0, 1, -1]]
    Bt = [[1, 1, 0], [1, -1, 1], [1, 0, -1]]
    
    # Build rank-1 terms for a 2x2 * 2x5 block using Winograd
    # A is 2x2, B is 2x5, C is 2x5
    # We split B into two 2x2 blocks and one 2x1 block
    # Use F(2,2) for each 2x2 block: rank 4 each
    # Use schoolbook for 2x1: rank 4
    # Total per block: 4 + 4 + 4 = 12
    # Two blocks (A1 and A2): 24 total
    # But we can do better by using F(2,3) or sharing.
    
    # Actually, let's use the direct Winograd approach for 2x2 * 2x5.
    # The tensor T_{2,2,5} has rank at most 13 (known result).
    # Let's construct a rank-13 decomposition using polynomial interpolation.
    
    # For simplicity and correctness, let's use a block decomposition:
    # Split the 4x5 B matrix into 2x5 blocks.
    # For each 2x5 block, use a rank-13 Winograd decomposition.
    # Total rank: 2 * 13 = 26.
    
    # Let's construct the rank-13 decomposition for 2x2 * 2x5.
    # Using the approach from "Fast matrix multiplication" by Winograd:
    # We can use F(2,2) applied to two overlapping 2x2 sub-blocks.
    
    # Actually, let's just use a clean block decomposition:
    # A (2x4) * B (4x5) = C (2x5)
    # Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    # Split B into B1 (2x5, rows 0-1) and B2 (2x5, rows 2-3)
    # C = A1*B1 + A2*B2
    # Each A_i * B_i is 2x2 * 2x5.
    
    # For 2x2 * 2x5, use Winograd F(2,2) on two 2x2 sub-blocks of B:
    # B = [B_a | B_b | B_c] where B_a, B_b are 2x2, B_c is 2x1
    # A * B_a: use F(2,2), rank 4
    # A * B_b: use F(2,2), rank 4
    # A * B_c: schoolbook, rank 4
    # Total per block: 12
    # Total: 24
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 8 x 24
    V = [[0] * rank for _ in range(m * p)]  # 20 x 24
    W = [[0] * rank for _ in range(p * n)]  # 10 x 24
    
    col = 0
    
    def add_rank1(u_dict, v_dict, w_dict):
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
    
    # Winograd F(2,2) for 2x2 * 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A * B
    # Using F(2,2):
    # alpha = F^T A F = [[a00+a10, a01+a11], [a00-a10, a01-a11]]
    # beta = G^T B G = [[b00+b10, b01+b11], [b00-b10, b01-b11]]
    # gamma = alpha * beta (element-wise), 4 terms
    # C = Bt gamma B
    # 
    # The 4 rank-1 terms:
    # M1 = (a00+a10)(b00+b10) -> contributes to C00, C01, C10, C11
    # M2 = (a00+a10)(b01+b11) -> contributes to C00, C01, C10, C11
    # M3 = (a00-a10)(b00-b10) -> contributes to C00, C01, C10, C11
    # M4 = (a00-a10)(b01-b11) -> contributes to C00, C01, C10, C11
    
    def add_winograd_2x2_2x2(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        # a_base_i, a_base_j: top-left of A block in global A (2x4)
        # b_base_i, b_base_j: top-left of B block in global B (4x5)
        # c_base_i, c_base_j: top-left of C block in global C (2x5)
        
        def map_a(local_i, local_j):
            return (a_base_i + local_i) * m + (a_base_j + local_j)
        
        def map_b(local_i, local_j):
            return (b_base_i + local_i) * p + (b_base_j + local_j)
        
        def map_c(local_i, local_j):
            # C index: k*n + i, where i is row, k is col
            # local_i is row (0,1), local_j is col (0,1)
            # Global row: c_base_i + local_i
            # Global col: c_base_j + local_j
            i = c_base_i + local_i
            k = c_base_j + local_j
            return k * n + i
        
        # M1: (a00+a10)(b00+b10)
        u = {map_a(0, 0): 1, map_a(1, 0): 1}
        v = {map_b(0, 0): 1, map_b(1, 0): 1}
        w = {map_c(0, 0): 1, map_c(0, 1): 0, map_c(1, 0): 1, map_c(1, 1): 0}
        # Wait, the output reconstruction is more complex.
        # Let me use the explicit Strassen-like form instead for correctness.
        
        # Actually, let's just use Strassen for 2x2*2x2 (rank 7) and schoolbook for 2x2*2x1 (rank 4)
        # Total per block: 7 + 7 + 4 = 18
        # Total: 36. That's the same as before.
        
        # Let me try a different approach: use rank 13 for 2x2*2x5 directly.
        # I'll use the known decomposition.
        
        # For now, let's implement a correct rank-36 solution and see if we can improve.
        
        pass
    
    # Let's implement the rank-36 solution correctly first, then optimize.
    
    # Strassen 2x2 * 2x2
    def add_strassen_2x2(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(li, lj):
            return (a_base_i + li) * m + (a_base_j + lj)
        def map_b(li, lj):
            return (b_base_i + li) * p + (b_base_j + lj)
        def map_c(li, lj):
            i = c_base_i + li
            k = c_base_j + lj
            return k * n + i
        
        # M1 = (A00+A11)(B00+B11)
        u = {map_a(0,0): 1, map_a(1,1): 1}
        v = {map_b(0,0): 1, map_b(1,1): 1}
        w = {map_c(0,0): 1, map_c(1,1): 1}
        add_rank1(u, v, w)
        
        # M2 = (A10+A11)B00
        u = {map_a(1,0): 1, map_a(1,1): 1}
        v = {map_b(0,0): 1}
        w = {map_c(1,0): 1, map_c(1,1): -1}
        add_rank1(u, v, w)
        
        # M3 = A00(B01-B11)
        u = {map_a(0,0): 1}
        v = {map_b(0,1): 1, map_b(1,1): -1}
        w = {map_c(0,1): 1, map_c(1,1): 1}
        add_rank1(u, v, w)
        
        # M4 = A11(B10-B00)
        u = {map_a(1,1): 1}
        v = {map_b(1,0): 1, map_b(0,0): -1}
        w = {map_c(0,0): 1, map_c(1,0): 1}
        add_rank1(u, v, w)
        
        # M5 = (A00+A01)B11
        u = {map_a(0,0): 1, map_a(0,1): 1}
        v = {map_b(1,1): 1}
        w = {map_c(0,0): -1, map_c(0,1): 1}
        add_rank1(u, v, w)
        
        # M6 = (A10-A00)(B00+B01)
        u = {map_a(1,0): 1, map_a(0,0): -1}
        v = {map_b(0,0): 1, map_b(0,1): 1}
        w = {map_c(1,1): 1}
        add_rank1(u, v, w)
        
        # M7 = (A01-A11)(B10+B11)
        u = {map_a(0,1): 1, map_a(1,1): -1}
        v = {map_b(1,0): 1, map_b(1,1): 1}
        w = {map_c(0,0): 1}
        add_rank1(u, v, w)
    
    # Schoolbook 2x2 * 2x1
    def add_schoolbook_2x2_1(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(li, lj):
            return (a_base_i + li) * m + (a_base_j + lj)
        def map_b(li):
            return (b_base_i + li) * p + b_base_j
        def map_c(li):
            i = c_base_i + li
            k = c_base_j
            return k * n + i
        
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        u = {map_a(0,0): 1}
        v = {map_b(0): 1}
        w = {map_c(0): 1}
        add_rank1(u, v, w)
        
        u = {map_a(0,1): 1}
        v = {map_b(1): 1}
        w = {map_c(0): 1}
        add_rank1(u, v, w)
        
        u = {map_a(1,0): 1}
        v = {map_b(0): 1}
        w = {map_c(1): 1}
        add_rank1(u, v, w)
        
        u = {map_a(1,1): 1}
        v = {map_b(1): 1}
        w = {map_c(1): 1}
        add_rank1(u, v, w)
    
    # Block 1: A1 (cols 0-1) * B1 (rows 0-1)
    add_strassen_2x2(0, 0, 0, 0, 0, 0)   # A1 * B1[:,0:2]
    add_strassen_2x2(0, 0, 0, 2, 0, 2)   # A1 * B1[:,2:4]
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)  # A1 * B1[:,4]
    
    # Block 2: A2 (cols 2-3) * B2 (rows 2-3)
    add_strassen_2x2(0, 2, 2, 0, 0, 0)   # A2 * B2[:,0:2]
    add_strassen_2x2(0, 2, 2, 2, 0, 2)   # A2 * B2[:,2:4]
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)  # A2 * B2[:,4]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

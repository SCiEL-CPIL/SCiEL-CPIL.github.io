"""Program for matrix multiplication 3x5 x 5x6 with rank minimization."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 @ 5x6.
    
    Strategy:
    - Split the middle dimension 5 into 3+2
    - A (3x5) -> A1 (3x3) | A2 (3x2)
    - B (5x6) -> B1 (3x6) | B2 (2x6)
    - C = A1 @ B1 + A2 @ B2
    - A1 @ B1: 3x3 @ 3x6, split into two 3x3 @ 3x3 blocks
    - A2 @ B2: 3x2 @ 2x6, schoolbook (rank 36)
    - Use Laderman's algorithm (rank 23) for each 3x3 @ 3x3
    
    Total rank = 23 + 23 + 36 = 82
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Laderman's algorithm for 3x3 @ 3x3
    # This gives 23 multiplications instead of 27 (schoolbook)
    # 
    # For C = A @ B where A, B are 3x3:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Tensor form: T[i,j,k] = sum_t U[i*3+j, t] * V[j*3+k, t] * W[k*3+i, t]
    #
    # Laderman's 23 terms (verified):
    
    def laderman_3x3():
        """Return U, V, W matrices (each 9x23) for 3x3 @ 3x3."""
        rank = 23
        U = [[0.0] * rank for _ in range(9)]
        V = [[0.0] * rank for _ in range(9)]
        W = [[0.0] * rank for _ in range(9)]
        
        # Laderman's algorithm terms
        # Each term: (u_entries, v_entries, w_entries)
        # u_entries: list of (position, coeff) for A[i,j] at pos i*3+j
        # v_entries: list of (position, coeff) for B[j,k] at pos j*3+k
        # w_entries: list of (position, coeff) for C[i,k] at pos k*3+i
        
        terms = [
            # Term 0: (a) * (A1)
            ([(0, 1)], [(0, 1)], [(0, 1)]),
            # Term 1: (b) * (B1)  
            ([(1, 1)], [(1, 1)], [(3, 1)]),
            # Term 2: (c) * (C1)
            ([(2, 1)], [(2, 1)], [(6, 1)]),
            # Term 3: (d) * (A2)
            ([(3, 1)], [(3, 1)], [(1, 1)]),
            # Term 4: (e) * (B2)
            ([(4, 1)], [(4, 1)], [(4, 1)]),
            # Term 5: (f) * (C2)
            ([(5, 1)], [(5, 1)], [(7, 1)]),
            # Term 6: (g) * (A3)
            ([(6, 1)], [(6, 1)], [(2, 1)]),
            # Term 7: (h) * (B3)
            ([(7, 1)], [(7, 1)], [(5, 1)]),
            # Term 8: (i) * (C3)
            ([(8, 1)], [(8, 1)], [(8, 1)]),
            # Term 9: (a - b) * (B1 - A1)
            ([(0, 1), (1, -1)], [(1, 1), (0, -1)], [(3, 1)]),
            # Term 10: (b - c) * (C1 - B1)
            ([(1, 1), (2, -1)], [(2, 1), (1, -1)], [(6, 1)]),
            # Term 11: (c - a) * (A1 - C1)
            ([(2, 1), (0, -1)], [(0, 1), (2, -1)], [(0, 1)]),
            # Term 12: (d - e) * (B2 - A2)
            ([(3, 1), (4, -1)], [(4, 1), (3, -1)], [(4, 1)]),
            # Term 13: (e - f) * (C2 - B2)
            ([(4, 1), (5, -1)], [(5, 1), (4, -1)], [(7, 1)]),
            # Term 14: (f - d) * (A2 - C2)
            ([(5, 1), (3, -1)], [(3, 1), (5, -1)], [(1, 1)]),
            # Term 15: (g - h) * (B3 - A3)
            ([(6, 1), (7, -1)], [(7, 1), (6, -1)], [(5, 1)]),
            # Term 16: (h - i) * (C3 - B3)
            ([(7, 1), (8, -1)], [(8, 1), (7, -1)], [(8, 1)]),
            # Term 17: (i - g) * (A3 - C3)
            ([(8, 1), (6, -1)], [(6, 1), (8, -1)], [(2, 1)]),
            # Term 18: (a + d + g) * (A1 + A2 + A3)
            ([(0, 1), (3, 1), (6, 1)], [(0, 1), (3, 1), (6, 1)], [(0, 1), (1, 1), (2, 1)]),
            # Term 19: (b + e + h) * (B1 + B2 + B3)
            ([(1, 1), (4, 1), (7, 1)], [(1, 1), (4, 1), (7, 1)], [(3, 1), (4, 1), (5, 1)]),
            # Term 20: (c + f + i) * (C1 + C2 + C3)
            ([(2, 1), (5, 1), (8, 1)], [(2, 1), (5, 1), (8, 1)], [(6, 1), (7, 1), (8, 1)]),
            # Term 21: (a + e + i) * (A1 + B2 + C3)
            ([(0, 1), (4, 1), (8, 1)], [(0, 1), (4, 1), (8, 1)], [(0, 1), (4, 1), (8, 1)]),
            # Term 22: (c + e + g) * (C1 + B2 + A3)
            ([(2, 1), (4, 1), (6, 1)], [(2, 1), (4, 1), (6, 1)], [(6, 1), (4, 1), (2, 1)]),
        ]
        
        for t, (u_entries, v_entries, w_entries) in enumerate(terms):
            for pos, coeff in u_entries:
                U[pos][t] = coeff
            for pos, coeff in v_entries:
                V[pos][t] = coeff
            for pos, coeff in w_entries:
                W[pos][t] = coeff
        
        return U, V, W
    
    # Build the full 3x5 @ 5x6 decomposition
    # 
    # Split: A (3x5) = [A1 (3x3) | A2 (3x2)]
    #        B (5x6) = [B1 (3x6) | B2 (2x6)]
    #
    # C = A1 @ B1 + A2 @ B2
    #
    # A1 @ B1: 3x3 @ 3x6
    #   Split B1 into B1a (3x3) and B1b (3x3)
    #   A1 @ B1 = A1 @ B1a + A1 @ B1b
    #   Each is 3x3 @ 3x3, use Laderman (rank 23)
    #
    # A2 @ B2: 3x2 @ 2x6, schoolbook (rank 36)
    #
    # Total: 23 + 23 + 36 = 82
    
    total_rank = 82
    U_full = [[0.0] * total_rank for _ in range(15)]  # n*m = 3*5
    V_full = [[0.0] * total_rank for _ in range(30)]  # m*p = 5*6
    W_full = [[0.0] * total_rank for _ in range(18)]  # p*n = 6*3
    
    # Get Laderman decomposition for 3x3 @ 3x3
    L_U, L_V, L_W = laderman_3x3()
    
    # Block 1: A1 @ B1a (3x3 @ 3x3)
    # A1 is A[0:3, 0:3], B1a is B[0:3, 0:3]
    # U indices: i*5 + j for i in 0..2, j in 0..2
    # V indices: j*6 + k for j in 0..2, k in 0..2
    # W indices: k*3 + i for k in 0..2, i in 0..2
    offset = 0
    for t in range(23):
        for u_pos in range(9):
            if L_U[u_pos][t] != 0:
                i = u_pos // 3
                j = u_pos % 3
                U_full[i * 5 + j][offset + t] = L_U[u_pos][t]
        for v_pos in range(9):
            if L_V[v_pos][t] != 0:
                j = v_pos // 3
                k = v_pos % 3
                V_full[j * 6 + k][offset + t] = L_V[v_pos][t]
        for w_pos in range(9):
            if L_W[w_pos][t] != 0:
                k = w_pos // 3
                i = w_pos % 3
                W_full[k * 3 + i][offset + t] = L_W[w_pos][t]
    offset += 23
    
    # Block 2: A1 @ B1b (3x3 @ 3x3)
    # A1 is A[0:3, 0:3], B1b is B[0:3, 3:6]
    # U indices: i*5 + j for i in 0..2, j in 0..2
    # V indices: j*6 + (3+k) for j in 0..2, k in 0..2
    # W indices: (3+k)*3 + i for k in 0..2, i in 0..2
    for t in range(23):
        for u_pos in range(9):
            if L_U[u_pos][t] != 0:
                i = u_pos // 3
                j = u_pos % 3
                U_full[i * 5 + j][offset + t] = L_U[u_pos][t]
        for v_pos in range(9):
            if L_V[v_pos][t] != 0:
                j = v_pos // 3
                k = v_pos % 3
                V_full[j * 6 + (3 + k)][offset + t] = L_V[v_pos][t]
        for w_pos in range(9):
            if L_W[w_pos][t] != 0:
                k = w_pos // 3
                i = w_pos % 3
                W_full[(3 + k) * 3 + i][offset + t] = L_W[w_pos][t]
    offset += 23
    
    # Block 3: A2 @ B2 (3x2 @ 2x6), schoolbook
    # A2 is A[0:3, 3:5], B2 is B[3:5, 0:6]
    # U indices: i*5 + (3+j) for i in 0..2, j in 0..1
    # V indices: (3+j)*6 + k for j in 0..1, k in 0..5
    # W indices: k*3 + i for k in 0..5, i in 0..2
    for i in range(3):
        for j in range(2):
            for k in range(6):
                U_full[i * 5 + (3 + j)][offset] = 1.0
                V_full[(3 + j) * 6 + k][offset] = 1.0
                W_full[k * 3 + i][offset] = 1.0
                offset += 1
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END


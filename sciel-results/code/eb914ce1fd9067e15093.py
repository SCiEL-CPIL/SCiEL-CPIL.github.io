"""Optimized tensor decomposition for 3x3x7 matrix multiplication using Strassen."""

def construct():
    """
    Construct a low-rank tensor decomposition for multiplying a 3x3 matrix
    by a 3x7 matrix.
    
    Rank: 62 (Improved from schoolbook 63)
    """
    n, m, p = 3, 3, 7
    rank = 62
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 62
    V = [[0] * rank for _ in range(m * p)]  # 21 x 62
    W = [[0] * rank for _ in range(p * n)]  # 21 x 62
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        """
        Add a rank-1 term to the decomposition.
        u_indices: list of (index_in_U, coeff)
        v_indices: list of (index_in_V, coeff)
        w_indices: list of (index_in_W, coeff)
        """
        nonlocal col
        for idx, coeff in u_indices:
            U[idx][col] += coeff
        for idx, coeff in v_indices:
            V[idx][col] += coeff
        for idx, coeff in w_indices:
            W[idx][col] += coeff
        col += 1

    # --- Part 1: Strassen for A_TL * B_TL (2x2 * 2x2) ---
    # A_TL indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # B_TL indices: 0(0,0), 1(0,1), 7(1,0), 8(1,1)
    # C_TL indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    
    # M1 = A[1,1] * (B[0,0] - B[1,1])
    add_term([(4, 1)], [(0, 1), (8, -1)], [(0, 1), (4, 1)])
    
    # M2 = (A[0,0] + A[1,1]) * B[1,1]
    add_term([(0, 1), (4, 1)], [(8, 1)], [(3, 1), (4, -1)])
    
    # M3 = (A[1,0] + A[1,1]) * B[0,0]
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (4, 1)])
    
    # M4 = A[0,0] * (B[1,0] - B[0,0])
    add_term([(0, 1)], [(7, 1), (0, -1)], [(0, 1), (3, 1)])
    
    # M5 = (A[0,0] + A[0,1]) * B[1,0]
    add_term([(0, 1), (1, 1)], [(7, 1)], [(0, -1), (1, 1)])
    
    # M6 = (A[0,1] - A[1,0]) * (B[0,0] + B[1,0])
    add_term([(1, 1), (3, -1)], [(0, 1), (7, 1)], [(4, 1)])
    
    # M7 = (A[0,1] - A[1,1]) * (B[1,1] + B[0,0])
    add_term([(1, 1), (4, -1)], [(8, 1), (0, 1)], [(0, 1)])

    # --- Part 2: Schoolbook for A_TL * B_TR (2x2 * 2x5) ---
    # B_TR columns k in 2..6
    # A_TL same as above
    # C_TL columns k in 2..6 (indices k, 3+k)
    
    for k in range(2, 7):
        # C[0,k] += A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] += A[1,0]*B[0,k] + A[1,1]*B[1,k]
        
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        add_term([(0, 1)], [(k, 1)], [(k, 1)])
        
        # Term 2: A[0,1] * B[1,k] -> C[0,k]
        add_term([(1, 1)], [(7 + k, 1)], [(k, 1)])
        
        # Term 3: A[1,0] * B[0,k] -> C[1,k]
        add_term([(3, 1)], [(k, 1)], [(3 + k, 1)])
        
        # Term 4: A[1,1] * B[1,k] -> C[1,k]
        add_term([(4, 1)], [(7 + k, 1)], [(3 + k, 1)])

    # --- Part 3: A_TR * B_BOT (2x1 * 1x7) ---
    # A_TR indices: 2(0,2), 5(1,2)
    # B_BOT indices: 14+j (2,j)
    # C_TOP indices: j(0,j), 3+j(1,j)
    
    for j in range(7):
        # C[0,j] += A[0,2] * B[2,j]
        add_term([(2, 1)], [(14 + j, 1)], [(j, 1)])
        
        # C[1,j] += A[1,2] * B[2,j]
        add_term([(5, 1)], [(14 + j, 1)], [(3 + j, 1)])

    # --- Part 4: A_BL * B_TOP (1x2 * 2x7) ---
    # A_BL indices: 6(2,0), 7(2,1)
    # B_TOP indices: k(0,k), 7+k(1,k)
    # C_BOT indices: 12+j (2,j)
    
    for k in range(7):
        # C[2,k] += A[2,0] * B[0,k]
        add_term([(6, 1)], [(k, 1)], [(12 + k, 1)])
        
        # C[2,k] += A[2,1] * B[1,k]
        add_term([(7, 1)], [(7 + k, 1)], [(12 + k, 1)])

    # --- Part 5: A_BR * B_BOT (1x1 * 1x7) ---
    # A_BR index: 8(2,2)
    # B_BOT indices: 14+j (2,j)
    # C_BOT indices: 12+j (2,j)
    
    for j in range(7):
        # C[2,j] += A[2,2] * B[2,j]
        add_term([(8, 1)], [(14 + j, 1)], [(12 + j, 1)])

    return {"U": U, "V": V, "W": W}

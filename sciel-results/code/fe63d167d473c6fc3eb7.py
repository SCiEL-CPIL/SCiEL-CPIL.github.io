def construct():
    n, m, p = 3, 3, 8
    
    # We'll build U, V, W for the full 3x3x8 problem
    # Split into three subproblems:
    # 1. A(3x3) * B(3x3) -> C(3x3) for columns 0-2 of B
    # 2. A(3x3) * B(3x3) -> C(3x3) for columns 3-5 of B  
    # 3. A(3x3) * B(3x2) -> C(3x2) for columns 6-7 of B
    
    # For subproblems 1 and 2, use Laderman (rank 23)
    # For subproblem 3, use a rank-14 decomposition
    
    # First, let's define the Laderman 3x3x3 decomposition
    # Each term: (u_entries, v_entries, w_entries)
    # u_entries: list of (index, value) where index = i*3+j
    # v_entries: list of (index, value) where index = j*3+k
    # w_entries: list of (index, value) where index = k*3+i
    
    laderman_333 = [
        # Term 0: a00*b00 -> c00
        ([(0, 1)], [(0, 1)], [(0, 1)]),
        # Term 1: a01*b10 -> c00  
        ([(1, 1)], [(3, 1)], [(0, 1)]),
        # Term 2: a02*b20 -> c00
        ([(2, 1)], [(6, 1)], [(0, 1)]),
        # Term 3: a10*b01 -> c10
        ([(3, 1)], [(1, 1)], [(3, 1)]),
        # Term 4: a11*b11 -> c11
        ([(4, 1)], [(4, 1)], [(4, 1)]),
        # Term 5: a12*b21 -> c11
        ([(5, 1)], [(7, 1)], [(4, 1)]),
        # Term 6: a20*b02 -> c20
        ([(6, 1)], [(2, 1)], [(6, 1)]),
        # Term 7: a21*b12 -> c21
        ([(7, 1)], [(5, 1)], [(7, 1)]),
        # Term 8: a22*b22 -> c22
        ([(8, 1)], [(8, 1)], [(8, 1)]),
    ]
    # This is just schoolbook for 9 terms. I need the full 23-term Laderman.
    # Let me abandon this manual approach and use a verified numerical method.
    
    # Instead, let me just use schoolbook for everything but try to optimize.
    # Schoolbook gives rank 72.
    
    # Actually, let me try a different approach: use the fact that we can
    # share some computations between the two 3x3 subproblems.
    
    # For now, let me just return the schoolbook decomposition to ensure correctness.
    rank = n * m * p  # 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

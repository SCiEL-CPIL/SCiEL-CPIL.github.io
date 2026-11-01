"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy:
    # Split m=5 into 1 + 4.
    # Part 1: 2x1x6 -> Rank 12
    # Part 2: 2x4x6 -> Split m=4 into 2+2.
    #         2 * Rank(2x2x6)
    #         Rank(2x2x6) -> Split p=6 into 2+2+2.
    #         3 * Rank(2x2x2)
    #         Rank(2x2x2) -> Strassen Rank 7
    # Total Rank = 12 + 2 * 3 * 7 = 12 + 42 = 54.
    
    rank = 54
    
    # Initialize matrices
    # U is (n*m, rank) = (10, 54)
    # V is (m*p, rank) = (30, 54)
    # W is (p*n, rank) = (12, 54)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen term for a 2x2x2 block
    # local_i in {0,1}, local_j in {0,1}, local_k in {0,1}
    # global_i_offset, global_j_offset, global_k_offset
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # 7 Strassen terms
        # Each term is (u_coeffs, v_coeffs, w_coeffs)
        # u: (A00, A01, A10, A11)
        # v: (B00, B01, B10, B11)
        # w: (C00, C01, C10, C11)
        
        terms = [
            # T1: (a+d)(e+h) -> C00+=1, C11+=1
            ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
            # T2: (c+d)e -> C10+=1, C11-=1
            ([2, 3], [1, 1], [0], [1], [2], [1], [3], [-1]),
            # T3: a(f-h) -> C01+=1, C11+=1
            ([0], [1], [1, 3], [1, -1], [1], [1], [3], [1]),
            # T4: d(g-e) -> C00+=1, C10+=1
            ([3], [1], [2, 0], [1, -1], [0], [1], [2], [1]),
            # T5: (a+b)h -> C00-=1, C01+=1
            ([0, 1], [1, 1], [3], [1], [0], [-1], [1], [1]),
            # T6: (c-a)(e+f) -> C11+=1
            ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
            # T7: (b-d)(g+h) -> C00+=1
            ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
        ]
        
        for term in terms:
            # Parse term structure
            # The structure varies slightly in length in my head, let's standardize
            # Actually, let's just use lists of (index, value) for each vector
            
            # Let's redefine terms as lists of tuples for clarity and correctness
            pass

        # Let's implement the application directly using the verified coefficients
        # T1: U: a(0,0), d(1,1); V: e(0,0), h(1,1); W: C00(0,0), C11(1,1)
        u_idx = [(0,0,1), (1,1,1)]
        v_idx = [(0,0,1), (1,1,1)]
        w_idx = [(0,0,1), (1,1,1)]
        _apply_term(u_idx, v_idx, w_idx, i_off, j_off, k_off)
        
        # T2: U: c(1,0), d(1,1); V: e(0,0); W: C10(1,0), C11(1,1)
        u_idx = [(1,0,1), (1,1,1)]
        v_idx = [(0,0,1)]
        w_idx = [(1,0,1), (1,1,-1)]
        _apply_term(u_idx, v_idx, w_idx, i_off, j_off, k_off)
        
        # T3: U: a(0,0); V: f(0,1), h(1,1); W: C01(0,1), C11(1,1)
        u_idx = [(0,0,1)]
        v_idx = [(0,1,1), (1,1,-1)]
        w_idx = [(0,1,1), (1,1,1)]
        _apply_term(u_idx, v_idx, w_idx, i_off, j_off, k_off)
        
        # T4: U: d(1,1); V: g(1,0), e(0,0); W: C00(0,0), C10(1,0)
        u_idx = [(1,1,1)]
        v_idx = [(1,0,1), (0,0,-1)]
        w_idx = [(0,0,1), (1,0,1)]
        _apply_term(u_idx, v_idx, w_idx, i_off, j_off, k_off)
        
        # T5: U: a(0,0), b(0,1); V: h(1,1); W: C00(0,0), C01(0,1)
        u_idx = [(0,0,1), (0,1,1)]
        v_idx = [(1,1,1)]
        w_idx = [(0,0,-1), (0,1,1)]
        _apply_term(u_idx, v_idx, w_idx, i_off, j_off, k_off)
        
        # T6: U: c(1,0), a(0,0); V: e(0,0), f(0,1); W: C11(1,1)
        u_idx = [(1,0,1), (0,0,-1)]
        v_idx = [(0,0,1), (0,1,1)]
        w_idx = [(1,1,1)]
        _apply_term(u_idx, v_idx, w_idx, i_off, j_off, k_off)
        
        # T7: U: b(0,1), d(1,1); V: g(1,0), h(1,1); W: C00(0,0)
        u_idx = [(0,1,1), (1,1,-1)]
        v_idx = [(1,0,1), (1,1,1)]
        w_idx = [(0,0,1)]
        _apply_term(u_idx, v_idx, w_idx, i_off, j_off, k_off)

    def _apply_term(u_idx, v_idx, w_idx, i_off, j_off, k_off):
        nonlocal col
        for (li, lj, val) in u_idx:
            gi = i_off + li
            gj = j_off + lj
            U[gi * m + gj][col] += val
        for (lj, lk, val) in v_idx:
            gj = j_off + lj
            gk = k_off + lk
            V[gj * p + gk][col] += val
        for (li, lk, val) in w_idx:
            gi = i_off + li
            gk = k_off + lk
            W[gk * n + gi][col] += val
        col += 1

    # Part 1: 2x1x6
    # A rows 0-1, cols 0-0. B rows 0-0, cols 0-5.
    # C rows 0-1, cols 0-5.
    # Multiplications: A[i,0] * B[0,k] -> C[i,k]
    for i in range(2):
        for k in range(6):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: 2x4x6
    # Split into two 2x2x6 blocks.
    # Block 1: j in [1, 2]
    # Block 2: j in [3, 4]
    
    for block_idx in range(2):
        j_base = 1 + block_idx * 2
        # For each 2x2x6 block, split p=6 into three 2x2x2 blocks.
        # k in [0,1], [2,3], [4,5]
        for s in range(3):
            k_base = s * 2
            add_strassen_block(0, j_base, k_base)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

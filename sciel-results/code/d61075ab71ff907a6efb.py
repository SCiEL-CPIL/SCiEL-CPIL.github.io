"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS # 3, 5, 6
    
    # We will build the decomposition by summing smaller tensor decompositions.
    # Total rank will be the sum of the ranks of the parts.
    
    # Part 1: A1 (3x2) * B1 (2x6)
    # Part 2: A2 (3x2) * B2 (2x6)
    # Part 3: A3 (3x1) * B3 (1x6)
    
    # We need to generate U, V, W for each part and then concatenate them.
    # U has shape (n*m, rank) = (15, rank)
    # V has shape (m*p, rank) = (30, rank)
    # W has shape (p*n, rank) = (18, rank)
    
    # Indices:
    # A is 3x5. Row i (0..2), Col k (0..4). Index in U: i*5 + k.
    # B is 5x6. Row k (0..4), Col j (0..5). Index in V: k*6 + j.
    # C is 3x6. Row i (0..2), Col j (0..5). Index in W: j*3 + i (Trace convention W[k*n+i, r] -> W[j*n+i, r]).
    # Wait, the task says W[k*n+i, r]. In matrix mult C = AB, C[i, j] = sum_k A[i, k] B[k, j].
    # The tensor is T[i, k, j] = A[i, k] B[k, j].
    # The output is C[i, j].
    # The task says "output factor follows the official trace convention W[k*n+i,r]".
    # Usually W index corresponds to the output matrix flattened.
    # If output is C[i, j], and convention is W[k*n+i], it might imply the output is indexed by (k, i) where k is col, i is row?
    # Let's check the seed code.
    # Seed: W[k * n + i][column] = 1.
    # Loop: for i in range(n): for j in range(m): for k in range(p):
    # U[i * m + j][column] = 1 (A[i, j])
    # V[j * p + k][column] = 1 (B[j, k])
    # W[k * n + i][column] = 1 (C[i, k]?)
    # In seed, A is n x m, B is m x p. C is n x p.
    # C[i, k] corresponds to row i, col k.
    # W index is k * n + i. This is col-major flattening of C.
    # So W index = col * n + row.
    
    # My indices:
    # A (3x5): i in 0..2, k in 0..4. U_idx = i*5 + k.
    # B (5x6): k in 0..4, j in 0..5. V_idx = k*6 + j.
    # C (3x6): i in 0..2, j in 0..5. W_idx = j*3 + i.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_decomposition(u_vals, v_vals, w_vals):
        # u_vals: list of (u_idx, val)
        # v_vals: list of (v_idx, val)
        # w_vals: list of (w_idx, val)
        # This function is a placeholder for logic. We will build the matrices directly.
        pass

    # We will create the full U, V, W matrices with a total rank R.
    # First, calculate total rank.
    # Rank(A1 B1) = 33
    # Rank(A2 B2) = 33
    # Rank(A3 B3) = 18
    # Total Rank = 84.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term: u_idx, v_idx, w_idx, coeff
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # --- Part 1: A1 (rows 0..2, cols 0..1) * B1 (rows 0..1, cols 0..5) ---
    # A1 is 3x2. B1 is 2x6.
    # Split A1 into A11 (2x2, rows 0..1) and A12 (1x2, row 2).
    # Split B1 into B11 (2x2, cols 0..1), B12 (2x2, cols 2..3), B13 (2x2, cols 4..5).
    
    # A11 * B11 (2x2 * 2x2)
    # Strassen 2x2x2
    # A11 indices: i in 0..1, k in 0..1. U_idx = i*5 + k.
    # B11 indices: k in 0..1, j in 0..1. V_idx = k*6 + j.
    # C11 indices: i in 0..1, j in 0..1. W_idx = j*3 + i.
    
    # Strassen formulas for A*B=C:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map the Strassen multiplications to the global U, V, W.
    # Each "multiplication" in Strassen is a product of a linear combo of A and a linear combo of B.
    # This corresponds to a rank-1 term in the tensor decomposition?
    # No, the rank-7 decomposition consists of 7 terms, each of which is a product of a linear form in A and a linear form in B, contributing to linear forms in C.
    
    # Actually, the standard Strassen decomposition gives 7 terms.
    # Term 1: U1 = A00+A11, V1 = B00+B11, W1 = C00+C11 - C01 - C10 (coeffs)
    # It's easier to use the known 7 vectors.
    
    # Let's use a helper function to add a 2x2x2 Strassen block.
    def add_strassen_222(u_base, v_base, w_base):
        # u_base: (i, k) -> global u_idx
        # v_base: (k, j) -> global v_idx
        # w_base: (i, j) -> global w_idx
        # Returns 7 columns added.
        
        # We will define the 7 terms explicitly.
        # This is tedious but doable.
        # Alternatively, we can use the fact that we just need to fill the matrices.
        
        # Let's hardcode the 7 terms for 2x2x2.
        # The tensors are:
        # T1: U=(1,0,1,0) V=(1,0,0,1) W=(1,0,0,1) -> A00+B00... wait.
        
        # Standard Strassen vectors (A, B, C):
        # 1: A=(1,0,0,1), B=(1,0,0,1), C=(1,0,0,1)
        # 2: A=(0,1,1,1), B=(1,0,0,0), C=(0,0,1,0)
        # 3: A=(1,0,0,0), B=(0,1,-1,0), C=(0,1,0,0)
        # 4: A=(0,0,0,1), B=(0,1,1,0), C=(0,0,0,1) -> Wait, checking formulas.
        
        # Let's use the formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mk is a product of a linear form in A and a linear form in B.
        # So each Mk corresponds to a rank-1 term in the decomposition of the multiplication tensor?
        # No, the sum of these 7 rank-1 terms (with appropriate signs in C) gives the identity.
        
        # So we add 7 terms to U, V, W.
        
        # Term 1: M1
        # A_vec: (1, 0, 0, 1) -> A00, A11
        # B_vec: (1, 0, 0, 1) -> B00, B11
        # C_vec: (1, 0, 0, 1) -> C00, C11 (with sign +1)
        
        # We need to map local indices (0..3) to global indices.
        
        def add_term_222(a_coeffs, b_coeffs, c_coeffs):
            nonlocal col
            # a_coeffs: dict {local_idx: coeff}
            # b_coeffs: dict {local_idx: coeff}
            # c_coeffs: dict {local_idx: coeff}
            
            for la, ca in a_coeffs.items():
                gi, gk = la // 2, la % 2
                u_idx = u_base(gi, gk)
                U[u_idx][col] += ca
            for lb, cb in b_coeffs.items():
                gk, gj = lb // 2, lb % 2
                v_idx = v_base(gk, gj)
                V[v_idx][col] += cb
            for lc, cc in c_coeffs.items():
                gi, gj = lc // 2, lc % 2
                w_idx = w_base(gi, gj)
                W[w_idx][col] += cc
            col += 1

        # M1
        add_term_222({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
        # M2
        add_term_222({1: 1, 3: 1}, {0: 1}, {2: 1})
        # M3
        add_term_222({0: 1}, {1: 1, 3: -1}, {1: 1})
        # M4
        add_term_222({3: 1}, {2: 1, 0: -1}, {2: 1})
        # M5
        add_term_222({0: 1, 1: 1}, {3: 1}, {0: -1})
        # M6
        add_term_222({1: 1, 0: -1}, {0: 1, 2: 1}, {3: 1})
        # M7
        add_term_222({2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1})

    # Define base index functions for A11, B11, C11
    # A11: i in 0..1, k in 0..1. Global U: i*5 + k
    def u_base_A11(i, k): return i * 5 + k
    # B11: k in 0..1, j in 0..1. Global V: k*6 + j
    def v_base_B11(k, j): return k * 6 + j
    # C11: i in 0..1, j in 0..1. Global W: j*3 + i
    def w_base_C11(i, j): return j * 3 + i

    add_strassen_222(u_base_A11, v_base_B11, w_base_C11)
    
    # A11 * B12 (2x2 * 2x2)
    # B12: k in 0..1, j in 2..3. Global V: k*6 + j
    def v_base_B12(k, j): return k * 6 + (j + 2)
    # C12: i in 0..1, j in 2..3. Global W: j*3 + i
    def w_base_C12(i, j): return (j + 2) * 3 + i
    add_strassen_222(u_base_A11, v_base_B12, w_base_C12)

    # A11 * B13 (2x2 * 2x2)
    # B13: k in 0..1, j in 4..5. Global V: k*6 + j
    def v_base_B13(k, j): return k * 6 + (j + 4)
    # C13: i in 0..1, j in 4..5. Global W: j*3 + i
    def w_base_C13(i, j): return (j + 4) * 3 + i
    add_strassen_222(u_base_A11, v_base_B13, w_base_C13)

    # A12 * B11 (1x2 * 2x2)
    # A12: i=2, k in 0..1. Global U: 2*5 + k
    # B11: k in 0..1, j in 0..1. Global V: k*6 + j
    # C12_row: i=2, j in 0..1. Global W: j*3 + 2
    def add_rank4_122(u_base_12, v_base_22, w_base_12):
        nonlocal col
        # A is 1x2. B is 2x2. C is 1x2.
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        # 4 terms.
        
        # Term 1: A0 B00 -> C0
        U[u_base_12(0)][col] += 1
        V[v_base_22(0, 0)][col] += 1
        W[w_base_12(0)][col] += 1
        col += 1
        
        # Term 2: A1 B10 -> C0
        U[u_base_12(1)][col] += 1
        V[v_base_22(1, 0)][col] += 1
        W[w_base_12(0)][col] += 1
        col += 1
        
        # Term 3: A0 B01 -> C1
        U[u_base_12(0)][col] += 1
        V[v_base_22(0, 1)][col] += 1
        W[w_base_12(1)][col] += 1
        col += 1
        
        # Term 4: A1 B11 -> C1
        U[u_base_12(1)][col] += 1
        V[v_base_22(1, 1)][col] += 1
        W[w_base_12(1)][col] += 1
        col += 1

    def u_base_A12(k): return 2 * 5 + k
    def w_base_C12_row(j): return j * 3 + 2
    add_rank4_122(u_base_A12, v_base_B11, w_base_C12_row)

    # A12 * B12
    def w_base_C12_row2(j): return (j + 2) * 3 + 2
    add_rank4_122(u_base_A12, v_base_B12, w_base_C12_row2)

    # A12 * B13
    def w_base_C12_row3(j): return (j + 4) * 3 + 2
    add_rank4_122(u_base_A12, v_base_B13, w_base_C12_row3)

    # --- Part 2: A2 (rows 0..2, cols 2..3) * B2 (rows 2..3, cols 0..5) ---
    # A2 is 3x2. B2 is 2x6.
    # Same structure as Part 1.
    # A2 indices: i in 0..2, k in 2..3. U_idx = i*5 + k.
    # B2 indices: k in 2..3, j in 0..5. V_idx = k*6 + j.
    # C2 indices: i in 0..2, j in 0..5. W_idx = j*3 + i.
    
    # A21 (2x2, rows 0..1, cols 2..3)
    def u_base_A21(i, k): return i * 5 + (k + 2)
    # B21 (2x2, rows 2..3, cols 0..1)
    def v_base_B21(k, j): return (k + 2) * 6 + j
    # C21 (2x2, rows 0..1, cols 0..1)
    def w_base_C21(i, j): return j * 3 + i
    add_strassen_222(u_base_A21, v_base_B21, w_base_C21)

    # A21 * B22 (2x2 * 2x2)
    def v_base_B22(k, j): return (k + 2) * 6 + (j + 2)
    def w_base_C22(i, j): return (j + 2) * 3 + i
    add_strassen_222(u_base_A21, v_base_B22, w_base_C22)

    # A21 * B23 (2x2 * 2x2)
    def v_base_B23(k, j): return (k + 2) * 6 + (j + 4)
    def w_base_C23(i, j): return (j + 4) * 3 + i
    add_strassen_222(u_base_A21, v_base_B23, w_base_C23)

    # A22 (1x2, row 2, cols 2..3)
    def u_base_A22(k): return 2 * 5 + (k + 2)
    # C22_row (1x2, row 2, cols 0..1)
    def w_base_C22_row(j): return j * 3 + 2
    add_rank4_122(u_base_A22, v_base_B21, w_base_C22_row)

    def w_base_C22_row2(j): return (j + 2) * 3 + 2
    add_rank4_122(u_base_A22, v_base_B22, w_base_C22_row2)

    def w_base_C22_row3(j): return (j + 4) * 3 + 2
    add_rank4_122(u_base_A22, v_base_B23, w_base_C22_row3)

    # --- Part 3: A3 (rows 0..2, col 4) * B3 (row 4, cols 0..5) ---
    # A3 is 3x1. B3 is 1x6.
    # C3 is 3x6.
    # C3[i, j] = A3[i, 0] * B3[0, j]
    # 18 terms.
    
    def u_base_A3(i): return i * 5 + 4
    def v_base_B3(j): return 4 * 6 + j
    def w_base_C3(i, j): return j * 3 + i
    
    for i in range(3):
        for j in range(6):
            add_term(u_base_A3(i), v_base_B3(j), w_base_C3(i, j), 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

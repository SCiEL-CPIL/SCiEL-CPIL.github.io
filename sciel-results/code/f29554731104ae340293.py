"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Part 1: A_2x2 * B_2x5
    # A indices: 0,1,2,3 (00,01,10,11)
    # B indices: 0..9 (00,01,02,03,04, 10,11,12,13,14)
    # W indices: 0..9 (00,10,01,11,02,12,03,13,04,14) -> k*n+i
    
    # Sub-part 1a: B cols 0,1 (2x2x2)
    # M1
    add_term(0, 1, 0, 1, 0, 1) # A00, B00, C00
    add_term(3, 1, 6, 1, 3, 1) # A11, B11, C11
    # Wait, add_term increments col. I need to fill the SAME column for one term.
    # My helper is wrong. It increments col for each entry.
    # I should fill U, V, W for the current col, then increment.

    # Reset
    col = 0
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] = val
        for idx, val in v_coeffs.items():
            V[idx][col] = val
        for idx, val in w_coeffs.items():
            W[idx][col] = val
        col += 1

    # 1a: 2x2x2 (Cols 0,1 of B)
    # A: 0:00, 1:01, 2:10, 3:11
    # B: 0:00, 1:01, 5:10, 6:11
    # W: 0:00, 2:10, 1:01, 3:11 (k*n+i: 0*2+0=0, 0*2+1=1? No. W[k*n+i]. k is B col. i is A row.
    # C_ik. i is row of A (0,1). k is col of B (0,1).
    # W row index = k*n + i.
    # C_00: k=0, i=0 -> 0
    # C_01: k=1, i=0 -> 2
    # C_10: k=0, i=1 -> 1
    # C_11: k=1, i=1 -> 3
    
    # M1: (A00+A11)(B00+B11) -> C00(1), C11(1)
    add_strassen_term({0:1, 3:1}, {0:1, 6:1}, {0:1, 3:1})
    # M2: (A10+A11)B00 -> C10(1), C11(-1)
    add_strassen_term({2:1, 3:1}, {0:1}, {1:1, 3:-1})
    # M3: A00(B10-B11) -> C01(1), C11(1)
    add_strassen_term({0:1}, {5:1, 6:-1}, {2:1, 3:1})
    # M4: A11(B00-B10) -> C00(1)
    add_strassen_term({3:1}, {0:1, 5:-1}, {0:1})
    # M5: (A00+A01)B11 -> C00(-1), C01(1)
    add_strassen_term({0:1, 1:1}, {6:1}, {0:-1, 2:1})
    # M6: (A10-A00)(B00+B01) -> C11(1)
    add_strassen_term({2:1, 0:-1}, {0:1, 1:1}, {3:1})
    # M7: (A01-A11)(B10+B11) -> C10(1), C00(-1)
    add_strassen_term({1:1, 3:-1}, {5:1, 6:1}, {1:1, 0:-1})

    # 1b: 2x2x2 (Cols 2,3 of B)
    # B: 2:02, 3:03, 7:12, 8:13
    # W: C02(k=2,i=0)->4, C12(k=2,i=1)->5, C03(k=3,i=0)->6, C13(k=3,i=1)->7
    
    add_strassen_term({0:1, 3:1}, {2:1, 8:1}, {4:1, 7:1})
    add_strassen_term({2:1, 3:1}, {2:1}, {5:1, 7:-1})
    add_strassen_term({0:1}, {7:1, 8:-1}, {6:1, 7:1})
    add_strassen_term({3:1}, {2:1, 7:-1}, {4:1})
    add_strassen_term({0:1, 1:1}, {8:1}, {4:-1, 6:1})
    add_strassen_term({2:1, 0:-1}, {2:1, 3:1}, {7:1})
    add_strassen_term({1:1, 3:-1}, {7:1, 8:1}, {5:1, 4:-1})

    # 1c: 2x2x1 (Col 4 of B)
    # B: 4:04, 9:14
    # W: C04(k=4,i=0)->8, C14(k=4,i=1)->9
    
    # M1: A00 B04 -> C04
    add_strassen_term({0:1}, {4:1}, {8:1})
    # M2: A01 B14 -> C04
    add_strassen_term({1:1}, {9:1}, {8:1})
    # M3: A10 B04 -> C14
    add_strassen_term({2:1}, {4:1}, {9:1})
    # M4: A11 B14 -> C14
    add_strassen_term({3:1}, {9:1}, {9:1})

    # Part 2: A_2x1 * B_1x5
    # A: 4:02, 5:12
    # B: 10:20, 11:21, 12:22, 13:23, 14:24
    # W: C_i,k. k is B col (0..4). i is A row (0,1).
    # C_00(k=0,i=0)->0, C_10(k=0,i=1)->1
    # C_01(k=1,i=0)->2, C_11(k=1,i=1)->3
    # C_02(k=2,i=0)->4, C_12(k=2,i=1)->5
    # C_03(k=3,i=0)->6, C_13(k=3,i=1)->7
    # C_04(k=4,i=0)->8, C_14(k=4,i=1)->9
    
    # 10 terms
    for k in range(5):
        # A02 * B2k -> C0k
        add_strassen_term({4:1}, {10+k:1}, {k*2:1})
        # A12 * B2k -> C1k
        add_strassen_term({5:1}, {10+k:1}, {k*2+1:1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for the 2x2 core (indices 0,1)
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10, C11
    # M3 = A00(B01-B11) -> C01, C11
    # M4 = A11(B10-B00) -> C00, C10
    # M5 = (A00+A01)B11 -> C00, C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00, C01
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict, col):
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val

    # M1
    add_term({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1}, 0)
    # M2
    add_term({3: 1, 4: 1}, {0: 1}, {1: 1, 4: 1}, 1) # C10 (idx 1), C11 (idx 4)
    # M3
    add_term({0: 1}, {1: 1, 5: -1}, {3: 1, 4: 1}, 2) # C01 (idx 3), C11 (idx 4)
    # M4
    add_term({4: 1}, {3: 1, 0: -1}, {0: 1, 1: 1}, 3) # C00 (idx 0), C10 (idx 1)
    # M5
    add_term({0: 1, 1: 1}, {4: 1}, {0: 1, 3: 1}, 4) # C00 (idx 0), C01 (idx 3)
    # M6
    add_term({3: 1, 0: -1}, {0: 1, 1: 1}, {4: 1}, 5) # C11 (idx 4)
    # M7
    add_term({1: 1, 4: -1}, {3: 1, 4: 1}, {0: 1, 3: 1}, 6) # C00 (idx 0), C01 (idx 3)

    # Schoolbook terms for the rest (19 terms)
    col = 7
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Skip the 2x2 core (i,j,k in {0,1})
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

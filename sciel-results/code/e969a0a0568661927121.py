"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        U[u_idx][term_id] += val
        V[v_idx][term_id] += val
        W[w_idx][term_id] += val

    term_id = 0

    # Strassen Terms (1-7)
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0) # C00
    add_term(0, 0, 1) # C01
    add_term(0, 0, 3) # C10
    add_term(0, 0, 4) # C11
    U[4][term_id] += 1 # A11
    V[4][term_id] += 1 # B11
    term_id += 1

    # M2 = (A10+A11)B00
    add_term(3, 0, 3) # C10
    add_term(3, 0, 4) # C11 (-1)
    U[4][term_id] += 1 # A11
    V[0][term_id] += 1 # B00
    W[4][term_id] -= 1
    term_id += 1

    # M3 = A00(B10-B11)
    add_term(0, 3, 1) # C01
    add_term(0, 3, 4) # C11
    V[4][term_id] -= 1 # B11
    term_id += 1

    # M4 = A11(B01-B00)
    add_term(4, 1, 0) # C00
    add_term(4, 1, 3) # C10
    V[0][term_id] -= 1 # B00
    term_id += 1

    # M5 = (A00+A10)B11
    add_term(0, 4, 0) # C00 (-1)
    add_term(0, 4, 1) # C01
    U[3][term_id] += 1 # A10
    W[0][term_id] -= 1
    term_id += 1

    # M6 = (A10-A00)(B00+B01)
    add_term(3, 0, 4) # C11
    add_term(3, 1, 4) # C11
    U[0][term_id] -= 1 # A00
    term_id += 1

    # M7 = (A01-A11)(B10+B11)
    add_term(1, 3, 0) # C00
    add_term(1, 3, 4) # C11
    U[4][term_id] -= 1 # A11
    V[4][term_id] += 1 # B11
    term_id += 1

    # Linear Terms (8-26)
    # T8: A02 B20 -> C00, C22
    add_term(2, 6, 0)
    add_term(2, 6, 8)
    term_id += 1

    # T9: A02 B21 -> C01
    add_term(2, 7, 1)
    term_id += 1

    # T10: A12 B20 -> C10
    add_term(5, 6, 3)
    term_id += 1

    # T11: A12 B21 -> C11, C22
    add_term(5, 7, 4)
    add_term(5, 7, 8)
    term_id += 1

    # T12: A00 B02 -> C02
    add_term(0, 2, 2)
    term_id += 1

    # T13: A01 B12 -> C02
    add_term(1, 5, 2)
    term_id += 1

    # T14: A02 B22 -> C02
    add_term(2, 8, 2)
    term_id += 1

    # T15: A10 B02 -> C12
    add_term(3, 2, 5)
    term_id += 1

    # T16: A11 B12 -> C12
    add_term(4, 5, 5)
    term_id += 1

    # T17: A12 B22 -> C12
    add_term(5, 8, 5)
    term_id += 1

    # T18: A00 B20 -> C20
    add_term(0, 6, 6)
    term_id += 1

    # T19: A10 B21 -> C20
    add_term(3, 7, 6)
    term_id += 1

    # T20: A22 B20 -> C20
    add_term(8, 6, 6)
    term_id += 1

    # T21: A01 B20 -> C21
    add_term(1, 6, 7)
    term_id += 1

    # T22: A11 B21 -> C21
    add_term(4, 7, 7)
    term_id += 1

    # T23: A22 B21 -> C21
    add_term(8, 7, 7)
    term_id += 1

    # T26: A22 B22 -> C22
    add_term(8, 8, 8)
    term_id += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

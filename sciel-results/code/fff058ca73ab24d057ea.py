"""Rewritten program for Matrix multiplication 2,2,13"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Target Rank: 41 (3*13 + 2)
    # We use a decomposition based on the structure of 2x2xp multiplication.
    # This is equivalent to 3p+2 multiplications.
    
    RANK = 41
    u_rows = n * m       # 4
    v_rows = m * p       # 26
    w_rows = p * n       # 26

    U = [[0] * RANK for _ in range(u_rows)]
    V = [[0] * RANK for _ in range(v_rows)]
    W = [[0] * RANK for _ in range(w_rows)]

    # This is a placeholder. The actual rank 41 decomposition is complex.
    # We fall back to the known valid Rank 46 decomposition.
    # If we can't find Rank 41, we return Rank 46.
    
    RANK = 46
    U = [[0] * RANK for _ in range(u_rows)]
    V = [[0] * RANK for _ in range(v_rows)]
    W = [[0] * RANK for _ in range(w_rows)]

    offset = 0

    def emit(u_map, v_map, w_map):
        nonlocal offset
        for idx, val in u_map.items():
            U[idx][offset] += val
        for idx, val in v_map.items():
            V[idx][offset] += val
        for idx, val in w_map.items():
            W[idx][offset] += val
        offset += 1

    # Six 2x2x2 Strassen blocks
    for blk in range(6):
        a = blk * 2
        b = blk * 2 + 1

        # M1 = (A00+A11)*(B0a+B1b)
        emit({0: 1, 3: 1}, {a: 1, 13 + b: 1}, {2*a: 1, 2*b + 1: 1})
        # M2 = (A10+A11)*B0a
        emit({2: 1, 3: 1}, {a: 1}, {2*a + 1: 1, 2*b + 1: -1})
        # M3 = A00*(B1b-B0b)
        emit({0: 1}, {b: 1, 13 + b: -1}, {2*b: 1, 2*b + 1: 1})
        # M4 = A11*(B1a-B0a)
        emit({3: 1}, {13 + a: 1, a: -1}, {2*a: 1, 2*a + 1: 1})
        # M5 = (A00+A01)*B1b
        emit({0: 1, 1: 1}, {13 + b: 1}, {2*a: -1, 2*b: 1})
        # M6 = (A10-A00)*(B0a+B0b)
        emit({2: 1, 0: -1}, {a: 1, b: 1}, {2*b + 1: 1})
        # M7 = (A01-A11)*(B1a+B1b)
        emit({1: 1, 3: -1}, {13 + a: 1, 13 + b: 1}, {2*a: 1})

    # Final 2x2x1 schoolbook block for column k=12
    k = 12
    emit({0: 1}, {k: 1}, {2*k: 1})
    emit({1: 1}, {13 + k: 1}, {2*k: 1})
    emit({2: 1}, {k: 1}, {2*k + 1: 1})
    emit({3: 1}, {13 + k: 1}, {2*k + 1: 1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

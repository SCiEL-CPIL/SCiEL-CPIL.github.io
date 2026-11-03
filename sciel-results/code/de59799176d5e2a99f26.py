"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # 6 rows
    V = [[0] * rank for _ in range(m * p)]   # 27 rows
    W = [[0] * rank for _ in range(p * n)]   # 18 rows

    col = 0

    # Part 1: A[:, 2] * B[2, :]
    # A indices: i in {0,1}, j=2
    # B indices: j=2, k in {0..8}
    # C indices: i in {0,1}, k in {0..8}
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    # Part 2: A[:, 0:2] * B[0:2, :]
    # A sub-block: 2x2 (cols 0,1)
    # B sub-block: 2x9 (rows 0,1)
    
    # 2a: Four 2x2x2 Strassen blocks for k=0..7
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Local indices mapping to global
        # A: i in {0,1}, j in {0,1} -> global j is 0,1
        # B: j in {0,1}, k in {0,1} -> global j is 0,1, k is k0,k1
        # C: i in {0,1}, k in {0,1} -> global i is 0,1, k is k0,k1
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00, A11
        U[0 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        # V: B00, B11
        V[0 * p + k0][col] += 1
        V[1 * p + k1][col] += 1
        # W: C00, C11
        W[k0 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}

        # M2 = (A10+A11)B00
        U[1 * m + 0][col] += 1
        U[1 * m + 1][col] += 1
        V[0 * p + k0][col] += 1
        W[k0 * n + 1][col] += 1
        col += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}

        # M3 = A00(B01-B11)
        U[0 * m + 0][col] += 1
        V[0 * p + k1][col] += 1
        V[1 * p + k1][col] -= 1
        W[k1 * n + 0][col] += 1
        col += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}

        # M4 = A11(B10-B00)
        U[1 * m + 1][col] += 1
        V[1 * p + k0][col] += 1
        V[0 * p + k0][col] -= 1
        W[k0 * n + 1][col] += 1
        W[k0 * n + 0][col] += 1
        col += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}

        # M5 = (A00+A01)B11
        U[0 * m + 0][col] += 1
        U[0 * m + 1][col] += 1
        V[1 * p + k1][col] += 1
        W[k1 * n + 1][col] += 1
        W[k1 * n + 0][col] += 1
        col += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}

        # M6 = (A10-A00)(B00+B01)
        U[1 * m + 0][col] += 1
        U[0 * m + 0][col] -= 1
        V[0 * p + k0][col] += 1
        V[0 * p + k1][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}

        # M7 = (A01-A10)(B10+B11)
        U[0 * m + 1][col] += 1
        U[1 * m + 0][col] -= 1
        V[1 * p + k0][col] += 1
        V[1 * p + k1][col] += 1
        W[k0 * n + 0][col] += 1
        col += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}

    # 2b: One 2x2x1 schoolbook block for k=8
    k0 = 8
    # C[0, k0] = A[0,0]B[0,k0] + A[0,1]B[1,k0]
    # Term 1: A[0,0] * B[0,k0] -> C[0,k0]
    U[0 * m + 0][col] = 1
    V[0 * p + k0][col] = 1
    W[k0 * n + 0][col] = 1
    col += 1
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Term 2: A[0,1] * B[1,k0] -> C[0,k0]
    U[0 * m + 1][col] = 1
    V[1 * p + k0][col] = 1
    W[k0 * n + 0][col] = 1
    col += 1
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # C[1, k0] = A[1,0]B[0,k0] + A[1,1]B[1,k0]
    # Term 3: A[1,0] * B[0,k0] -> C[1,k0]
    U[1 * m + 0][col] = 1
    V[0 * p + k0][col] = 1
    W[k0 * n + 1][col] = 1
    col += 1
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Term 4: A[1,1] * B[1,k0] -> C[1,k0]
    U[1 * m + 1][col] = 1
    V[1 * p + k0][col] = 1
    W[k0 * n + 1][col] = 1
    col += 1
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # 10 seconds deadline, well within 120s limit
    
    n, m, p = DIMENSIONS
    rank = 44

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # We process 4 blocks of B, each of size 3x2
    for b in range(4):
        if time.time() > deadline:
            break
            
        k0 = 2 * b
        k1 = 2 * b + 1

        # --- Strassen part for A_top (2x2) * B_top (2x2) ---
        # A_top flat indices: 0, 1, 3, 4 (rows 0,1; cols 0,1)
        # B_top flat indices: 0*p+k0, 0*p+k1, 1*p+k0, 1*p+k1
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][col] = 1
        U[4][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M2 = (A10 + A11) * B0k0
        U[3][col] = 1
        U[4][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1

        # M3 = A00 * (B0k1 - B1k1)
        U[0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M4 = A11 * (B1k0 - B0k0)
        U[4][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # M5 = (A00 + A01) * B1k1
        U[0][col] = 1
        U[1][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1

        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[3][col] = 1
        U[0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][col] = 1
        U[4][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

        # --- Schoolbook part for A_bot (2x1) * B_bot (1x2) ---
        # A_bot flat indices: 2, 5 (rows 0,1; col 2)
        # B_bot flat indices: 2*p+k0, 2*p+k1 (row 2; cols k0, k1)

        # P1 = A02 * B2k0
        U[2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

        # P2 = A02 * B2k1
        U[2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1

        # P3 = A12 * B2k0
        U[5][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # P4 = A12 * B2k1
        U[5][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

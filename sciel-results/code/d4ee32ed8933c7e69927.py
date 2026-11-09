"""Matrix multiplication 2x3x6 rank-minimal decomposition."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # safety margin before 120s limit

    n, m, p = 2, 3, 6

    # Strategy: Split A (2x3) into A1 (2x2) and A2 (2x1).
    # Split B (3x6) into B1 (2x6) and B2 (1x6).
    # C = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2 times 2x6): split B1 into three 2x2 blocks B1a, B1b, B1c.
    # Use Strassen's algorithm (rank 7) for each 2x2x2 block multiplication.
    # That gives 3*7 = 21 terms for A1*B1.
    #
    # For A2*B2 (2x1 times 1x6): schoolbook rank = 2*1*6 = 12 terms.
    #
    # Total rank = 21 + 12 = 33.
    #
    # Indexing conventions:
    # U has shape (n*m, rank) = (6, 33): rows indexed by i*m + j where i in [0,2), j in [0,3)
    # V has shape (m*p, rank) = (18, 33): rows indexed by j*p + k where j in [0,3), k in [0,6)
    # W has shape (p*n, rank) = (12, 33): rows indexed by k*n + i where k in [0,6), i in [0,2)
    #
    # The tensor identity is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = A[i,j]*B[j,k]
    #
    # Strassen rank-7 decomposition of 2x2*2x2:
    # For A = [[a,b],[c,d]] and B = [[e,f],[g,h]]:
    #   m1 = (a+d)(e+h)
    #   m2 = (c+d)(e)
    #   m3 = (a)(f-h)
    #   m4 = (d)(g-e)
    #   m5 = (a+b)(h)
    #   m6 = (c-a)(e+g)
    #   m7 = (b-d)(f+h)
    #   C[0,0] = m1 + m4 - m5 - m7
    #   C[0,1] = m3 + m5
    #   C[1,0] = m2 + m4
    #   C[1,1] = m1 - m2 + m3 - m6
    #
    # Each Strassen term becomes a rank-1 tensor:
    #   m1: U has A[0,0]=1, A[1,1]=1; V has B[0,0]=1, B[1,1]=1; W has C[0,0]=1, C[1,1]=1
    #   m2: U has A[0,1]=1, A[1,1]=1; V has B[0,0]=1; W has C[1,0]=1
    #   m3: U has A[0,0]=1; V has B[0,1]=1, B[1,1]=-1; W has C[0,1]=1, C[1,1]=1
    #   m4: U has A[1,1]=1; V has B[1,0]=1, B[0,0]=-1; W has C[0,0]=1, C[1,0]=1
    #   m5: U has A[0,0]=1, A[0,1]=1; V has B[1,1]=1; W has C[0,1]=1, C[1,1]=-1
    #   m6: U has A[0,1]=1, A[0,0]=-1; V has B[0,0]=1, B[1,0]=1; W has C[1,1]=-1
    #   m7: U has A[0,1]=1, A[1,1]=-1; V has B[0,1]=1, B[1,1]=1; W has C[0,0]=-1, C[1,1]=1

    rank = 33
    U = [[0] * rank for _ in range(n * m)]   # 6 rows
    V = [[0] * rank for _ in range(m * p)]   # 18 rows
    W = [[0] * rank for _ in range(p * n)]   # 12 rows

    col = 0

    # Strassen 7-term decomposition for 2x2*2x2
    # For a block at column offset 2*block_idx in B:
    #   A sub-matrix: rows i1 in {0,1}, cols j1 in {0,1} (j1 corresponds to global j = j1)
    #   B sub-matrix: rows j1 in {0,1}, cols k1 in {0,1} (global k = 2*block_idx + k1)
    #   C sub-matrix: rows i1 in {0,1}, cols k1 in {0,1} (global k = 2*block_idx + k1)

    # Define the 7 Strassen rank-1 tensors as (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict mapping (i, j) -> coeff
    # V_coeffs: dict mapping (j, k) -> coeff
    # W_coeffs: dict mapping (i, k) -> coeff

    strassen_terms = [
        # m1 = (a+d)(e+h)
        (
            {(0,0): 1, (1,1): 1},          # U: a and d
            {(0,0): 1, (1,1): 1},          # V: e and h
            {(0,0): 1, (1,1): 1},          # W: C[0,0] and C[1,1]
        ),
        # m2 = (c+d)(e)
        (
            {(0,1): 1, (1,1): 1},          # U: c and d
            {(0,0): 1},                    # V: e
            {(1,0): 1},                    # W: C[1,0]
        ),
        # m3 = (a)(f-h)
        (
            {(0,0): 1},                    # U: a
            {(0,1): 1, (1,1): -1},        # V: f and -h
            {(0,1): 1, (1,1): 1},         # W: C[0,1] and C[1,1]
        ),
        # m4 = (d)(g-e)
        (
            {(1,1): 1},                    # U: d
            {(1,0): 1, (0,0): -1},        # V: g and -e
            {(0,0): 1, (1,0): 1},         # W: C[0,0] and C[1,0]
        ),
        # m5 = (a+b)(h)
        (
            {(0,0): 1, (0,1): 1},         # U: a and b
            {(1,1): 1},                    # V: h
            {(0,1): 1, (1,1): -1},        # W: C[0,1] and -C[1,1]
        ),
        # m6 = (c-a)(e+g)
        (
            {(0,1): 1, (0,0): -1},        # U: c and -a
            {(0,0): 1, (1,0): 1},         # V: e and g
            {(1,1): -1},                  # W: -C[1,1]
        ),
        # m7 = (b-d)(f+h)
        (
            {(0,1): 1, (1,1): -1},        # U: b and -d
            {(0,1): 1, (1,1): 1},         # V: f and h
            {(0,0): -1, (1,1): 1},        # W: -C[0,0] and C[1,1]
        ),
    ]

    # Apply Strassen to each of the 3 blocks of B1 (2x6 split into three 2x2 blocks)
    for block_idx in range(3):
        k_offset = 2 * block_idx  # column offset in B (and C)
        for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
            # Fill U: A[i, j] where i in {0,1}, j in {0,1}
            for (i, j), coeff in u_coeffs.items():
                u_row = i * m + j
                U[u_row][col] = coeff

            # Fill V: B[j, k] where j in {0,1}, k = k_offset + k1
            for (j, k1), coeff in v_coeffs.items():
                v_row = j * p + (k_offset + k1)
                V[v_row][col] = coeff

            # Fill W: C[i, k] where i in {0,1}, k = k_offset + k1
            for (i, k1), coeff in w_coeffs.items():
                w_row = (k_offset + k1) * n + i
                W[w_row][col] = coeff

            col += 1

    # Schoolbook for A2*B2: A[2,2] (2x1) * B[2,:] (1x6)
    # For each i in {0,1}, k in {0..5}: C[i,k] += A[i,2] * B[2,k]
    for i in range(n):
        for k in range(p):
            u_row = i * m + 2       # A[i, 2]
            v_row = 2 * p + k       # B[2, k]
            w_row = k * n + i       # C[i, k]
            U[u_row][col] = 1
            V[v_row][col] = 1
            W[w_row][col] = 1
            col += 1

    # Verify correctness
    assert col == rank, f"Expected rank {rank}, got {col}"

    # Quick verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): sum={s}")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

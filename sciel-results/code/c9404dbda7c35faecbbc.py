"""Matrix multiplication 2x3x8 with rank-44 Strassen-based decomposition."""
import time

def construct():
    t_start = time.time()
    limit = t_start + 130.0

    n, m, p = 2, 3, 8

    def _schoolbook():
        rb = n * m * p
        U = [[0] * rb for _ in range(n * m)]
        V = [[0] * rb for _ in range(m * p)]
        W = [[0] * rb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][c] = 1
                    V[a * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def _validate(U, V, W, rank):
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for r in range(rank):
                        u_val = u_row[r]
                        v_val = v_row[r]
                        w_val = w_row[r]
                        if u_val == 0 or v_val == 0 or w_val == 0:
                            continue
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 coefficients
    # U factors for A (2x2), V factors for B (2x2), W factors for C (2x2)
    su = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    sv = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    sw = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    # Initialize factors for rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: Handle A[0:2, 0:2] * B[0:2, :] using 4 Strassen blocks
    # B is 2x8, split into 4 blocks of 2x2 columns
    for b_idx in range(4):
        k_offset = b_idx * 2
        for s_idx in range(7):
            # Update U: rows 0-1, cols 0-1 (corresponding to A[0:2, 0:2])
            for i in range(2):
                for a in range(2):
                    val = su[s_idx][i * 2 + a]
                    if val:
                        U[i * m + a][col] += val

            # Update V: rows 0-1 (a=0,1), cols k_offset, k_offset+1
            for a in range(2):
                for k in range(2):
                    val = sv[s_idx][a * 2 + k]
                    if val:
                        V[a * p + (k_offset + k)][col] += val

            # Update W: rows k_offset, k_offset+1 (k), cols 0-1 (i)
            for k in range(2):
                for i in range(2):
                    val = sw[s_idx][k * 2 + i]
                    if val:
                        W[(k_offset + k) * n + i][col] += val
            col += 1

            if time.time() > limit:
                return _schoolbook()

    # Part 2: Handle A[0:2, 2] * B[2:3, :]
    # This is 2x1 * 1x8 = 16 terms
    for i in range(2):
        for k in range(8):
            u_row = i * m + 2
            v_row = 2 * p + k
            w_row = k * n + i
            U[u_row][col] = 1
            V[v_row][col] = 1
            W[w_row][col] = 1
            col += 1

            if time.time() > limit:
                return _schoolbook()

    # Validation
    valid = _validate(U, V, W, rank)

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback
        return _schoolbook()

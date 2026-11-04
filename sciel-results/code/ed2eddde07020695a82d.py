import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Build rank-49 decomposition via Kronecker product of Strassen factors
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)

    # Permutation to map Kronecker row order to standard row-major order
    # For U: row index S = i*4 + j, where i = i_hi*2 + i_lo, j = j_hi*2 + j_lo
    # Kronecker row R = (i_hi*2 + j_hi)*4 + (i_lo*2 + j_lo)
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_hi = i // 2
        i_lo = i % 2
        j_hi = j // 2
        j_lo = j % 2
        a_idx = i_hi * 2 + j_hi
        b_idx = i_lo * 2 + j_lo
        perm[S] = a_idx * 4 + b_idx

    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]

    # Fast vectorized verification
    try:
        u_t = U4_fixed.reshape(4, 4, 49)
        v_t = V4_fixed.reshape(4, 4, 49)
        w_t = W4_fixed.reshape(4, 4, 49)
        s = np.einsum('ijr,jkr,kir->ijk', u_t, v_t, w_t, optimize=True)
        if not np.all(s == 1):
            raise ValueError("Verification failed")
    except Exception:
        if time.time() - start_time > deadline:
            n, m, p = 4, 4, 4
            rank = n * m * p
            U_f = [[0] * rank for _ in range(n * m)]
            V_f = [[0] * rank for _ in range(m * p)]
            W_f = [[0] * rank for _ in range(p * n)]
            col = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_f[ii * m + jj][col] = 1
                        V_f[jj * p + kk][col] = 1
                        W_f[kk * n + ii][col] = 1
                        col += 1
            return {"U": U_f, "V": V_f, "W": W_f}
        else:
            # If einsum fails unexpectedly, fall back to schoolbook
            n, m, p = 4, 4, 4
            rank = n * m * p
            U_f = [[0] * rank for _ in range(n * m)]
            V_f = [[0] * rank for _ in range(m * p)]
            W_f = [[0] * rank for _ in range(p * n)]
            col = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_f[ii * m + jj][col] = 1
                        V_f[jj * p + kk][col] = 1
                        W_f[kk * n + ii][col] = 1
                        col += 1
            return {"U": U_f, "V": V_f, "W": W_f}

    if time.time() - start_time > deadline:
        n, m, p = 4, 4, 4
        rank = n * m * p
        U_f = [[0] * rank for _ in range(n * m)]
        V_f = [[0] * rank for _ in range(m * p)]
        W_f = [[0] * rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_f[ii * m + jj][col] = 1
                    V_f[jj * p + kk][col] = 1
                    W_f[kk * n + ii][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }

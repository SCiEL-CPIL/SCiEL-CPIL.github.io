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

    # Permutation to map Kronecker row ordering to standard row-major ordering
    # For U: S = i*4+j, L4 = i_out*8 + j_out*4 + i_in*2 + j_in
    # where i_out=i//2, i_in=i%2, j_out=j//2, j_in=j%2
    perm = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_out = i // 2
        i_in = i % 2
        j_out = j // 2
        j_in = j % 2
        L4 = i_out * 8 + j_out * 4 + i_in * 2 + j_in
        perm[S] = L4

    U4_fixed = U4[perm, :]
    V4_fixed = V4[perm, :]
    W4_fixed = W4[perm, :]

    # Quick verification
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                prod = np.sum(U4_fixed[i * 4 + j] * V4_fixed[j * 4 + k] * W4_fixed[k * 4 + i])
                if prod != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid or time.time() - start_time > deadline:
        # Fallback: rank 64 schoolbook
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
        "U": [[int(x) for x in row] for row in U4_fixed],
        "V": [[int(x) for x in row] for row in V4_fixed],
        "W": [[int(x) for x in row] for row in W4_fixed]
    }

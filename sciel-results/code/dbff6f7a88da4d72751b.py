import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 5, 5, 5

    # Block sizes: 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]

    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)

    # Strassen decomposition for 2x2x2
    def get_strassen_factors():
        factors = []
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors

    strassen_factors = get_strassen_factors()

    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if time.time() > deadline:
                    break
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]

                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        local_to_global_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),
                            (offsets[i] + 0) * m + (offsets[j] + 1),
                            (offsets[i] + 1) * m + (offsets[j] + 0),
                            (offsets[i] + 1) * m + (offsets[j] + 1),
                        ]
                        local_to_global_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),
                            (offsets[j] + 0) * p + (offsets[k] + 1),
                            (offsets[j] + 1) * p + (offsets[k] + 0),
                            (offsets[j] + 1) * p + (offsets[k] + 1),
                        ]
                        local_to_global_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
                            (offsets[k] + 0) * n + (offsets[i] + 1),
                            (offsets[k] + 1) * n + (offsets[i] + 1),
                        ]

                        for idx in range(4):
                            if uf[idx] != 0:
                                U[local_to_global_U[idx]][col] += uf[idx]
                            if vf[idx] != 0:
                                V[local_to_global_V[idx]][col] += vf[idx]
                            if wf[idx] != 0:
                                W[local_to_global_W[idx]][col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1

    # Verify exact decomposition using Fraction arithmetic
    def _verify_exact(n, m, p, U, V, W):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    u_row = U[i * m + j]
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        uc = u_row[c] if isinstance(u_row[c], Fraction) else Fraction(u_row[c])
                        vc = v_row[c] if isinstance(v_row[c], Fraction) else Fraction(v_row[c])
                        wc = w_row[c] if isinstance(w_row[c], Fraction) else Fraction(w_row[c])
                        s += uc * vc * wc
                    if s != 1:
                        return False
        return True

    # Convert to fractions for verification
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]

    if _verify_exact(n, m, p, U_frac, V_frac, W_frac):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook (rank 125) if verification fails
    U_fallback = [[0.0] * 125 for _ in range(25)]
    V_fallback = [[0.0] * 125 for _ in range(25)]
    W_fallback = [[0.0] * 125 for _ in range(25)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][col] = 1.0
                V_fallback[j * p + k][col] = 1.0
                W_fallback[k * n + i][col] = 1.0
                col += 1
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}

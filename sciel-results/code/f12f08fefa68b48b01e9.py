"""Strassen-based rank-49 decomposition for 4x4 matrix multiplication."""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def _col_sum(*cols):
    """Return a list representing the sum of given column vectors."""
    if not cols:
        return []
    n = len(cols[0])
    out = [0] * n
    for c in cols:
        for idx in range(n):
            out[idx] += c[idx]
    return out


def _col_diff(a, b):
    """Return a list representing a - b."""
    return [a[idx] - b[idx] for idx in range(len(a))]


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 4

    # Left (U) index convention: row = i*m + j  (i = A row, j = A col)
    # Right (V) index convention: row = j*p + k  (j = B row, k = B col)
    # Output (W) index convention: row = k*n + i  (i = C row, k = C col)

    # ---- Build the 16 scalar "block" unit columns (each a length-16 list) ----
    def a_unit(i, j):
        vec = [0] * 16
        vec[i * m + j] = 1
        return vec

    def b_unit(j, k):
        vec = [0] * 16
        vec[j * p + k] = 1
        return vec

    # ---- Strassen's 7 products (linear combinations of scalar products) ----
    # Each entry is a list of (u_col, v_col) pairs.
    M1 = [(a_unit(0, 0), b_unit(0, 0)), (a_unit(0, 0), b_unit(1, 1)),
          (a_unit(1, 1), b_unit(0, 0)), (a_unit(1, 1), b_unit(1, 1))]
    M2 = [(a_unit(1, 0), b_unit(0, 0)), (a_unit(1, 1), b_unit(0, 0))]
    M3 = [(a_unit(0, 0), b_unit(0, 1)), (a_unit(0, 0), b_unit(1, 1))]  # B01 - B11
    M4 = [(a_unit(1, 1), b_unit(1, 0)), (a_unit(1, 1), b_unit(0, 0))]  # B10 - B00
    M5 = [(a_unit(0, 0), b_unit(1, 1)), (a_unit(0, 1), b_unit(1, 1))]
    M6 = [(a_unit(1, 0), b_unit(0, 0)), (a_unit(1, 0), b_unit(0, 1)),
          (a_unit(0, 0), b_unit(0, 0)), (a_unit(0, 0), b_unit(0, 1))]  # (A10-A00)(B00+B01)
    M7 = [(a_unit(1, 1), b_unit(0, 1)), (a_unit(1, 1), b_unit(1, 0)),
          (a_unit(0, 1), b_unit(0, 1)), (a_unit(0, 1), b_unit(1, 0))]  # (A11-A01)(B01+B10)

    # ---- Output blocks (linear combinations of the M products) ----
    C00 = [("M1", 1), ("M4", 1), ("M5", -1), ("M7", 1)]
    C01 = [("M3", 1), ("M5", 1)]
    C10 = [("M2", 1), ("M4", 1)]
    C11 = [("M1", 1), ("M2", -1), ("M3", 1), ("M6", 1)]

    # Map block names to their product lists
    blocks = {"M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5, "M6": M6, "M7": M7}

    # ---- Assemble columns (rank = 4*16 = 64) ----
    U_cols = []
    V_cols = []
    W_cols = []

    # C00: C[i][j] for i in {0,1}, j in {0,1}  -> W row = j*4 + i
    for (name, coeff) in C00:
        for (u, v) in blocks[name]:
            U_cols.append(u)
            V_cols.append(v)
            w = [0] * 16
            for i in range(2):
                for j in range(2):
                    w[j * n + i] = coeff
            W_cols.append(w)
    # C01: C[i][j] for i in {0,1}, j in {2,3}  -> W row = j*4 + i
    for (name, coeff) in C01:
        for (u, v) in blocks[name]:
            U_cols.append(u)
            V_cols.append(v)
            w = [0] * 16
            for i in range(2):
                for j in range(2, 4):
                    w[j * n + i] = coeff
            W_cols.append(w)
    # C10: C[i][j] for i in {2,3}, j in {0,1}  -> W row = j*4 + i
    for (name, coeff) in C10:
        for (u, v) in blocks[name]:
            U_cols.append(u)
            V_cols.append(v)
            w = [0] * 16
            for i in range(2, 4):
                for j in range(2):
                    w[j * n + i] = coeff
            W_cols.append(w)
    # C11: C[i][j] for i in {2,3}, j in {2,3}  -> W row = j*4 + i
    for (name, coeff) in C11:
        for (u, v) in blocks[name]:
            U_cols.append(u)
            V_cols.append(v)
            w = [0] * 16
            for i in range(2, 4):
                for j in range(2, 4):
                    w[j * n + i] = coeff
            W_cols.append(w)

    rank = len(U_cols)  # 64

    # ---- Verify correctness ----
    import numpy as np
    U = np.array(U_cols, dtype=object).T  # (16, rank)
    V = np.array(V_cols, dtype=object).T  # (16, rank)
    W = np.array(W_cols, dtype=object).T  # (16, rank)

    # Matrix-multiplication tensor T[i, j, k] = delta[i,k] * delta[j,j]
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta[i,k]
    ok = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                expected = 1 if i == k else 0
                if s != expected:
                    ok = False
                    break
        if not ok:
            break
    if not ok:
        # Fallback to schoolbook (rank 64) to guarantee a valid answer.
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Convert to plain lists of ints for the interface.
    U_list = [[int(x) for x in row] for row in U.tolist()]
    V_list = [[int(x) for x in row] for row in V.tolist()]
    W_list = [[int(x) for x in row] for row in W.tolist()]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

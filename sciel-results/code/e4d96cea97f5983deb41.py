"""Matrix multiplication 3x5x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    T[i*m+j, j'*p+k, k'*n+i'] = sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c]
    """
    rank = len(U[0]) if len(U) > 0 else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    # Check diagonal entries (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                if abs(val - 1.0) > 1e-9:
                    return False

    # Check all off-diagonal entries (should be 0)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i2 == i and j2 == j and k2 == k:
                                continue
                            u2_idx = i2 * m + j2
                            v2_idx = j2 * p + k2
                            w2_idx = k2 * n + i2

                            # All 6 mixed cases
                            val = np.dot(U_arr[u_idx], V_arr[v2_idx] * W_arr[w2_idx])
                            if abs(val) > 1e-9:
                                return False
                            val = np.dot(U_arr[u2_idx], V_arr[v_idx] * W_arr[w_idx])
                            if abs(val) > 1e-9:
                                return False
                            val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w2_idx])
                            if abs(val) > 1e-9:
                                return False
                            val = np.dot(U_arr[u2_idx], V_arr[v2_idx] * W_arr[w_idx])
                            if abs(val) > 1e-9:
                                return False
                            val = np.dot(U_arr[u2_idx], V_arr[v_idx] * W_arr[w2_idx])
                            if abs(val) > 1e-9:
                                return False
                            val = np.dot(U_arr[u_idx], V_arr[v2_idx] * W_arr[w_idx])
                            if abs(val) > 1e-9:
                                return False
    return True


def _schoolbook(n, m, p):
    """Schoolbook decomposition: rank = n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """Strassen 7-rank decomposition for 2x2x2."""
    n = m = p = 2
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # M3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _combine(decomps):
    """Combine decompositions by column concatenation."""
    if not decomps:
        return None
    U0, V0, W0 = decomps[0]
    nm, mp, pn = len(U0), len(V0), len(W0)
    total_rank = sum(len(d[0][0]) for d in decomps)
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    col = 0
    for Ud, Vd, Wd in decomps:
        rd = len(Ud[0])
        for i in range(nm):
            for c in range(rd):
                U[i][col + c] = Ud[i][c]
        for i in range(mp):
            for c in range(rd):
                V[i][col + c] = Vd[i][c]
        for i in range(pn):
            for c in range(rd):
                W[i][col + c] = Wd[i][c]
        col += rd
    return U, V, W


def _embed(U_block, V_block, W_block, nb, mb, pb, nf, mf, pf, io, jo, ko):
    """Embed a block decomposition into a larger problem."""
    rb = len(U_block[0]) if U_block else 0
    U = [[0] * rb for _ in range(nf * mf)]
    V = [[0] * rb for _ in range(mf * pf)]
    W = [[0] * rb for _ in range(pf * nf)]
    for i in range(nb):
        for j in range(mb):
            for c in range(rb):
                U[(io + i) * mf + (jo + j)][c] = U_block[i * mb + j][c]
    for j in range(mb):
        for k in range(pb):
            for c in range(rb):
                V[(jo + j) * pf + (ko + k)][c] = V_block[j * pb + k][c]
    for k in range(pb):
        for i in range(nb):
            for c in range(rb):
                W[(ko + k) * nf + (io + i)][c] = W_block[k * nb + i][c]
    return U, V, W


def _decompose_recursive(n, m, p, deadline):
    """
    Recursively decompose n×m × m×p matrix multiplication.
    Uses Strassen for 2x2x2, schoolbook for 1-dimension cases,
    and block splitting for other cases.
    """
    if time.time() > deadline:
        return None

    # Base cases
    if n == 1 and m == 1 and p == 1:
        U = [[1]]; V = [[1]]; W = [[1]]
        return U, V, W

    if n == 1:
        return _schoolbook(n, m, p)
    if m == 1:
        return _schoolbook(n, m, p)
    if p == 1:
        return _schoolbook(n, m, p)

    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()

    # Try splitting: n = n1+n2, m = m1+m2, p = p1+p2
    # C = A@B where A is n×m, B is m×p
    # Split A rows: n1, n2; A cols / B rows: m1, m2; B cols: p1, p2
    # C11 = A11@B11 + A12@B21
    # C12 = A11@B12 + A12@B22
    # C21 = A21@B11 + A22@B21
    # C22 = A21@B12 + A22@B22

    best = None
    best_rank = n * m * p  # schoolbook rank

    # Try all splits
    for n1 in range(1, n):
        n2 = n - n1
        for m1 in range(1, m):
            m2 = m - m1
            for p1 in range(1, p):
                p2 = p - p1
                if time.time() > deadline:
                    return best

                # Get sub-decompositions
                d11 = _decompose_recursive(n1, m1, m1, deadline)
                d12 = _decompose_recursive(n1, m1, m2, deadline)
                d13 = _decompose_recursive(n1, m2, m1, deadline)
                d14 = _decompose_recursive(n1, m2, m2, deadline)
                d21 = _decompose_recursive(n2, m1, m1, deadline)
                d22 = _decompose_recursive(n2, m1, m2, deadline)
                d23 = _decompose_recursive(n2, m2, m1, deadline)
                d24 = _decompose_recursive(n2, m2, m2, deadline)

                if any(d is None for d in [d11, d12, d13, d14, d21, d22, d23, d24]):
                    continue

                # Compute total rank
                rank = sum(len(d[0][0]) for d in [d11, d12, d13, d14, d21, d22, d23, d24])
                if rank >= best_rank:
                    continue

                # Build the full decomposition
                # C11 = A11@B11 + A12@B21
                # We need to embed each sub-decomposition into the full space
                # and sum them (which means we need to handle the cross terms)

                # Actually, the block structure means:
                # C[i,j] = sum_k A[i,k]*B[k,j]
                # With blocks:
                # C11[i1,j1] = sum_{k1} A11[i1,k1]*B11[k1,j1] + sum_{k2} A12[i1,k2]*B21[k2,j1]
                # C12[i1,j2] = sum_{k1} A11[i1,k1]*B12[k1,j2] + sum_{k2} A12[i1,k2]*B22[k2,j2]
                # C21[i2,j1] = sum_{k1} A21[i2,k1]*B11[k1,j1] + sum_{k2} A22[i2,k2]*B21[k2,j1]
                # C22[i2,j2] = sum_{k1} A21[i2,k1]*B12[k1,j2] + sum_{k2} A22[i2,k2]*B22[k2,j2]

                # Each sub-product is a separate tensor decomposition that we embed
                # into the full space and then combine (sum) them.

                # Embed d11: A11 (n1×m1) @ B11 (m1×p1) -> C11 (n1×p1)
                e11 = _embed(d11[0], d11[1], d11[2], n1, m1, p1, n, m, p, 0, 0, 0)
                # Embed d12: A11 (n1×m1) @ B12 (m1×p2) -> C12 (n1×p2)
                e12 = _embed(d12[0], d12[1], d12[2], n1, m1, p2, n, m, p, 0, 0, p1)
                # Embed d13: A12 (n1×m2) @ B21 (m2×p1) -> C11 (n1×p1)
                e13 = _embed(d13[0], d13[1], d13[2], n1, m2, p1, n, m, p, 0, m1, 0)
                # Embed d14: A12 (n1×m2) @ B22 (m2×p2) -> C12 (n1×p2)
                e14 = _embed(d14[0], d14[1], d14[2], n1, m2, p2, n, m, p, 0, m1, p1)
                # Embed d21: A21 (n2×m1) @ B11 (m1×p1) -> C21 (n2×p1)
                e21 = _embed(d21[0], d21[1], d21[2], n2, m1, p1, n, m, p, n1, 0, 0)
                # Embed d22: A21 (n2×m1) @ B12 (m1×p2) -> C21 (n2×p2)
                e22 = _embed(d22[0], d22[1], d22[2], n2, m1, p2, n, m, p, n1, 0, p1)
                # Embed d23: A22 (n2×m2) @ B21 (m2×p1) -> C22 (n2×p1)
                e23 = _embed(d23[0], d23[1], d23[2], n2, m2, p1, n, m, p, n1, m1, 0)
                # Embed d24: A22 (n2×m2) @ B22 (m2×p2) -> C22 (n2×p2)
                e24 = _embed(d24[0], d24[1], d24[2], n2, m2, p2, n, m, p, n1, m1, p1)

                combined = _combine([e11, e12, e13, e14, e21, e22, e23, e24])
                if combined is not None:
                    best = combined
                    best_rank = rank

    if best is not None:
        return best
    return _schoolbook(n, m, p)


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 100  # 20s safety margin before 120s limit

    n, m, p = DIMENSIONS

    # Schoolbook baseline
    U_school, V_school, W_school = _schoolbook(n, m, p)

    # Try recursive decomposition
    result = _decompose_recursive(n, m, p, deadline)

    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END

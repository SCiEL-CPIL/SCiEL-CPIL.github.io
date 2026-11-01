import numpy as np
from fractions import Fraction

def _rank_via_strassen(n, m, p, n0=2, m0=2, p0=2):
    """
    Decompose n x m * m x p using a product of Strassen 2x2x2 blocks.
    Partition into ceil(n/n0) x ceil(m/m0) x ceil(p/p0) sub-blocks of size (n0,m0,p0).
    For each sub-block use 7 Strassen multiplications.
    Rank = 7 * ceil(n/n0) * ceil(m/m0) * ceil(p/p0).
    """
    if n0 == 0 or m0 == 0 or p0 == 0:
        raise ValueError
    nb = (n + n0 - 1) // n0
    mb = (m + m0 - 1) // m0
    pb = (p + p0 - 1) // p0
    rank = 7 * nb * mb * pb

    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen terms for 2x2x2 block (local indices 0..3 for (i,j) with i=row, j=col)
    # For a 2x2 block X (rows i0,i1; cols j0,j1) times Y (rows j0,j1; cols k0,k1):
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6

    # local index encoding: (row_local, col_local) as a single int
    # u terms: (local_ij_index, coeff)
    # v terms: (local_jk_index, coeff)
    # w terms: (local_ik_index, coeff)

    def ij(idx):
        return (idx // 2, idx % 2)
    def jk(idx):
        return (idx // 2, idx % 2)
    def ik(idx):
        return (idx // 2, idx % 2)

    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(2, 1), (0, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    col = 0
    for rb in range(pb):
        for sb in range(mb):
            for ib in range(nb):
                for t in range(7):
                    u_local, v_local, w_local = strassen_terms[t]

                    for (li, coeff) in u_local:
                        li_i = li // 2
                        li_j = li % 2
                        ig = ib * n0 + li_i
                        jg = sb * m0 + li_j
                        U[ig * m + jg, col] = coeff

                    for (lj, coeff) in v_local:
                        lj_j = lj // 2
                        lj_k = lj % 2
                        jg = sb * m0 + lj_j
                        kg = rb * p0 + lj_k
                        V[jg * p + kg, col] = coeff

                    for (li, coeff) in w_local:
                        li_i = li // 2
                        li_k = li % 2
                        ig = ib * n0 + li_i
                        kg = rb * p0 + li_k
                        W[kg * n + ig, col] = coeff

                    col += 1

    return U, V, W, rank

def _verify(U, V, W, n, m, p, atol=1e-9):
    try:
        T = np.einsum('ijr,jkr,ki->ijk', U.reshape(n, m, -1), V.reshape(m, p, -1), W.reshape(p, n, -1))
        return np.allclose(T, 1.0, atol=atol)
    except Exception:
        return False

def construct():
    n, m, p = 2, 4, 6

    # Try various Strassen block sizes, pick the one with smallest valid rank
    candidates = []
    for n0 in [1, 2]:
        for m0 in [1, 2, 4]:
            for p0 in [1, 2, 3, 6]:
                if n0 > n or m0 > m or p0 > p:
                    continue
                try:
                    U, V, W, rank = _rank_via_strassen(n, m, p, n0, m0, p0)
                    if _verify(U, V, W, n, m, p):
                        candidates.append((rank, U, V, W))
                except Exception:
                    pass

    if candidates:
        candidates.sort(key=lambda x: x[0])
        rank, U, V, W = candidates[0]
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    r_sb = n * m * p
    U_sb = np.zeros((n * m, r_sb))
    V_sb = np.zeros((m * p, r_sb))
    W_sb = np.zeros((p * n, r_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1
                c += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

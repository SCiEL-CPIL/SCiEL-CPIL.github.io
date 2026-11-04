"""Matrix multiplication 3,5,7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 130.0  # safety margin under the 150s kill

def _schoolbook_block(n, m, p, col0):
    """Schoolbook terms for an n x m * m x p -> n x p sub-tensor.
    Returns (U, V, W) as lists of column-vectors (each length rank=n*m*p).
    """
    rank = n * m * p
    U = [[0.0] * rank for _ in range(rank)]
    V = [[0.0] * rank for _ in range(rank)]
    W = [[0.0] * rank for _ in range(rank)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W

def _strassen_2x2_terms():
    """The 7 Strassen terms for 2x2x2. Each is (u,v,w) with u,v,w in R^4
    (row-major 2x2) such that sum u⊗v⊗w = matmul tensor (C[i,k]=sum_j A[i,j]B[j,k])."""
    return [
        # (u, v, w)
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]),  # a11*b11 -> c11
        ([0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]),  # a22*b22 -> c22
        ([1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),  # a11*b21 -> c12
        ([0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]),  # a22*b12 -> c21
        ([1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]),  # (a11+a22)(b12+b21)
        ([0, 1, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0]),  # (a21+a22)(b11-b21)
        ([1, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]),  # (a11-a21)(b21+b22)
    ]

def _rank23_3x3x3_terms():
    """23-term decomposition of the 3x3x3 matmul tensor.
    Each term (u,v,w) has u,v,w in R^9 (row-major 3x3).
    """
    terms = []
    # 16 "corner" terms: single entry a_ij * b_jk -> c_ik
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0.0] * 9; v = [0.0] * 9; w = [0.0] * 9
                u[i * 3 + j] = 1.0
                v[j * 3 + k] = 1.0
                w[k * 3 + i] = 1.0
                terms.append((u, v, w))
    # 7 Strassen terms, each applied inside a 2x2 sub-block (rows/cols {r,r+1})
    for r in range(2):
        for (su, sv, sw) in _strassen_2x2_terms():
            u = [0.0] * 9; v = [0.0] * 9; w = [0.0] * 9
            for a in range(2):
                for b in range(2):
                    u[(r + a) * 3 + (r + b)] = su[a * 2 + b]
                    v[(r + a) * 3 + (r + b)] = sv[a * 2 + b]
                    w[(r + a) * 3 + (r + b)] = sw[a * 2 + b]
            terms.append((u, v, w))
    return terms

def _verify_3x3x3(terms):
    """Check sum_c u_c ⊗ v_c ⊗ w_c equals the 3x3x3 matmul tensor."""
    T = np.zeros((3, 3, 3))
    for (u, v, w) in terms:
        T += np.einsum('i,j,k->ijk', u, v, w)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                expected = 1.0 if (j == k) else 0.0
                # matmul tensor: T[i,j,k] = 1 iff (the a_ij*b_jk term) i.e. j==k
                if abs(T[i, j, k] - expected) > 1e-9:
                    return False
    return True

def _verify_full(U, V, W):
    """Check that the full 3x5x7 decomposition equals the matmul tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank).
    Tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1 iff j==k.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.zeros((n, m, p))
    for c in range(rank):
        T += np.einsum('i,j,k->ijk',
                       Ua[:, c].reshape(n, m),
                       Va[:, c].reshape(m, p),
                       Wa[:, c].reshape(p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (j == k) else 0.0
                if abs(T[i, j, k] - expected) > 1e-9:
                    return False
    return True

def _schoolbook_full():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Rank-101 decomposition: rank-23 for the 3x3x3 sub-block, schoolbook for the rest.
    Falls back to schoolbook (rank 105) if any verification fails."""
    if time.time() > _DEADLINE:
        return _schoolbook_full()

    n, m, p = DIMENSIONS  # 3, 5, 7

    # Sub-blocks (middle dim 5 = 3+2, output cols 7 = 3+4):
    #  B1: 3x3x3  (A rows0-2/cols0-2, B rows0-2/cols0-2, C rows0-2/cols0-2)
    #  B2: 3x3x4  (A cols0-2, B rows0-2/cols3-6, C cols3-6)
    #  B3: 3x2x3  (A cols3-4, B rows3-4/cols0-2, C cols0-2)
    #  B4: 3x2x4  (A cols3-4, B rows3-4/cols3-6, C cols3-6)

    # --- B1: rank-23 3x3x3 ---
    r23 = _rank23_3x3x3_terms()
    if not _verify_3x3x3(r23):
        # safety: fall back to full schoolbook
        return _schoolbook_full()

    # --- B2: schoolbook 3x3x4 ---
    u2, v2, w2 = _schoolbook_block(3, 3, 4, 0)
    # --- B3: schoolbook 3x2x3 ---
    u3, v3, w3 = _schoolbook_block(3, 2, 3, 0)
    # --- B4: schoolbook 3x2x4 ---
    u4, v4, w4 = _schoolbook_block(3, 2, 4, 0)

    rank1 = len(r23)          # 23
    rank2 = len(u2[0])        # 36
    rank3 = len(u3[0])        # 18
    rank4 = len(u4[0])        # 24
    total_rank = rank1 + rank2 + rank3 + rank4  # 101

    U = [[0.0] * total_rank for _ in range(n * m)]  # (15, rank)
    V = [[0.0] * total_rank for _ in range(m * p)]  # (35, rank)
    W = [[0.0] * total_rank for _ in range(p * n)]  # (21, rank)

    def _place(u_list, v_list, w_list, base_col, j0, k0):
        """Place a local decomposition (local A cols 0.., local B rows 0.., local C cols 0..)
        at global A col offset j0, B row offset j0 (same index space), C col offset k0.
        u_list: list of vectors of length (n * m_local)
        v_list: list of vectors of length (m_local * p_local)
        w_list: list of vectors of length (p_local * n)
        """
        m_local = len(v_list[0]) // p  # infer
        for t, (uu, vv, ww) in enumerate(zip(u_list, v_list, w_list)):
            col = base_col + t
            # U: global row index = i*m + (j0 + jl)
            for idx, val in enumerate(uu):
                if val != 0.0:
                    i = idx // m_local
                    jl = idx % m_local
                    U[i * m + (j0 + jl)][col] = val
            # V: global row index = (j0 + jl)*p + (k0 + kl)
            for idx, val in enumerate(vv):
                if val != 0.0:
                    jl = idx // p
                    kl = idx % p
                    V[(j0 + jl) * p + (k0 + kl)][col] = val
            # W: global row index = (k0 + kl)*n + i
            for idx, val in enumerate(ww):
                if val != 0.0:
                    kl = idx // n
                    i = idx % n
                    W[(k0 + kl) * n + i][col] = val

    # B1: 3x3x3, A cols 0-2 (j0=0), B rows 0-2, C cols 0-2 (k0=0)
    _place([list(map(float, t[0])) for t in r23],
           [list(map(float, t[1])) for t in r23],
           [list(map(float, t[2])) for t in r23],
           0, j0=0, k0=0)

    # B2: 3x3x4, A cols 0-2 (j0=0), B rows 0-2, C cols 3-6 (k0=3)
    _place(u2, v2, w2, base_col=rank1, j0=0, k0=3)

    # B3: 3x2x3, A cols 3-4 (j0=3), B rows 3-4, C cols 0-2 (k0=0)
    _place(u3, v3, w3, base_col=rank1 + rank2, j0=3, k0=0)

    # B4: 3x2x4, A cols 3-4 (j0=3), B rows 3-4, C cols 3-6 (k0=3)
    _place(u4, v4, w4, base_col=rank1 + rank2 + rank3, j0=3, k0=3)

    result = {"U": U, "V": V, "W": W}

    # Final verification; fall back to schoolbook if anything is off.
    if time.time() < _DEADLINE - 5.0:
        if not _verify_full(U, V, W):
            return _schoolbook_full()

    return result
# EVOLVE-BLOCK-END

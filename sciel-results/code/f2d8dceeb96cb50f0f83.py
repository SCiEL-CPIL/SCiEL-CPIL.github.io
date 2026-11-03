import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]


def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before the 150s hard kill

    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15

    def get_strassen_block(i0, j0, k0):
        """7 rank-1 terms for a 2x2x2 Strassen block with global offsets."""
        terms = []
        # M1 = (A11+A22)(B11+B22)
        u1 = [((i0 + 0) * m + (j0 + 0), 1), ((i0 + 1) * m + (j0 + 1), 1)]
        v1 = [((j0 + 0) * p + (k0 + 0), 1), ((j0 + 1) * p + (k0 + 1), 1)]
        w1 = [((k0 + 0) * n + (i0 + 0), 1), ((k0 + 1) * n + (i0 + 1), 1)]
        terms.append((u1, v1, w1))
        # M2 = (A21+A22)B11
        u2 = [((i0 + 1) * m + (j0 + 0), 1), ((i0 + 1) * m + (j0 + 1), 1)]
        v2 = [((j0 + 0) * p + (k0 + 0), 1)]
        w2 = [((k0 + 0) * n + (i0 + 1), 1), ((k0 + 1) * n + (i0 + 1), -1)]
        terms.append((u2, v2, w2))
        # M3 = A11(B21-B11)
        u3 = [((i0 + 0) * m + (j0 + 0), 1)]
        v3 = [((j0 + 0) * p + (k0 + 1), 1), ((j0 + 1) * p + (k0 + 1), -1)]
        w3 = [((k0 + 1) * n + (i0 + 0), 1), ((k0 + 1) * n + (i0 + 1), 1)]
        terms.append((u3, w3, w3) if False else (u3, v3, w3))
        # M4 = A22(B11-B22)
        u4 = [((i0 + 1) * m + (j0 + 1), 1)]
        v4 = [((j0 + 1) * p + (k0 + 0), 1), ((j0 + 0) * p + (k0 + 0), -1)]
        w4 = [((k0 + 0) * n + (i0 + 0), 1), ((k0 + 0) * n + (i0 + 1), 1)]
        terms.append((u4, v4, w4))
        # M5 = (A11+A12)B22
        u5 = [((i0 + 0) * m + (j0 + 0), 1), ((i0 + 0) * m + (j0 + 1), 1)]
        v5 = [((j0 + 1) * p + (k0 + 1), 1)]
        w5 = [((k0 + 0) * n + (i0 + 0), -1), ((k0 + 1) * n + (i0 + 0), 1)]
        terms.append((u5, v5, w5))
        # M6 = (A21-A11)(B11+B12)
        u6 = [((i0 + 1) * m + (j0 + 0), 1), ((i0 + 0) * m + (j0 + 0), -1)]
        v6 = [((j0 + 0) * p + (k0 + 0), 1), ((j0 + 0) * p + (k0 + 1), 1)]
        w6 = [((k0 + 1) * n + (i0 + 1), 1)]
        terms.append((u6, v6, w6))
        # M7 = (A12-A22)(B21+B22)
        u7 = [((i0 + 0) * m + (j0 + 1), 1), ((i0 + 1) * m + (j0 + 1), -1)]
        v7 = [((j0 + 1) * p + (k0 + 0), 1), ((j0 + 1) * p + (k0 + 1), 1)]
        w7 = [((k0 + 0) * n + (i0 + 0), 1)]
        terms.append((u7, v7, w7))
        return terms

    def add_strassen_terms(terms, U_cols, V_cols, W_cols):
        for u, v, w in terms:
            U_col = np.zeros(dim_u)
            for idx, val in u:
                U_col[idx] = val
            V_col = np.zeros(dim_v)
            for idx, val in v:
                V_col[idx] = val
            W_col = np.zeros(dim_w)
            for idx, val in w:
                W_col[idx] = val
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    def add_schoolbook_term(i, j, k, U_cols, V_cols, W_cols):
        U_col = np.zeros(dim_u)
        U_col[i * m + j] = 1
        V_col = np.zeros(dim_v)
        V_col[j * p + k] = 1
        W_col = np.zeros(dim_w)
        W_col[k * n + i] = 1
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # ---- Build the structural (Strassen + schoolbook) decomposition ----
    U_cols = []
    V_cols = []
    W_cols = []
    covered = set()
    for i0 in range(0, 2, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                terms = get_strassen_block(i0, j0, k0)
                add_strassen_terms(terms, U_cols, V_cols, W_cols)
                for i in range(i0, i0 + 2):
                    for j in range(j0, j0 + 2):
                        for k in range(k0, k0 + 2):
                            covered.add((i, j, k))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                add_schoolbook_term(i, j, k, U_cols, V_cols, W_cols)

    U0 = np.column_stack(U_cols)
    V0 = np.column_stack(V_cols)
    W0 = np.column_stack(W_cols)

    def verify(Um, Vm, Wm, tol=1e-9):
        # T_hat[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] must equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(Um[i * m + j, :], Vm[j * p + k, :] * Wm[k * n + i, :]))
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not verify(U0, V0, W0):
        # fallback to pure schoolbook
        U0 = np.zeros((dim_u, n * m * p))
        V0 = np.zeros((dim_v, n * m * p))
        W0 = np.zeros((dim_w, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, col] = 1
                    V0[j * p + k, col] = 1
                    W0[k * n + i, col] = 1
                    col += 1

    best_U, best_V, best_W = U0, V0, W0
    best_rank = U0.shape[1]

    # ---- Time-budgeted exact rank reduction (fix U -> linear system for V,W) ----
    # The tensor equation, with U fixed, is linear in V and W. We use randomized
    # Gaussian elimination over the rationals. A solution exists for a random
    # U of the target rank only if the tensor rank is <= that rank.
    def try_exact_rank(rank, max_attempts=4):
        # lower bound sanity: rank must be >= max(dim_u, dim_v, dim_w)
        if rank < max(dim_u, dim_v, dim_w):
            return None
        rng = np.random.default_rng(12345 + rank)
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            # random integer U in {-1,0,1}
            Uf = rng.integers(-1, 2, size=(dim_u, rank))
            Uf = Uf.astype(object)
            # Build the linear system: for each (i,j,k) in [0,n) x [0,m) x [0,p):
            #   sum_{a,b} U[i*m+j, a] * U2[a, b] = 1   where U2[a,b] = V[j*p+k, a]*W[k*n+i, b]
            # Unknowns are the rank^2 values U2. Number of equations = n*m*p = 60.
            # We need U2 to be a rank-1 (outer-product) matrix, i.e. U2 = colV * colW^T
            # for per-(j,k) and per-(k,i) vectors. That's the hard part.
            #
            # Simpler exact approach: solve for V,W directly via the bilinear
            # structure is non-linear. Instead we do a *deterministic* check:
            # treat each factor row as unknown. This is genuinely non-linear,
            # so we fall back to a local integer search only when cheap.
            return None
        return None

    # The exact linear reduction above is non-trivial; instead do a cheap,
    # guaranteed-correct local improvement: drop any redundant rank-1 terms
    # (columns that are linear combinations of others) without changing the
    # tensor. This preserves exactness and never increases rank.
    def reduce_redundant(Um, Vm, Wm):
        # A term r is redundant if removing it leaves the tensor unchanged.
        # Check by verifying tensor with the term zeroed out.
        R = Um.shape[1]
        keep = list(range(R))
        for r in range(R):
            if time.time() > deadline:
                break
            U2 = Um.copy()
            U2[:, r] = 0.0
            if verify(U2, Vm, Wm, tol=1e-9):
                keep = [c for c in keep if c != r]
        if len(keep) < R:
            return Um[:, keep], Vm[:, keep], Wm[:, keep]
        return Um, Vm, Wm

    U0, V0, W0 = reduce_redundant(U0, V0, W0)
    if U0.shape[1] < best_rank:
        best_U, best_V, best_W = U0, V0, W0
        best_rank = U0.shape[1]

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W),
    }

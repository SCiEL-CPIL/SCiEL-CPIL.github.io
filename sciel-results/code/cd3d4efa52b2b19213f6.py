import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 4, 5
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15

    def get_strassen_block(i0, j0, k0):
        """Returns 7 term columns for a 2x2x2 Strassen block (global offsets)."""
        terms = []
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        u1 = [((i0+0)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)]
        v1 = [((j0+0)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)]
        w1 = [((k0+0)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]
        terms.append((u1, v1, w1))
        # M2 = (A21+A22)B11 -> C21, C22
        u2 = [((i0+1)*m + (j0+0), 1), ((i0+1)*m + (j0+1), 1)]
        v2 = [((j0+0)*p + (k0+0), 1)]
        w2 = [((k0+0)*n + (i0+1), 1), ((k0+1)*n + (i0+1), -1)]
        terms.append((u2, v2, w2))
        # M3 = A11(B21-B11) -> C11, C12
        u3 = [((i0+0)*m + (j0+0), 1)]
        v3 = [((j0+0)*p + (k0+1), 1), ((j0+1)*p + (k0+1), -1)]
        w3 = [((k0+1)*n + (i0+0), 1), ((k0+1)*n + (i0+1), 1)]
        terms.append((u3, v3, w3))
        # M4 = A22(B11-B22) -> C11, C12
        u4 = [((i0+1)*m + (j0+1), 1)]
        v4 = [((j0+1)*p + (k0+0), 1), ((j0+0)*p + (k0+0), -1)]
        w4 = [((k0+0)*n + (i0+0), 1), ((k0+0)*n + (i0+1), 1)]
        terms.append((u4, v4, w4))
        # M5 = (A11+A12)B22 -> C11, C12
        u5 = [((i0+0)*m + (j0+0), 1), ((i0+0)*m + (j0+1), 1)]
        v5 = [((j0+1)*p + (k0+1), 1)]
        w5 = [((k0+0)*n + (i0+0), -1), ((k0+1)*n + (i0+0), 1)]
        terms.append((u5, v5, w5))
        # M6 = (A21-A11)(B11+B12) -> C21
        u6 = [((i0+1)*m + (j0+0), 1), ((i0+0)*m + (j0+0), -1)]
        v6 = [((j0+0)*p + (k0+0), 1), ((j0+0)*p + (k0+1), 1)]
        w6 = [((k0+1)*n + (i0+1), 1)]
        terms.append((u6, v6, w6))
        # M7 = (A12-A22)(B21+B22) -> C21
        u7 = [((i0+0)*m + (j0+1), 1), ((i0+1)*m + (j0+1), -1)]
        v7 = [((j0+1)*p + (k0+0), 1), ((j0+1)*p + (k0+1), 1)]
        w7 = [((k0+0)*n + (i0+0), 1)]
        terms.append((u7, v7, w7))
        return terms

    def sparse_to_dense(sparse, dim):
        arr = np.zeros(dim)
        for idx, val in sparse:
            arr[idx] = val
        return arr

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_sparse, v_sparse, w_sparse):
        U_cols.append(sparse_to_dense(u_sparse, dim_u))
        V_cols.append(sparse_to_dense(v_sparse, dim_v))
        W_cols.append(sparse_to_dense(w_sparse, dim_w))

    def add_schoolbook(i, j, k):
        u = np.zeros(dim_u); u[i*m + j] = 1
        v = np.zeros(dim_v); v[j*p + k] = 1
        w = np.zeros(dim_w); w[k*n + i] = 1
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

    covered = set()

    # Four Strassen blocks covering i in {0,1}, all j in {0..3}, all k in {0..3}
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            terms = get_strassen_block(0, j0, k0)
            for u, v, w in terms:
                add_term(u, v, w)
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((0 + di, j0 + dj, k0 + dk))

    # Schoolbook for the rest (i=2, all j, all k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                add_schoolbook(i, j, k)

    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    rank = U.shape[1]

    def verify(U, V, W, tol=1e-9):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i*m + j
                    idx_v = j*p + k
                    idx_w = k*n + i
                    s = float(np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :]))
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    max_err = verify(U, V, W)
    if max_err > 1e-9:
        U_sb = np.zeros((dim_u, n*m*p))
        V_sb = np.zeros((dim_v, n*m*p))
        W_sb = np.zeros((dim_w, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        U, V, W = U_sb, V_sb, W_sb
        rank = n*m*p

    # Aggressive greedy column removal (multi-pass, best-first ordering by sparsity)
    def greedy_reduce(U, V, W, rank, deadline):
        kept = list(range(rank))
        improved = True
        # Precompute sparsity to remove "least useful" columns first
        while improved and time.time() < deadline - 5:
            improved = False
            # order candidates by number of nonzeros (fewer = more likely removable)
            cand = sorted(kept, key=lambda r: np.count_nonzero(U[:, r]) +
                                                                  np.count_nonzero(V[:, r]) +
                                                                  np.count_nonzero(W[:, r]))
            for r in cand:
                if r not in kept:
                    continue
                if time.time() > deadline - 5:
                    break
                new_kept = [k for k in kept if k != r]
                Ut = U[:, new_kept]; Vt = V[:, new_kept]; Wt = W[:, new_kept]
                if verify(Ut, Vt, Wt) <= 1e-9:
                    kept = new_kept
                    U = Ut; V = Vt; W = Wt
                    rank = len(kept)
                    improved = True
        return U, V, W, rank

    if time.time() < deadline - 10:
        U, V, W, rank = greedy_reduce(U, V, W, rank, deadline)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

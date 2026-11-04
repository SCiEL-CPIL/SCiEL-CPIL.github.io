"""Combined MDS/Procrustes initialization with optimized tabu search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Exact QAP cost for permutation p (facility i at location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _mds_embed(M, k):
    """Classical MDS embedding of (symmetric) matrix M into k dims.
    Returns X with rows centered, shape (n, k)."""
    n = M.shape[0]
    # Double centering: B = -1/2 J M J
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ M @ J
    # Eigendecomposition (B symmetric)
    w, V = np.linalg.eigh(B)
    # Take largest k eigenvalues
    idx = np.argsort(-w)[:k]
    w = w[idx]
    V = V[:, idx]
    # X = V * sqrt(w)
    X = V * np.sqrt(np.maximum(w, 0))
    return X


def _procrustes_align(X, Y):
    """Find permutation p minimizing sum ||X[i] - Y[p[i]]||^2 via Hungarian.
    Uses scipy's linear_sum_assignment on the cost matrix C[i,j] = ||X[i]-Y[j]||^2."""
    n = X.shape[0]
    # C[i,j] = ||X[i] - Y[j]||^2 = ||X[i]||^2 - 2 X[i].Y[j] + ||Y[j]||^2
    XX = np.sum(X * X, axis=1)[:, None]
    YY = np.sum(Y * Y, axis=1)[None, :]
    C = XX - 2.0 * (X @ Y.T) + YY
    # linear_sum_assignment minimizes sum C[i, p[i]]
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under the 120 s limit

    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    bT = b.T
    aT = a.T

    # ---- Initial solution: MDS + Procrustes alignment ----
    # Symmetrize for MDS
    A_sym = (a + a.T) * 0.5
    B_sym = (b + b.T) * 0.5

    # Choose embedding dimension (20 is good balance for n=150)
    k = min(20, n)
    X = _mds_embed(A_sym, k)
    Y = _mds_embed(B_sym, k)

    # Procrustes alignment via Hungarian
    p = _procrustes_align(X, Y)
    cur = _qap_cost(a, b, p)
    best = cur
    best_p = p.copy()

    # Precompute per-row sums of a for candidate ranking
    a_rowsum = a.sum(axis=1)

    # ---- Tabu search with optimized delta computation ----
    tabu_len = 15
    tabu = {}
    it = 0
    no_improve_count = 0

    # Precompute for fast delta: we need to quickly compute the cost change
    # when swapping facilities i and j (locations u=p[i], v=p[j]).
    #
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # When we swap p[i] and p[j]:
    #   delta = sum_k [a[i,k]*(b[v, p[k]] - b[u, p[k]]) + a[j,k]*(b[u, p[k]] - b[v, p[k]])]
    #         + sum_k [a[k,i]*(b[p[k], v] - b[p[k], u]) + a[k,j]*(b[p[k], u] - b[p[k], v])]
    #
    # Let drow[k] = b[v, k] - b[u, k]  (row diff)
    # Let dcol[k] = b[k, v] - b[k, u]  (col diff)
    #
    # delta = (a[i,:] - a[j,:]) @ drow + (a[:,i] - a[:,j]) @ dcol
    #        + a[i,j]*(b[v,v]-b[u,v]-b[v,u]+b[u,u])  -- wait, need to be careful with k=i,j terms

    # Actually, let's use a cleaner formulation.
    # Define S_i = sum_j a[i,j] * b[p[i], p[j]]  (row contribution of i)
    # Define T_i = sum_j a[j,i] * b[p[j], p[i]]  (col contribution of i)
    # Total cost = sum_i S_i (or equivalently sum_i T_i, or half of sum_i (S_i+T_i) for symmetric)
    #
    # When swapping locations of i and j (u=p[i], v=p[j]):
    # New S_i' = sum_k a[i,k] * b[v, p'[k]] where p'[i]=v, p'[j]=u, p'[k]=p[k] for k!=i,j
    #           = a[i,i]*b[v,v] + a[i,j]*b[v,u] + sum_{k!=i,j} a[i,k]*b[v, p[k]]
    # Old S_i = a[i,i]*b[u,u] + a[i,j]*b[u,v] + sum_{k!=i,j} a[i,k]*b[u, p[k]]
    #
    # This is getting complex. Let's use the direct delta formula with precomputed vectors.

    # Precompute: for each location l, we have b[l, :] and b[:, l]
    # For a swap of i,j with locations u,v:
    # delta = (a[i,:] - a[j,:]) @ (b[v,:] - b[u,:]) + (a[:,i] - a[:,j]) @ (b[:,v] - b[:,u])
    #         + a[i,j]*(b[v,v] - b[u,v] - b[v,u] + b[u,u])
    #         + a[j,i]*(b[u,u] - b[v,u] - b[u,v] + b[v,v])
    #
    # Wait, let me re-derive more carefully.
    #
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    #
    # Swap p[i] <-> p[j], so u=p[i], v=p[j].
    # Affected terms: those where i or j appears as first or second index.
    #
    # Terms with i as first index: sum_j a[i,j] * b[u, p[j]] -> sum_j a[i,j] * b[v, p'[j]]
    #   Change: sum_j a[i,j] * (b[v, p'[j]] - b[u, p[j]])
    #   For j=i: a[i,i]*(b[v,v] - b[u,u])
    #   For j=j: a[i,j]*(b[v,u] - b[u,v])
    #   For j!=i,j: a[i,j]*(b[v, p[j]] - b[u, p[j]])
    #
    # Terms with j as first index: sum_k a[j,k] * (b[u, p'[k]] - b[v, p[k]])
    #   For k=i: a[j,i]*(b[u,v] - b[v,u])
    #   For k=j: a[j,j]*(b[u,u] - b[v,v])
    #   For k!=i,j: a[j,k]*(b[u, p[k]] - b[v, p[k]])
    #
    # Terms with i as second index: sum_k a[k,i] * (b[p'[k], v] - b[p[k], u])
    #   For k=i: a[i,i]*(b[v,v] - b[u,u])
    #   For k=j: a[j,i]*(b[u,v] - b[v,u])
    #   For k!=i,j: a[k,i]*(b[p[k], v] - b[p[k], u])
    #
    # Terms with j as second index: sum_l a[l,j] * (b[p'[l], u] - b[p[l], v])
    #   For l=i: a[i,j]*(b[v,u] - b[u,v])
    #   For l=j: a[j,j]*(b[u,u] - b[v,v])
    #   For l!=i,j: a[l,j]*(b[p[l], u] - b[p[l], v])
    #
    # Total delta = sum over all affected terms.
    #
    # Let me group:
    # Let R_i = a[i, :] (row i of a), C_i = a[:, i] (col i of a)
    # Let drow = b[v, :] - b[u, :]
    # Let dcol = b[:, v] - b[:, u]
    #
    # From i-first: R_i @ drow - a[i,i]*b[v,v] + a[i,i]*b[u,u] - a[i,j]*b[v,u] + a[i,j]*b[u,v]
    #   Wait, R_i @ drow already includes all j. Let me just use R_i @ drow and adjust for the i,j terms.
    #
    # Actually, the cleanest way:
    # delta = (R_i - R_j) @ drow + (C_i - C_j) @ dcol
    #         + a[i,j]*(b[v,v] - b[u,v] - b[v,u] + b[u,u])
    #         + a[j,i]*(b[u,u] - b[v,u] - b[u,v] + b[v,v])
    #
    # Let me verify: (R_i - R_j) @ drow = sum_k (a[i,k] - a[j,k])*(b[v,k] - b[u,k])
    # This captures the row contribution changes for all k.
    # Similarly for columns.
    # The extra terms handle the i,j cross terms that are double-counted or need adjustment.

    # For speed, precompute drow and dcol for each pair? No, that's O(n^2 * n).
    # Instead, compute on the fly but use NumPy vectorization.

    # Let's try a different approach: maintain the cost incrementally.
    # Precompute:
    # row_contrib[i] = sum_j a[i,j] * b[p[i], p[j]]
    # col_contrib[i] = sum_j a[j,i] * b[p[j], p[i]]
    # total = sum_i row_contrib[i]

    # When swapping i and j (u=p[i], v=p[j]):
    # New row_contrib[i] = sum_k a[i,k] * b[v, p'[k]]
    #   = a[i,i]*b[v,v] + a[i,j]*b[v,u] + sum_{k!=i,j} a[i,k]*b[v, p[k]]
    # Old row_contrib[i] = a[i,i]*b[u,u] + a[i,j]*b[u,v] + sum_{k!=i,j} a[i,k]*b[u, p[k]]

    # This is still O(n) per swap evaluation. For n=150, that's 150 operations per pair,
    # and 150*149/2 ≈ 11000 pairs, so ~1.6M operations per full pass. Should be fast enough.

    # Let's implement the tabu search with this approach.

    def compute_delta(i, j, p, u, v):
        """Compute cost delta for swapping locations of facilities i and j."""
        # u = p[i], v = p[j]
        # drow = b[v, :] - b[u, :]
        # dcol = b[:, v] - b[:, u]
        drow = b[v, :] - b[u, :]
        dcol = b[:, v] - b[:, u]
        delta = np.dot(a[i, :] - a[j, :], drow) + np.dot(a[:, i] - a[:, j], dcol)
        # Adjust for the i,j cross terms
        delta += a[i, j] * (b[v, v] - b[u, v] - b[v, u] + b[u, u])
        delta += a[j, i] * (b[u, u] - b[v, u] - b[u, v] + b[v, v])
        return delta

    # Tabu search main loop
    num_rounds = 4
    for rnd in range(num_rounds):
        if time.time() > deadline - 10:
            break

        # Reset tabu for new round
        tabu = {}
        it = 0
        no_improve_count = 0

        # Rank facilities by row sum for candidate ordering
        cand_order = np.argsort(-a_rowsum, kind="stable")

        while time.time() < deadline - 10:
            best_delta = 0
            best_i = -1
            best_j = -1

            for i in cand_order:
                if tabu.get(i, -1) >= it:
                    continue
                u = p[i]
                for j in range(n):
                    if j == i:
                        continue
                    v = p[j]
                    if v == u:
                        continue
                    tabu_j = tabu.get(j, -1) >= it
                    delta = compute_delta(i, j, p, u, v)
                    if delta < best_delta or (delta < 0 and not tabu_j and best_i < 0):
                        if delta < best_delta:
                            best_delta = delta
                            best_i, best_j = i, j
                        elif not tabu_j and best_delta >= 0:
                            best_delta = delta
                            best_i, best_j = i, j

            if best_i < 0:
                break

            i, j = best_i, best_j
            u, v = p[i], p[j]
            p[i], p[j] = v, u

            cur += best_delta
            if cur < best:
                best = cur
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            tabu[i] = it + tabu_len
            tabu[j] = it + tabu_len
            it += 1

            if no_improve_count > 80:
                break

        # Kick to escape local minimum (except after last round)
        if rnd < num_rounds - 1 and time.time() < deadline - 15:
            # Perturb by swapping a few poorly-placed facilities
            num_swaps = 6
            weights = a_rowsum + 1
            weights = weights / weights.sum()
            chosen = np.random.choice(n, size=min(num_swaps, n // 2), replace=False, p=weights)
            for idx in chosen:
                j = np.random.randint(0, n)
                if j == idx:
                    j = (j + 1) % n
                u, v = p[idx], p[j]
                if u == v:
                    continue
                delta = compute_delta(idx, j, p, u, v)
                if delta < 0 or np.random.random() < 0.15:
                    p[idx], p[j] = v, u
                    cur += delta

    return best_p.tolist()
# EVOLVE-BLOCK-END

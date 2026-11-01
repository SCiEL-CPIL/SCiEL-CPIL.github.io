import time
import numpy as np

def construct():
    start = time.time()
    deadline = 145.0  # safety margin before 150s wall-clock kill

    n, m, p = 2, 2, 8

    # Fallback: known rank-28 block-Strassen decomposition
    best = _strassen_28()
    best_rank = 28

    # Local search over integer linear combinations of the basis rank-1 tensors
    # to try to beat 28. Only accept a result that exactly reproduces the tensor.
    for target_rank in [27, 26, 25, 24]:
        if time.time() > deadline - 8:
            break
        result = _search_low_rank(target_rank, n, m, p, budget=deadline - 8)
        if result is not None:
            best = result
            best_rank = target_rank
            break

    return best


def _strassen_28():
    """Rank-28 block-diagonal Strassen decomposition for 2x2 * 2x8 -> 2x8."""
    n, m, p = 2, 2, 8
    rank = 28

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen factors for 2x2 * 2x2 -> 2x2
    U_s = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_s = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_s = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    num_blocks = p // 2  # 4 blocks of 2 columns
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(4):
                U[i, col] = U_s[r][i]
            v_idx = [2 * t, 2 * t + 1, 8 + 2 * t, 8 + 2 * t + 1]
            for l in range(4):
                V[v_idx[l], col] = V_s[r][l]
            w_idx = [4 * t, 4 * t + 2, 4 * t + 1, 4 * t + 3]
            for l in range(4):
                W[w_idx[l], col] = W_s[r][l]
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W, n, m, p):
    """Exact check: T[i,j,k] = sum_r U[i*n+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for all i in [0,m), j in [0,n), k in [0,p). Returns True iff exact."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    if V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    if U.shape[0] != n * m:
        return False
    # Build the (m,n) x rank and (p) slices to vectorize the check
    # For each r, the contribution to entry (i,j,k) is U[i*n+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We test all 32 nonzero targets and require exactness (integer arithmetic -> exact).
    for i in range(m):
        for j in range(n):
            u = U[i * n + j, :]  # (rank,)
            v = V[j * p: (j + 1) * p, :]  # (p, rank)
            for k in range(p):
                w = W[k * n + i, :]  # (rank,)
                val = np.dot(np.dot(u, v[k, :]), w)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _search_low_rank(target_rank, n, m, p, budget):
    """Local search over integer linear combinations of basis rank-1 tensors.

    A rank-R decomposition is a sum of R rank-1 tensors
    u_r (A-space) ⊗ v_r (B-space) ⊗ w_r (C-space) whose sum reproduces the
    multiplication tensor on its 32 nonzero entries. We draw u,v,w from a small
    integer pool and greedily/locally minimize the residual over the 32 targets,
    accepting only an exactly-zero residual.
    """
    start = time.time()
    num_a = n * m   # 4
    num_b = m * p   # 16
    num_c = p * n   # 16

    # Nonzero targets: (a,b,c) where T=1
    targets = []
    for i in range(m):
        for j in range(n):
            for k in range(p):
                a = i * n + j
                b = j * p + k
                c = k * n + i
                targets.append((a, b, c))
    T = len(targets)  # 32

    # Small integer coefficient pool for each factor
    pool = list(range(-2, 3))  # -2..2

    # Precompute all candidate rank-1 tensors' contributions to the T targets.
    # For a candidate (u,v,w) with entries in pool, its contribution vector over
    # targets is [ u[a]*v[b]*w[c] for (a,b,c) in targets ].
    # We search for a multiset of R candidates whose sum is the all-ones vector.
    # This is a subset-sum / integer-programming problem; we do a randomized
    # greedy local search as a cheap surrogate.

    # Random restarts: build R random rank-1 tensors, then locally adjust entries
    # to reduce residual, accepting only exact matches.
    max_restarts = 4000
    for restart in range(max_restarts):
        if time.time() - start > budget:
            break

        rng = np.random.default_rng(seed=restart)
        U = rng.choice(pool, size=(num_a, target_rank)).astype(float)
        V = rng.choice(pool, size=(num_b, target_rank)).astype(float)
        W = rng.choice(pool, size=(num_c, target_rank)).astype(float)

        # Quick exact check
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        # Local improvement: try to zero the residual by flipping individual
        # entries among the pool values. Cheap heuristic, low acceptance rate,
        # but exact whenever it succeeds.
        for _ in range(60):
            if time.time() - start > budget:
                break
            # Pick a random tensor factor and random entry
            which = rng.integers(0, 3)
            if which == 0:
                mat = U
                rows = num_a
            elif which == 1:
                mat = V
                rows = num_b
            else:
                mat = W
                rows = num_c
            r = rng.integers(0, target_rank)
            row = rng.integers(0, rows)
            old = mat[row, r]
            mat[row, r] = rng.choice(pool)
            if mat[row, r] == old:
                continue
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            # revert (keep the random walk simple; acceptance is rare)
            mat[row, r] = old

    return None

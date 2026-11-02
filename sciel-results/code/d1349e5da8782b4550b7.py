"""Combined: rank-28 Strassen baseline + structured rank-reduction search."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 8
    nm, mp, pn = n * m, m * p, p * n

    # ------------------------------------------------------------------
    # 1. Build the rank-28 Strassen solution (known valid, half-integer)
    # ------------------------------------------------------------------
    best = _strassen_28(n, m, p)

    # ------------------------------------------------------------------
    # 2. Build the target tensor T[a,b,c]
    # ------------------------------------------------------------------
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # ------------------------------------------------------------------
    # 3. Try to find a lower-rank exact decomposition
    #    Strategy: take the 28 Strassen factors, try to express some
    #    columns as linear combinations of others, then verify.
    #    Also try random half-integer search with ALS refinement.
    # ------------------------------------------------------------------
    U28 = np.array(best["U"], dtype=float)
    V28 = np.array(best["V"], dtype=float)
    W28 = np.array(best["W"], dtype=float)

    # Try ranks from 27 down to 20
    for target_rank in range(27, 19, -1):
        if time.time() > deadline:
            break

        # --- Approach A: structured reduction of Strassen columns ---
        result = _structured_reduction(U28, V28, W28, target_rank, n, m, p,
                                       T_target, start_time, deadline)
        if result is not None:
            best = result
            break

        # --- Approach B: random half-integer init + ALS refinement ---
        result = _random_als_search(n, m, p, nm, mp, pn, target_rank,
                                    T_target, start_time, deadline)
        if result is not None:
            best = result
            break

    # Ensure return format: list of lists of floats
    U_out = [[float(x) for x in row] for row in np.array(best["U"]).tolist()]
    V_out = [[float(x) for x in row] for row in np.array(best["V"]).tolist()]
    W_out = [[float(x) for x in row] for row in np.array(best["W"]).tolist()]

    return {"U": U_out, "V": V_out, "W": W_out}


def _strassen_28(n, m, p):
    """Rank-28 solution: 4 independent 2x2 Strassen blocks."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    A_forms = [
        {0: 1, 3: 1},       # M1: a+d
        {2: 1, 3: 1},       # M2: c+d
        {0: 1},             # M3: a
        {3: 1},             # M4: d
        {0: 1, 1: 1},       # M5: a+b
        {2: 1, 0: -1},      # M6: c-a
        {1: 1, 3: -1}       # M7: b-d
    ]
    B_forms = [
        {0: 1, 3: 1},       # M1: x0+y1
        {0: 1},             # M2: x0
        {1: 1, 3: -1},      # M3: x1-y1
        {2: 1, 0: -1},      # M4: y0-x0
        {3: 1},             # M5: y1
        {0: 1, 1: 1},       # M6: x0+x1
        {2: 1, 3: 1}        # M7: y0+y1
    ]
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},
        {2: 1, 4: 1},
        {1: 1, 3: 1},
        {0: 1, 1: -1, 2: 1, 5: 1}
    ]

    for block in range(4):
        k0 = block * 2
        for s in range(7):
            r = block * 7 + s
            for idx, val in A_forms[s].items():
                i = idx // 2
                j = idx % 2
                U[i * m + j][r] = val
            for idx, val in B_forms[s].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global][r] = val
            for c_idx, coeffs in enumerate(C_coeffs):
                if s in coeffs:
                    coeff = coeffs[s]
                    i = c_idx // 2
                    k_local = c_idx % 2
                    k_global = k0 + k_local
                    W[k_global * n + i][r] = coeff

    return {"U": U, "V": V, "W": W}


def _structured_reduction(U28, V28, W28, target_rank, n, m, p, T_target,
                          start_time, deadline):
    """
    Try to reduce rank by finding linear dependencies among the 28 columns.
    For each subset of columns to keep (size target_rank), check if the
    remaining columns can be expressed as linear combinations.
    """
    nm, mp, pn = n * m, m * p, p * n
    R = 28
    to_remove = R - target_rank

    # Enumerate which columns to drop (combinations)
    # For small to_remove (1-8), this is feasible
    from itertools import combinations

    if to_remove > 8:
        return None

    # Build the full tensor from the 28 factors
    T_full = np.einsum('ar,br,cr->abc', U28, V28, W28)

    # For each combination of columns to remove, check if the tensor
    # is preserved when we drop those columns.
    # Actually, we need: does the tensor from remaining columns equal T_full?
    # That's only true if the removed columns contribute zero, which they don't.
    # Instead: we need to find NEW factors of lower rank that produce T_full.
    # This is a different problem. Let's use a different approach.

    # Better approach: try to find a rank-r sub-decomposition by
    # selecting r columns and checking if they span the tensor.
    # The tensor has rank r if it can be written as sum of r rank-1 terms.
    # We check: can we find r columns (possibly new) that reproduce T_full?

    # Alternative: use the fact that the tensor T_full has a specific structure.
    # Try: pick target_rank columns from the 28, and see if we can adjust
    # them (by linear combination) to reproduce T_full.

    # Actually the cleanest approach: for each set of target_rank columns,
    # project the tensor onto the subspace spanned by those columns'
    # outer products and check if the residual is zero.

    # This is expensive. Let's just try a few random subsets.
    rng = np.random.RandomState(42)

    for attempt in range(50):
        if time.time() > deadline:
            return None

        # Randomly choose target_rank columns out of 28
        keep = sorted(rng.choice(R, size=target_rank, replace=False))

        U_k = U28[:, keep]
        V_k = V28[:, keep]
        W_k = W28[:, keep]

        # Check if these columns reproduce T_full
        T_k = np.einsum('ar,br,cr->abc', U_k, V_k, W_k)

        if np.allclose(T_k, T_full, atol=1e-9):
            # Verify it matches T_target
            if np.allclose(T_k, T_target, atol=1e-9):
                return {"U": U_k.tolist(), "V": V_k.tolist(), "W": W_k.tolist()}

    return None


def _random_als_search(n, m, p, nm, mp, pn, target_rank, T_target,
                       start_time, deadline):
    """
    Random half-integer initialization + Alternating Least Squares.
    """
    max_attempts = 30
    max_iters = 300

    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None

        rng = np.random.RandomState(attempt * 7 + target_rank)

        # Initialize with small random half-integers
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        U = rng.choice(vals, size=(nm, target_rank))
        V = rng.choice(vals, size=(mp, target_rank))
        W = rng.choice(vals, size=(pn, target_rank))

        # ALS refinement
        success = False
        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Update U: for each a, solve M_a @ U[a,:] = rhs_a
            for a in range(nm):
                i = a // m
                j = a % m
                # Equations: for each b where T_target[a,b,c] != 0
                # Actually: sum_r U[a,r]*V[b,r]*W[c,r] = T_target[a,b,c]
                # For fixed a: for each (b,c) with T_target[a,b,c]=1:
                #   sum_r U[a,r] * (V[b,r]*W[c,r]) = 1
                # Build system
                M = np.zeros((p, target_rank))  # p equations (one per k)
                # b = j*p + k, c = k*n + i
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                rhs = np.ones(p)
                try:
                    U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                except:
                    pass

            # Update V: for each b
            for b in range(mp):
                j = b // p
                k = b % p
                M = np.zeros((n, target_rank))  # n equations (one per i)
                for i in range(n):
                    a = i * m + j
                    c = k * n + i
                    M[i, :] = U[a, :] * W[c, :]
                rhs = np.ones(n)
                try:
                    V[b, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                except:
                    pass

            # Update W: for each c
            for c in range(pn):
                k = c // n
                i = c % n
                M = np.zeros((m, target_rank))  # m equations (one per j)
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    M[j, :] = U[a, :] * V[b, :]
                rhs = np.ones(m)
                try:
                    W[c, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                except:
                    pass

            # Check convergence
            T_cur = np.einsum('ar,br,cr->abc', U, V, W)
            max_err = np.max(np.abs(T_cur - T_target))
            if max_err < 1e-10:
                success = True
                break

        if success:
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2

            T_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            if np.allclose(T_r, T_target, atol=1e-9):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    return None

# EVOLVE-BLOCK-END

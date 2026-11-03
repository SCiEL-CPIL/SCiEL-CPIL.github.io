"""Quadratic assignment for QAPLIB wil100: tabu search with rank-matching init."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _compute_delta(a, b, p, i, j):
    pi = p[i]
    pj = p[j]
    if pi == pj:
        return 0
    # Contribution of row i and column i (and row j, column j) to the objective.
    # When we swap p[i] and p[j], only these four rows/cols of b change.
    row_i = b[pi]
    col_i = b[:, pi]
    row_j = b[pj]
    col_j = b[:, pj]

    # For facility i moving to location pj:
    #   delta_i = sum_k a[i,k]*(b[pj,k] - b[pi,k]) + sum_k a[k,i]*(b[k,pj] - b[k,pi])
    # For facility j moving to location pi:
    #   delta_j = sum_k a[j,k]*(b[pi,k] - b[pj,k]) + sum_k a[k,j]*(b[k,pi] - b[k,pj])
    # Note: the i<->j cross terms cancel because b[pi,pj]==b[pj,pi] is not required,
    # but a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi] (before) vs a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj] (after)
    # The difference is (a[i,j]-a[j,i])*(b[pi,pj]-b[pj,pi]) which is zero if a is symmetric.
    # wil100 has symmetric a and b, so cross terms cancel. But to be safe, compute fully.

    # Full delta without assuming symmetry:
    # before: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    # after:  a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    cross_before = a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi]
    cross_after = a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj]
    cross_delta = cross_after - cross_before

    # Non-cross terms for i:
    delta_i_noncross = np.dot(a[i, :], row_j - row_i) + np.dot(a[:, i], col_j - col_i)
    # Remove the j-index contributions from the non-cross sums (they're double-counted)
    delta_i_noncross -= a[i, j] * (b[pj, j] - b[pi, j]) + a[j, i] * (b[j, pj] - b[j, pi])

    # Non-cross terms for j:
    delta_j_noncross = np.dot(a[j, :], row_i - row_j) + np.dot(a[:, j], col_i - col_j)
    delta_j_noncross -= a[j, i] * (b[pi, i] - b[pj, i]) + a[i, j] * (b[i, pi] - b[i, pj])

    return int(cross_delta + delta_i_noncross + delta_j_noncross)


def _rank_matching_init(a, b):
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def solve(a, b):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    n = len(a)

    # Precompute: a is the flow matrix, b is the distance matrix
    # Use int64 for exact arithmetic
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)

    # Initial solution via rank matching
    p = _rank_matching_init(a, b).copy()
    best_cost = _compute_cost(a, b, p)
    best_p = p.copy()

    # Tabu search parameters
    tabu_tenure = 15
    max_iter = 50000
    no_improve_limit = 800

    # Tabu list: for each facility, the last location it was moved to
    tabu = np.full(n, -1, dtype=np.int64)
    tabu_until = np.zeros(n, dtype=np.int64)

    iter_count = 0
    no_improve = 0

    while iter_count < max_iter:
        # Check deadline
        if time.time() > deadline - 5.0:
            break

        best_delta = 0  # 0 means no improving move; we allow tabu moves if they're best
        best_i = -1
        best_j = -1
        best_aspiration = False

        # Evaluate all pairs (i, j) with i < j
        # For speed, we can vectorize partially, but for n=100, 4950 pairs is fine
        for i in range(n):
            for j in range(i + 1, n):
                delta = _compute_delta(a, b, p, i, j)

                is_tabu = tabu_until[i] > iter_count or tabu_until[j] > iter_count

                # Aspiration: accept tabu move if it improves best
                if delta < best_cost:
                    best_aspiration = True
                else:
                    best_aspiration = False

                if not is_tabu or best_aspiration:
                    if delta < best_delta or (delta == best_delta and is_tabu and not best_aspiration):
                        # Prefer non-tabu improving moves, then tabu aspiration
                        if not is_tabu:
                            if delta < best_delta or (best_i == -1):
                                best_delta = delta
                                best_i = i
                                best_j = j
                                best_aspiration = False
                        elif best_aspiration:
                            if best_i == -1 or delta < best_delta:
                                best_delta = delta
                                best_i = i
                                best_j = j
                                best_aspiration = True

        # If no valid move found, try any move (relax tabu)
        if best_i == -1:
            for i in range(n):
                for j in range(i + 1, n):
                    delta = _compute_delta(a, b, p, i, j)
                    if best_i == -1 or delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        best_aspiration = False

        if best_i == -1:
            break

        # Apply the move
        p[best_i], p[best_j] = p[best_j], p[best_i]

        # Set tabu
        tabu_until[best_i] = iter_count + tabu_tenure
        tabu_until[best_j] = iter_count + tabu_tenure

        current_cost = _compute_cost(a, b, p)

        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve = 0
        else:
            no_improve += 1

        iter_count += 1

        if no_improve > no_improve_limit:
            # Restart with a perturbation
            if time.time() > deadline - 10.0:
                break
            # Random perturbation: swap 5 random pairs
            for _ in range(5):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
            current_cost = _compute_cost(a, b, p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve = 0
            tabu_until[:] = 0

    return best_p.tolist()
# EVOLVE-BLOCK-END

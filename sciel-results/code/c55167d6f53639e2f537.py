"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _compute_C(a, b, p):
    """Compute C = a * b[p][:, p], the per-pair cost matrix."""
    Bp = b[np.ix_(p, p)]
    return a * Bp


def _delta_swap_fast(C, a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    Uses precomputed C = a * b[p][:, p].
    """
    pi = p[i]
    pj = p[j]
    n = len(p)

    # Contribution from pairs involving i or j
    # Old: C[i,k]*1 + C[k,i]*1 for all k != i,j, plus C[i,j], C[j,i], C[i,i], C[j,j]
    # New after swap: facility i at loc pj, facility j at loc pi
    # C_new[i,k] = a[i,k]*b[pj, p[k]] for k != i,j
    # C_new[k,i] = a[k,i]*b[p[k], pj] for k != i,j
    # C_new[j,k] = a[j,k]*b[pi, p[k]] for k != i,j
    # C_new[k,j] = a[k,j]*b[p[k], pi] for k != i,j
    # C_new[i,j] = a[i,j]*b[pj, pi]
    # C_new[j,i] = a[j,i]*b[pi, pj]
    # C_new[i,i] = a[i,i]*b[pj, pj]
    # C_new[j,j] = a[j,j]*b[pi, pi]

    # Sum over k != i,j of (C[i,k] + C[k,i]) - (a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj]
    #                                          + a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi])
    # = sum_k [C[i,k] + C[k,i] - a[i,k]*b[pj,p[k]] - a[k,i]*b[p[k],pj]
    #           - a[j,k]*b[pi,p[k]] - a[k,j]*b[p[k],pi]]
    # But C[i,k] = a[i,k]*b[pi,p[k]], so:
    # = sum_k [a[i,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,i]*(b[p[k],pi] - b[p[k],pj])
    #           - a[j,k]*b[pi,p[k]] - a[k,j]*b[p[k],pi]]

    # Vectorized: compute for all k, then subtract k=i and k=j terms
    # For all k:
    term_i_row = a[i, :] * (b[pi, p] - b[pj, p])  # a[i,k]*(b[pi,p[k]] - b[pj,p[k]])
    term_i_col = a[:, i] * (b[p, pi] - b[p, pj])  # a[k,i]*(b[p[k],pi] - b[p[k],pj])
    term_j_row = a[j, :] * b[pi, p]               # a[j,k]*b[pi,p[k]]
    term_j_col = a[:, j] * b[p, pi]               # a[k,j]*b[p[k],pi]

    total = np.sum(term_i_row + term_i_col - term_j_row - term_j_col)

    # Subtract the k=i and k=j terms (they were included in the sums above)
    # For k=i:
    ki_i = term_i_row[i] + term_i_col[i] - term_j_row[i] - term_j_col[i]
    # For k=j:
    kj_j = term_i_row[j] + term_i_col[j] - term_j_row[j] - term_j_col[j]

    total -= ki_i + kj_j

    # Self terms:
    # Old self: C[i,i] + C[j,j] + C[i,j] + C[j,i]
    #           = a[i,i]*b[pi,pi] + a[j,j]*b[pj,pj] + a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    # New self: a[i,i]*b[pj,pj] + a[j,j]*b[pi,pi] + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    old_self = (a[i, i] * b[pi, pi] + a[j, j] * b[pj, pj]
                + a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi])
    new_self = (a[i, i] * b[pj, pj] + a[j, j] * b[pi, pi]
                + a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj])

    return int(total + new_self - old_self)


def _local_search_fast(a, b, p, deadline, max_passes=100):
    """
    First-improvement local search with 2-opt swaps using precomputed C.
    Returns (p, cost).
    """
    n = len(p)
    cost = _cost(a, b, p)
    C = _compute_C(a, b, p)

    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p, cost
            for j in range(i + 1, n):
                delta = _delta_swap_fast(C, a, b, p, i, j)
                if delta < 0:
                    # Swap i and j
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    # Update C
                    pi = p[i]
                    pj = p[j]
                    # Update row i of C: C[i,k] = a[i,k]*b[pi, p[k]]
                    C[i, :] = a[i, :] * b[pi, p]
                    # Update col i of C: C[k,i] = a[k,i]*b[p[k], pi]
                    C[:, i] = a[:, i] * b[p, pi]
                    # Update row j of C: C[j,k] = a[j,k]*b[pj, p[k]]
                    C[j, :] = a[j, :] * b[pj, p]
                    # Update col j of C: C[k,j] = a[k,j]*b[p[k], pj]
                    C[:, j] = a[:, j] * b[p, pj]
                    improved = True
        if not improved:
            break

    return p, cost


def _spectral_init(a, b, n, rng):
    """
    Spectral initialization: align facility ordering with location ordering
    based on eigenvectors of the flow and distance matrices.
    """
    try:
        evals_a, evecs_a = np.linalg.eigh(a)
        evals_b, evecs_b = np.linalg.eigh(b)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)

    idx_a = np.argmax(evals_a)
    idx_b = np.argmax(evals_b)

    vec_a = evecs_a[:, idx_a]
    vec_b = evecs_b[:, idx_b]

    facility_order = np.argsort(-vec_a, kind="stable")
    location_order = np.argsort(-vec_b, kind="stable")

    p = np.empty(n, dtype=np.int64)
    p[facility_order] = location_order

    if rng.random() < 0.5:
        location_order = location_order[::-1]
        p = np.empty(n, dtype=np.int64)
        p[facility_order] = location_order

    return p


def _best_improvement_pass_fast(a, b, p, deadline):
    """
    One pass of best-improvement using precomputed C.
    Returns (p, cost, best_delta).
    """
    n = len(p)
    cost = _cost(a, b, p)
    C = _compute_C(a, b, p)
    best_delta = 0
    best_i = -1
    best_j = -1

    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            delta = _delta_swap_fast(C, a, b, p, i, j)
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_j = j

    if best_i >= 0:
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += best_delta

    return p, cost, best_delta


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0

    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    best_p = None
    best_cost = np.inf

    rng = np.random.default_rng(12345)

    # Phase 1: Multiple heuristic initializations with local search
    # Heuristic 1: Rank-matching
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    p1, c1 = _local_search_fast(a, b, p1.copy(), deadline)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()

    # Heuristic 2: Reverse rank-matching
    heavy2 = np.argsort(a_row_sums, kind="stable")
    central2 = np.argsort(-b_row_sums, kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy2] = central2
    p2, c2 = _local_search_fast(a, b, p2.copy(), deadline)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()

    # Heuristic 3: Spectral
    p3 = _spectral_init(a, b, n, rng)
    p3, c3 = _local_search_fast(a, b, p3, deadline)
    if c3 < best_cost:
        best_cost = c3
        best_p = p3.copy()

    # Heuristic 4: Spectral variant
    p4 = _spectral_init(a, b, n, rng)
    p4, c4 = _local_search_fast(a, b, p4, deadline)
    if c4 < best_cost:
        best_cost = c4
        best_p = p4.copy()

    # Heuristic 5: Random
    p5 = rng.permutation(n).astype(np.int64)
    p5, c5 = _local_search_fast(a, b, p5, deadline)
    if c5 < best_cost:
        best_cost = c5
        best_p = p5.copy()

    # Phase 2: ILS
    current_p = best_p.copy()
    current_cost = best_cost
    ilse_deadline = deadline - 5.0

    iteration = 0
    while time.time() < ilse_deadline:
        iteration += 1
        k = 2 + (iteration % 4)
        perturbed_p = current_p.copy()
        for _ in range(k):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]

        perturbed_p, perturbed_cost = _local_search_fast(a, b, perturbed_p, ilse_deadline)

        if perturbed_cost < best_cost:
            best_cost = perturbed_cost
            best_p = perturbed_p.copy()
            current_p = perturbed_p
            current_cost = perturbed_cost
        elif perturbed_cost == current_cost:
            if rng.random() < 0.1:
                current_p = perturbed_p
                current_cost = perturbed_cost

    # Phase 3: Final polish
    if time.time() < deadline - 2.0:
        polish_deadline = deadline - 1.0
        polish_p = best_p.copy()
        polish_cost = best_cost
        for _ in range(50):
            if time.time() > polish_deadline:
                break
            polish_p, polish_cost, bd = _best_improvement_pass_fast(a, b, polish_p, polish_deadline)
            if bd == 0:
                break
        if polish_cost < best_cost:
            best_cost = polish_cost
            best_p = polish_p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END

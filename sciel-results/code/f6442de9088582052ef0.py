"""QAP tai256c solver: iterated local search with VNS (k-opt relocations, cyclic shifts),
vectorized 2-opt, Hungarian re-optimization, and adaptive diversification."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s limit

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]

        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]

        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return int(s1 + s2 + cross)

    def local_search(p, cost, max_passes=30, exact=False):
        """Steepest Descent Local Search with vectorized delta calculation for speed."""
        improved = True
        passes = 0
        p = p.copy()

        # Precompute symmetric A for vectorized delta calculation
        A_sym = a + a.T
        D = b[np.ix_(p, p)].copy()

        while improved and passes < max_passes:
            improved = False
            passes += 1

            for i in range(n - 1):
                if time.time() > deadline:
                    return p, cost

                if exact:
                    # Exact O(n) delta calculation for final polish
                    best_delta = 0
                    best_j = -1
                    for j in range(i + 1, n):
                        delta = fast_swap_delta(p, i, j)
                        if delta < best_delta:
                            best_delta = delta
                            best_j = j
                    min_delta = best_delta
                    min_idx = None
                    if best_j >= 0:
                        j = best_j
                    else:
                        continue
                else:
                    js = np.arange(i + 1, n)

                    # Vectorized delta calculation
                    diff_A = A_sym[i] - A_sym[js]
                    diff_D = D[js] - D[i]

                    S = np.sum(diff_A * diff_D, axis=1)

                    # Correction terms
                    coeff_i = A_sym[i, i] - A_sym[js, i]
                    term_i = coeff_i * (D[js, i] - D[i, i])

                    coeff_j = A_sym[i, js] - np.diag(A_sym[js])
                    term_j = coeff_j * (D[js, js] - D[i, js])

                    diag_term = (a[i, i] - np.diag(a[js])) * (D[js, js] - D[i, i])
                    cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])

                    deltas = S - term_i - term_j + diag_term + cross_term

                    min_idx = np.argmin(deltas)
                    min_delta = int(deltas[min_idx])

                    if min_delta < 0:
                        j = js[min_idx]
                    else:
                        continue

                if min_delta < 0:
                    p[i], p[j] = p[j], p[i]

                    # Update D
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()

                    cost += min_delta
                    improved = True

        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]

        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]

        C = M1 + M2

        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def hungarian_init():
        """Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix."""
        cost_matrix = a @ b + a.T @ b.T
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def tabu_kick(p, cost, n_kicks=8):
        """Apply several random tabu-like kicks to diversify, keeping best result."""
        best_p = p.copy()
        best_cost = cost
        tabu_tenure = 12
        tabu_list = {}
        iteration = 0

        for _ in range(n_kicks):
            if time.time() > deadline - 5:
                break
            # Pick a random pair
            i, j = np.random.choice(n, 2, replace=False)
            key = (min(i, j), max(i, j))

            # Check tabu
            is_tabu = key in tabu_list and tabu_list[key] > iteration

            # Compute delta
            delta = fast_swap_delta(p, i, j)

            # Accept if not tabu, or if it improves global best (aspiration)
            if not is_tabu or (cost + delta < best_cost):
                p[i], p[j] = p[j], p[i]
                cost += delta
                tabu_list[key] = iteration + tabu_tenure

                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()

            iteration += 1

        return best_p, best_cost

    def relocate(p, cost, k=1):
        """Perform k random relocations (move facility to random different location)."""
        p = p.copy()
        for _ in range(k):
            if time.time() > deadline:
                break
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            # Compute delta for moving facility i to location j (swap with facility at j)
            delta = fast_swap_delta(p, i, j)
            p[i], p[j] = p[j], p[i]
            cost += delta
        return p, cost

    def cyclic_shift(p, cost, length):
        """Perform a cyclic shift of a random sub-permutation of given length."""
        p = p.copy()
        start = np.random.randint(max(1, n - length + 1))
        # Extract the sub-permutation
        segment = p[start:start + length].copy()
        # Rotate by 1
        segment = np.roll(segment, -1)
        p[start:start + length] = segment
        new_cost = compute_cost(p)
        return p, new_cost

    def vns_perturb(p, cost, level=1):
        """Variable Neighborhood Search perturbation at given level.
        Level 1: single relocation
        Level 2: double relocation
        Level 3: cyclic shift of length 3
        Level 4: cyclic shift of length 4
        Level 5: random 3-swap
        Level 6: tabu kick
        """
        if time.time() > deadline:
            return p, cost

        if level == 1:
            return relocate(p, cost, k=1)
        elif level == 2:
            return relocate(p, cost, k=2)
        elif level == 3:
            return cyclic_shift(p, cost, length=3)
        elif level == 4:
            return cyclic_shift(p, cost, length=4)
        elif level == 5:
            # Random 3-swap
            p = p.copy()
            indices = np.random.choice(n, 3, replace=False)
            # Swap in a cycle: 0->1, 1->2, 2->0
            p[indices[0]], p[indices[1]], p[indices[2]] = p[indices[2]], p[indices[0]], p[indices[1]]
            new_cost = compute_cost(p)
            return p, new_cost
        elif level == 6:
            return tabu_kick(p, cost, n_kicks=8)
        else:
            # Fallback: random swaps
            p = p.copy()
            for _ in range(level):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
            new_cost = compute_cost(p)
            return p, new_cost

    # Phase 1: Hungarian Initialization
    p_hung = hungarian_init()
    best_p = p_hung.copy()
    best_cost = compute_cost(best_p)

    # Refine with local search
    current_p, current_cost = local_search(best_p, best_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Phase 2: Rank matching initialization (alternative starting point)
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    p_rank = p_rank.astype(np.int64)

    p_rank_ls, cost_rank = local_search(p_rank, compute_cost(p_rank))
    if cost_rank < best_cost:
        best_p = p_rank_ls.copy()
        best_cost = cost_rank

    # Phase 2b: Another initialization via column sums
    a_col_sums = a.sum(axis=0)
    b_row_sums = b.sum(axis=1)
    heavy2 = np.argsort(-a_col_sums, kind="stable")
    central2 = np.argsort(b_row_sums, kind="stable")
    p_rank2 = np.empty(n, dtype=np.int64)
    p_rank2[heavy2] = central2
    p_rank2 = p_rank2.astype(np.int64)

    p_rank2_ls, cost_rank2 = local_search(p_rank2, compute_cost(p_rank2))
    if cost_rank2 < best_cost:
        best_p = p_rank2_ls.copy()
        best_cost = cost_rank2

    # Phase 3: Iterated Local Search with VNS
    iteration = 0
    consecutive_no_improve = 0
    vns_level = 1
    max_vns_level = 6

    while time.time() < deadline - 15:  # Leave 15s buffer for final polish
        iteration += 1

        # Adaptive VNS level: increase level when stagnant
        if consecutive_no_improve > 3:
            vns_level = min(max_vns_level, 1 + consecutive_no_improve // 2)
        else:
            vns_level = 1

        # Perturb using VNS
        p = best_p.copy()
        cost = best_cost
        p, cost = vns_perturb(p, cost, level=vns_level)

        # Local search from perturbed
        p, cost = local_search(p, cost)

        # Every few iterations, try Hungarian re-optimization
        if iteration % 4 == 0 and time.time() < deadline - 30:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            consecutive_no_improve = 0
            vns_level = 1  # Reset VNS level on improvement
        else:
            consecutive_no_improve += 1

    # Final high-precision polish: use exact delta calculation with more passes
    if time.time() < deadline - 10:
        final_p = best_p.copy()
        final_cost = best_cost
        # First do a vectorized pass to get close
        final_p, final_cost = local_search(final_p, final_cost, max_passes=15)
        if final_cost < best_cost:
            best_p = final_p.copy()
            best_cost = final_cost

        # Then do exact polish passes
        if time.time() < deadline - 5:
            exact_p = best_p.copy()
            exact_cost = best_cost
            exact_p, exact_cost = local_search(exact_p, exact_cost, max_passes=10, exact=True)
            if exact_cost < best_cost:
                best_p = exact_p
                best_cost = exact_cost

        # One more vectorized pass to catch any remaining improvements
        if time.time() < deadline - 2:
            final_p = best_p.copy()
            final_cost = best_cost
            final_p, final_final_cost = local_search(final_p, final_cost, max_passes=5)
            if final_final_cost < best_cost:
                best_p = final_p
                best_cost = final_final_cost

        # Final exact pass
        if time.time() < deadline - 1:
            exact_p = best_p.copy()
            exact_cost = best_cost
            exact_p, exact_cost = local_search(exact_p, exact_cost, max_passes=5, exact=True)
            if exact_cost < best_cost:
                best_p = exact_p
                best_cost = exact_cost

    return best_p.tolist()

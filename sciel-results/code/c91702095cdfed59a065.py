"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    """
    Solve the Quadratic Assignment Problem for a n x n instance.
    Uses rank-matching initialization followed by efficient 2-opt local search
    with time-bounded iterated local search and adaptive perturbations.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin within 120s limit

    n = len(a)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def compute_cost(p):
        """Compute the exact QAP cost for assignment p."""
        B_p = b[np.ix_(p, p)]
        return int(np.sum(a * B_p))

    def greedy_init():
        """Rank-matching: high-flow facilities to low-distance locations."""
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def swap_delta_fast(p, i, j):
        """
        Fast O(n) swap delta for swapping facilities i and j.
        delta = sum_k (a[i,k]-a[j,k])*(b[pj,k]-b[pi,k]) + sum_k (a[k,i]-a[k,j])*(b[k,pj]-b[k,pi])
        """
        pi = p[i]
        pj = p[j]
        row_diff = b[pj] - b[pi]
        col_diff = b[:, pj] - b[:, pi]
        delta = np.dot(a[i, :] - a[j, :], row_diff) + np.dot(a[:, i] - a[:, j], col_diff)
        return delta

    def local_search(p, max_passes=100):
        """2-opt local search with first-improvement and random offset."""
        p = p.copy()
        cost = compute_cost(p)
        n_local = len(p)

        for _pass in range(max_passes):
            if time.time() > deadline:
                break

            improved = False
            offset = np.random.randint(n_local)
            for ii in range(n_local):
                i = (offset + ii) % n_local
                for jj in range(ii + 1, n_local):
                    j = (offset + jj) % n_local
                    if j == i:
                        continue
                    delta = swap_delta_fast(p, i, j)
                    if delta < -1e-9:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if time.time() > deadline:
                    return p, cost

            if not improved:
                break

        return p, cost

    def polish(p, max_passes=50):
        """High-precision polish: multiple full passes with best-improvement until no improvement."""
        p = p.copy()
        cost = compute_cost(p)
        n_local = len(p)

        for _pass in range(max_passes):
            if time.time() > deadline - 1.0:
                break

            best_delta = 0.0
            best_i, best_j = -1, -1
            for i in range(n_local):
                for j in range(i + 1, n_local):
                    delta = swap_delta_fast(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j = i, j

            if best_i < 0 or best_delta >= -1e-12:
                break

            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta

        return p, cost

    def perturb(p, num_swaps):
        """Apply random multi-swap perturbation with occasional 3-opt."""
        p = p.copy()
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        # Occasionally add a 3-opt move for more diversity
        if np.random.random() < 0.2 and num_swaps >= 3:
            i, j, k = np.random.choice(n, 3, replace=False)
            p[i], p[j], p[k] = p[j], p[k], p[i]
        return p

    # Initial solution via rank matching
    best_p = greedy_init()
    best_cost = compute_cost(best_p)

    # First local search to convergence
    best_p, best_cost = local_search(best_p, max_passes=200)

    # Iterated local search with adaptive perturbation
    stall_count = 0
    stall_threshold = 4

    while time.time() < deadline - 5.0:
        if stall_count < stall_threshold:
            num_swaps = 2
        elif stall_count < 2 * stall_threshold:
            num_swaps = 4
        else:
            num_swaps = 6

        p_pert = perturb(best_p, num_swaps)
        p_new, cost_new = local_search(p_pert, max_passes=80)

        if cost_new < best_cost:
            best_p = p_new
            best_cost = cost_new
            stall_count = 0
        else:
            if stall_count > 3 * stall_threshold and np.random.random() < 0.15:
                best_p = p_new
                best_cost = cost_new
            stall_count += 1

        # If heavily stalled, try fresh initialization
        if stall_count > 12:
            p_fresh = greedy_init()
            p_new, cost_new = local_search(p_fresh, max_passes=80)
            if cost_new < best_cost:
                best_p = p_new
                best_cost = cost_new
            stall_count = 0

    # Final high-precision polish
    if time.time() < deadline - 3.0:
        best_p, best_cost = polish(best_p, max_passes=100)

    # One more local search pass if time allows
    if time.time() < deadline - 1.5:
        best_p, best_cost = local_search(best_p, max_passes=10)

    return best_p.tolist()

# EVOLVE-BLOCK-END

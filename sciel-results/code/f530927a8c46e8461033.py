"""Maximum Cut via fast vectorized local search with spectral init, 1-flip and 2-flip moves, SA escape, and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses fast vectorized greedy local search, 2-flip moves to escape local optima,
    simulated annealing, multiple restarts, and an explicit deadline guard.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Deadline: return best result before 140s (10s safety margin from 150s kill)
    deadline = time.time() + 140.0

    # Precompute degree for gain calculation
    degree = adj.sum(axis=1)

    # Spectral initialization: compute Fiedler vector for a good starting point
    def spectral_init():
        """Generate initial labeling using the Fiedler vector (2nd smallest eigenvalue of Laplacian)."""
        try:
            # Build sparse Laplacian
            L = laplacian(adj.astype(np.float64))
            # Get the second smallest eigenvalue and its eigenvector
            # Use sigma shift for efficiency
            if n > 2:
                vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
                # The second smallest eigenvalue corresponds to the Fiedler vector
                # eigsh with sigma=0, which='LM' finds eigenvalues closest to 0
                # Sort by eigenvalue to get the second smallest
                idx = np.argsort(vals)
                fiedler_vec = vecs[:, idx[1]]
                # Convert to 0/1 labels: positive -> 1, negative -> 0
                labels = (fiedler_vec >= 0).astype(np.int8)
            else:
                labels = np.zeros(n, dtype=np.int8)
        except Exception:
            # Fallback to random if spectral fails
            labels = np.random.randint(0, 2, size=n).astype(np.int8)
        return labels

    def compute_cut(labels):
        """Number of edges crossing the partition. Fast vectorized version."""
        # cut * 2 = sum_v sum_u adj[v,u] * (labels[v] != labels[u])
        # Use: (labels[None,:] != labels[:,None]) creates n x n bool matrix
        diff = (labels[None, :] != labels[:, None])
        cut2 = (adj * diff).sum()
        return cut2 // 2

    def fast_local_search(labels, max_rounds=200):
        """
        Fast greedy 1-flip local search.
        Key insight: gain[v] = 2 * (num neighbors with same label as v) - degree[v]
        We maintain same_count incrementally for speed.
        """
        current_cut = compute_cut(labels)

        # Precompute same_count for each vertex
        # same_count[v] = number of neighbors of v with same label as v
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)
        gains = 2 * same_count - degree

        for _ in range(max_rounds):
            if time.time() > deadline:
                break

            best_v = np.argmax(gains)
            best_gain = gains[best_v]

            if best_gain <= 0:
                break

            # Flip vertex best_v
            old_label = labels[best_v]
            labels[best_v] = 1 - old_label
            current_cut += best_gain

            # Update gains for neighbors of best_v
            # When we flip v, the same_count for each neighbor u changes:
            # If u and v had same label, after flip they differ -> same_count[u] decreases by 1
            # If u and v had different labels, after flip they same -> same_count[u] increases by 1
            # So: same_count[u] += (1 if labels[u] == new_label else -1) for each neighbor u
            # new_label = 1 - old_label
            # For neighbor u: if labels[u] == new_label, same_count[u] += 1; else same_count[u] -= 1
            # This is equivalent to: same_count[u] += (2 * labels[u] - 1) * (2 * new_label - 1)
            # Simpler: for each neighbor u of v, same_count[u] changes by +1 if labels[u] == new_label else -1
            # = 2*labels[u] - 1 if new_label == 1, or -(2*labels[u] - 1) if new_label == 0
            # = (2*labels[u] - 1) * (2*new_label - 1)
            # Actually: if new_label == 1: delta = 1 if labels[u]==1 else -1 = 2*labels[u] - 1
            # if new_label == 0: delta = 1 if labels[u]==0 else -1 = -(2*labels[u] - 1)
            # So delta = (2*labels[u] - 1) * (2*new_label - 1)

            new_label = 1 - old_label
            # Get neighbors of best_v
            neighbors = np.where(adj[best_v] == 1)[0]
            if len(neighbors) > 0:
                # For each neighbor, compute delta to same_count
                # delta_same[u] = 1 if labels[u] == new_label else -1
                # = 2*labels[u] - 1 if new_label == 1, else -(2*labels[u] - 1)
                # = (2*labels[u] - 1) * (2*new_label - 1)
                sign = 2 * new_label - 1  # 1 if new_label=1, -1 if new_label=0
                delta_same = (2 * labels[neighbors] - 1) * sign
                same_count[neighbors] += delta_same
                gains[neighbors] += 2 * delta_same

            # Also update gain for best_v itself (it's now flipped)
            # same_count[best_v] should be recomputed or updated
            # After flip, same_count[best_v] = degree[best_v] - old_same_count[best_v]
            # Because neighbors that were same are now different, and vice versa
            same_count[best_v] = degree[best_v] - same_count[best_v]
            gains[best_v] = 2 * same_count[best_v] - degree[best_v]

        return labels, current_cut

    def exhaustive_two_flip_polish(labels, max_passes=10):
        """
        Exhaustive 2-flip polish: check all pairs and flip if beneficial.
        For n=100, there are 4950 pairs, which is fast.
        """
        current_cut = compute_cut(labels)

        for _ in range(max_passes):
            if time.time() > deadline:
                break

            improved = False
            # Check all pairs
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break

                    # Compute gain of flipping both i and j
                    old_li = labels[i]
                    old_lj = labels[j]

                    # Flip both
                    labels[i] = 1 - old_li
                    labels[j] = 1 - old_lj

                    new_cut = compute_cut(labels)
                    gain_2flip = new_cut - current_cut

                    if gain_2flip > 0:
                        current_cut = new_cut
                        improved = True
                        break
                    else:
                        # Revert
                        labels[i] = old_li
                        labels[j] = old_lj

                if improved:
                    break

            if not improved:
                break

        return labels, current_cut

    def simulated_annealing(labels, initial_temp=15.0, cooling=0.9995, max_iters=20000):
        """
        Simulated Annealing with 1-flip moves.
        Uses incremental gain computation for speed.
        """
        current_cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = current_cut

        # Precompute same_count
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (adj * same_mask).sum(axis=1)

        temp = initial_temp
        no_improve_count = 0

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Random vertex
            v = np.random.randint(n)

            # Compute gain of flipping v
            same_count_v = same_count[v]
            gain = 2 * same_count_v - degree[v]

            # Accept if gain > 0 or with probability exp(gain / temp) for gain < 0
            accepted = False
            if gain >= 0:
                accepted = True
            else:
                # probability = exp(gain / temp), gain is negative
                if np.random.rand() < np.exp(gain / temp):
                    accepted = True

            if accepted:
                old_label = labels[v]
                labels[v] = 1 - old_label
                current_cut += gain

                # Update same_count for neighbors
                new_label = 1 - old_label
                neighbors = np.where(adj[v] == 1)[0]
                if len(neighbors) > 0:
                    sign = 2 * new_label - 1
                    delta_same = (2 * labels[neighbors] - 1) * sign
                    same_count[neighbors] += delta_same

                # Update same_count for v
                same_count[v] = degree[v] - same_count[v]

                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            temp *= cooling

            # Adaptive cooling: cool faster if stuck
            if no_improve_count > 500:
                temp *= 0.9
                no_improve_count = 0

        return best_labels, best_cut

    # Multiple restarts with different initializations
    best_labels_overall = None
    best_cut_overall = -1
    rng = np.random.RandomState(42)

    def try_restart(labels_init, tag=""):
        nonlocal best_labels_overall, best_cut_overall
        if time.time() > deadline:
            return

        labels = labels_init.copy()
        # Fast local search
        labels, cut = fast_local_search(labels, max_rounds=300)

        # Try exhaustive 2-flip polish
        if time.time() < deadline - 5:
            labels2, cut2 = exhaustive_two_flip_polish(labels, max_passes=10)
            if cut2 > cut:
                labels, cut = labels2, cut2
                # Polish after 2-flip
                labels, cut = fast_local_search(labels, max_rounds=100)

        if cut > best_cut_overall:
            best_cut_overall = cut
            best_labels_overall = labels.copy()

    # Restart 1: Spectral initialization (primary, uses global structure)
    labels_spectral = spectral_init()
    try_restart(labels_spectral, "spectral")

    # Restart 2: alternating
    try_restart(np.arange(n) % 2, "alt")

    # Restart 3: random
    try_restart(rng.randint(0, 2, size=n), "rand")

    # Restart 4: Spectral with sign flip (equivalent but tests robustness)
    labels_spectral2 = 1 - labels_spectral
    try_restart(labels_spectral2, "spectral_flip")

    # Restart 5: random with bias
    try_restart(rng.randint(0, 2, size=n), "rand2")

    # If we have time, do SA from best solution
    if best_labels_overall is not None and time.time() < deadline - 10:
        labels_sa = best_labels_overall.copy()
        labels_sa, cut_sa = simulated_annealing(labels_sa, initial_temp=20.0, cooling=0.999, max_iters=30000)

        # Polish
        labels_sa, cut_sa = fast_local_search(labels_sa, max_rounds=200)

        if cut_sa > best_cut_overall:
            best_cut_overall = cut_sa
            best_labels_overall = labels_sa.copy()

    # Additional SA pass from spectral init if still have time
    if time.time() < deadline - 15:
        labels_sa2 = spectral_init()
        labels_sa2, cut_sa2 = simulated_annealing(labels_sa2, initial_temp=25.0, cooling=0.998, max_iters=40000)
        labels_sa2, cut_sa2 = fast_local_search(labels_sa2, max_rounds=200)

        if cut_sa2 > best_cut_overall:
            best_cut_overall = cut_sa2
            best_labels_overall = labels_sa2.copy()

    # Final exhaustive 2-flip polish
    if best_labels_overall is not None and time.time() < deadline - 5:
        labels_final = best_labels_overall.copy()
        labels_final, cut_final = exhaustive_two_flip_polish(labels_final, max_passes=15)
        if cut_final > best_cut_overall:
            best_cut_overall = cut_final
            best_labels_overall = labels_final.copy()

    # Final 1-flip polish
    if best_labels_overall is not None and time.time() < deadline - 2:
        labels_final = best_labels_overall.copy()
        labels_final, cut_final = fast_local_search(labels_final, max_rounds=300)
        if cut_final > best_cut_overall:
            best_cut_overall = cut_final
            best_labels_overall = labels_final.copy()

    return best_labels_overall
# EVOLVE-BLOCK-END

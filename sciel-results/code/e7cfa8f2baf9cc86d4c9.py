"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin before 150s kill

    adj = np.array(adjacency, dtype=np.int8)
    best_labels = None
    best_cut = -1

    # Precompute neighbor lists for fast access
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]

    def compute_cut(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def fast_2swap_local_search(labels, max_passes=30):
        """
        2-swap local search: try flipping pairs of vertices simultaneously.
        This can escape 1-flip local optima.
        Returns (best_labels, best_cut).
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)
        best_local_cut = current_cut
        best_local_labels = labels.copy()

        for _ in range(max_passes):
            if time.time() > deadline - 2.0:
                break
            improved = False
            # Try all pairs in random order for diversity
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((i, j))
            random.shuffle(pairs)

            for (i, j) in pairs:
                if time.time() > deadline - 2.0:
                    break

                # Compute delta for 2-swap: flip both i and j
                # Current: edges (i,u) crossing if labels[i]!=labels[u]
                # After: labels[i] flips, labels[j] flips
                # Change in cut = sum over all edges affected by i or j
                # Edges involving i: (i,u) for all u != j
                #   Old crossing: labels[i] != labels[u]
                #   New crossing: (1-labels[i]) != labels[u] = labels[i] == labels[u]
                #   Delta_i = (#u where labels[i]==labels[u], u!=j) - (#u where labels[i]!=labels[u], u!=j)
                # Similarly for j.
                # But edge (i,j) is counted twice if we do separately. Let's be careful.
                # Actually, let's just compute directly for small n.
                # For n=100, pairs are 4950, and computing delta naively is O(deg(i)+deg(j))
                # which is fine.

                li = labels[i]
                lj = labels[j]
                nli = 1 - li
                nlj = 1 - lj

                # Edges from i to all except j
                delta = 0
                for u in neighbors[i]:
                    if u == j:
                        continue
                    if labels[i] == labels[u]:
                        # Currently same, after flip i is different -> edge becomes crossing: +1
                        delta += 1
                    else:
                        # Currently crossing, after flip i is same -> edge no longer crossing: -1
                        delta -= 1

                # Edges from j to all except i
                for u in neighbors[j]:
                    if u == i:
                        continue
                    if labels[j] == labels[u]:
                        delta += 1
                    else:
                        delta -= 1

                # Edge (i,j) itself:
                # Old: crossing if li != lj
                # New: crossing if nli != nlj which is same as li != lj
                # So edge (i,j) contribution doesn't change! No delta from (i,j).

                if delta > 0:
                    labels[i] = nli
                    labels[j] = nlj
                    current_cut += delta
                    improved = True
                    if current_cut > best_local_cut:
                        best_local_cut = current_cut
                        best_local_labels = labels.copy()

            if not improved:
                break

        return best_local_labels, best_local_cut

    def fast_1flip_local_search(labels, max_passes=50):
        """
        1-flip local search: try flipping single vertices.
        Returns (best_labels, best_cut).
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)
        best_local_cut = current_cut
        best_local_labels = labels.copy()

        for _ in range(max_passes):
            if time.time() > deadline - 2.0:
                break
            improved = False
            order = list(range(n))
            random.shuffle(order)

            for v in order:
                if time.time() > deadline - 2.0:
                    break
                # Compute delta for flipping v
                same_count = 0
                opp_count = 0
                for u in neighbors[v]:
                    if labels[v] == labels[u]:
                        same_count += 1
                    else:
                        opp_count += 1
                delta = same_count - opp_count
                if delta > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    improved = True
                    if current_cut > best_local_cut:
                        best_local_cut = current_cut
                        best_local_labels = labels.copy()

            if not improved:
                break

        return best_local_labels, best_local_cut

    def simulated_annealing_with_2swap(labels, T_initial, T_final, steps):
        """
        Simulated annealing with 2-swap moves.
        Returns (best_labels, best_cut).
        """
        labels = labels.copy()
        current_cut = compute_cut(labels)
        best_local_cut = current_cut
        best_local_labels = labels.copy()

        T = T_initial
        if T_initial > 0 and T_final > 0 and steps > 0:
            cooling = (T_final / T_initial) ** (1.0 / steps)
        else:
            cooling = 0.999

        for step in range(steps):
            if time.time() > deadline - 2.0:
                break

            # Choose move type: 70% 1-flip, 30% 2-swap
            if random.random() < 0.7:
                # 1-flip
                i = random.randint(0, n - 1)
                same_count = 0
                opp_count = 0
                for u in neighbors[i]:
                    if labels[i] == labels[u]:
                        same_count += 1
                    else:
                        opp_count += 1
                delta = same_count - opp_count

                accept = False
                if delta > 0:
                    accept = True
                elif T > 0:
                    if random.random() < np.exp(delta / T):
                        accept = True

                if accept:
                    labels[i] = 1 - labels[i]
                    current_cut += delta
                    if current_cut > best_local_cut:
                        best_local_cut = current_cut
                        best_local_labels = labels.copy()
            else:
                # 2-swap
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue

                li = labels[i]
                lj = labels[j]

                delta = 0
                for u in neighbors[i]:
                    if u == j:
                        continue
                    if labels[i] == labels[u]:
                        delta += 1
                    else:
                        delta -= 1

                for u in neighbors[j]:
                    if u == i:
                        continue
                    if labels[j] == labels[u]:
                        delta += 1
                    else:
                        delta -= 1

                accept = False
                if delta > 0:
                    accept = True
                elif T > 0:
                    if random.random() < np.exp(delta / T):
                        accept = True

                if accept:
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    current_cut += delta
                    if current_cut > best_local_cut:
                        best_local_cut = current_cut
                        best_local_labels = labels.copy()

            T *= cooling

        return best_local_labels, best_local_cut

    def spectral_init():
        """
        Spectral initialization using Fiedler vector of Laplacian.
        Returns 0/1 label array.
        """
        degree = adj.sum(axis=1).astype(np.float64)
        D = np.diag(degree)
        L = D - adj.astype(np.float64)
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Fiedler vector is eigenvector of second smallest eigenvalue
            fiedler = eigvecs[:, 1]
            labels = (fiedler > 0).astype(np.int8)
            # Randomly flip if needed to try both orientations
            if random.random() < 0.5:
                labels = 1 - labels
            return labels
        except:
            return np.random.randint(0, 2, size=n).astype(np.int8)

    def random_init():
        return np.random.randint(0, 2, size=n).astype(np.int8)

    def perturbed_init(base_labels, flip_prob=0.3):
        labels = base_labels.copy()
        mask = np.random.random(n) < flip_prob
        labels[mask] = 1 - labels[mask]
        return labels

    # Track best
    def try_labels(labels):
        nonlocal best_labels, best_cut
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Phase 1: Multiple spectral inits + local search
    for _ in range(5):
        if time.time() > deadline - 15.0:
            break
        labels = spectral_init()
        polished, cut = fast_1flip_local_search(labels, max_passes=20)
        try_labels(polished)
        polished2, cut2 = fast_2swap_local_search(polished, max_passes=15)
        try_labels(polished2)

    # Phase 2: SA with 2-swap from various inits
    num_sa_restarts = 25
    for r in range(num_sa_restarts):
        if time.time() > deadline - 10.0:
            break

        # Vary initialization
        if r % 4 == 0:
            init_labels = spectral_init()
        elif r % 4 == 1:
            init_labels = random_init()
        elif r % 4 == 2:
            init_labels = perturbed_init(spectral_init(), flip_prob=0.3)
        else:
            init_labels = perturbed_init(spectral_init(), flip_prob=0.5)

        T_initial = n / 3.0
        T_final = 0.05
        steps = 3000

        sa_labels, sa_cut = simulated_annealing_with_2swap(init_labels, T_initial, T_final, steps)
        try_labels(sa_labels)

        # Polish
        if time.time() < deadline - 5.0:
            polished, cut = fast_1flip_local_search(sa_labels, max_passes=30)
            try_labels(polished)
            polished2, cut2 = fast_2swap_local_search(polished, max_passes=20)
            try_labels(polished2)

    # Phase 3: Final intensive polish on best found
    if best_labels is not None and time.time() < deadline - 1.0:
        final = best_labels.copy()
        for _ in range(3):
            if time.time() > deadline - 1.0:
                break
            final, c1 = fast_1flip_local_search(final, max_passes=50)
            final, c2 = fast_2swap_local_search(final, max_passes=30)
        try_labels(final)

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END

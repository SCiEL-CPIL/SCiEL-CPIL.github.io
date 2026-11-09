"""Maximum Cut via multi-start spectral + Cerny tabu + optimized 2-swap local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s limit

    if n == 0:
        return np.array([], dtype=np.int8)

    adj = np.array(adjacency, dtype=np.int32)
    total_edges = int(np.sum(adj)) // 2

    # Precompute neighbor lists and degrees
    neighbors = [np.where(adj[i] > 0)[0] for i in range(n)]
    degrees = np.array([len(neighbors[i]) for i in range(n)], dtype=np.int32)

    def compute_cut(labels):
        """Compute the cut size for a given labeling."""
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff)) // 2

    def compute_same_count(labels):
        """For each vertex, count neighbors on the same side."""
        same_count = np.zeros(n, dtype=np.int32)
        for v in range(n):
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[v] += 1
        return same_count

    def flip_greedy(labels, max_passes=80):
        """Fast 1-flip greedy using incremental cut tracking with randomized order."""
        labels = labels.copy()
        same_count = compute_same_count(labels)

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            changed = False
            order = list(range(n))
            random.shuffle(order)
            for v in order:
                if time.time() > deadline:
                    break
                deg = degrees[v]
                gain = 2 * same_count[v] - deg
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    changed = True
                    old_same = same_count[v]
                    same_count[v] = deg - old_same
                    for u in neighbors[v]:
                        if labels[u] == labels[v]:
                            same_count[u] += 1
                        else:
                            same_count[u] -= 1
            if not changed:
                break
        return labels

    def cerny_tabu(labels, max_iters=6000, tabu_tenure=18):
        """Cerny tabu search: full-scan, accepts worsening moves, tabu with aspiration."""
        labels = labels.copy()
        same_count = compute_same_count(labels)
        current_cut = compute_cut(labels)
        best_cut = current_cut
        best_labels = labels.copy()

        # tabu_until[v] = iteration index at which v becomes non-tabu
        tabu_until = {}
        iteration = 0
        no_improve = 0
        stall_limit = max_iters // 4

        while iteration < max_iters:
            if time.time() > deadline:
                break

            # Find best move among all non-tabu vertices (aspiration: allow tabu if beats best)
            best_gain = -999999
            best_v = -1

            for v in range(n):
                gain = 2 * same_count[v] - degrees[v]
                is_tabu = (v in tabu_until) and (tabu_until[v] > iteration)
                if is_tabu:
                    # Aspiration: only allow if this move exceeds current best
                    if current_cut + gain > best_cut:
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                else:
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v

            if best_v < 0:
                break

            # Perform the flip
            v = best_v
            labels[v] = 1 - labels[v]
            tabu_until[v] = iteration + tabu_tenure

            old_same = same_count[v]
            same_count[v] = degrees[v] - old_same
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    same_count[u] += 1
                else:
                    same_count[u] -= 1

            current_cut += best_gain
            iteration += 1

            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > stall_limit:
                    break

        return best_labels

    def two_swap_search(labels, max_passes=12):
        """2-swap local search with incremental updates."""
        labels = labels.copy()
        same_count = compute_same_count(labels)

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            order = list(range(n))
            random.shuffle(order)

            for i in order:
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    if labels[i] == labels[j]:
                        continue

                    e_ij = adj[i, j]
                    # After swapping i and j:
                    new_same_i = degrees[i] - same_count[i] - e_ij
                    new_same_j = degrees[j] - same_count[j] - e_ij
                    delta_same = (new_same_i + new_same_j) - (same_count[i] + same_count[j])

                    if delta_same > 0:
                        labels[i], labels[j] = labels[j], labels[i]
                        improved = True

                        old_same_i = same_count[i]
                        old_same_j = same_count[j]
                        same_count[i] = new_same_i
                        same_count[j] = new_same_j

                        for u in neighbors[i]:
                            if u == j:
                                continue
                            if labels[u] == labels[i]:
                                same_count[u] += 1
                            else:
                                same_count[u] -= 1

                        for u in neighbors[j]:
                            if u == i:
                                continue
                            if labels[u] == labels[j]:
                                same_count[u] += 1
                            else:
                                same_count[u] -= 1

            if not improved:
                break

        return labels

    def spectral_partition(k=32, num_random=5):
        """Spectral partitioning with multiple random hyperplanes."""
        try:
            D = np.diag(degrees.astype(np.float64))
            L = D - adj.astype(np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(L)

            k = min(k, n - 1)
            embedding = eigenvectors[:, 1:k + 1]

            norms = np.linalg.norm(embedding, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embedding = embedding / norms

            best_labels = None
            best_cut = -1

            for _ in range(num_random):
                random_vec = np.random.randn(k)
                random_vec /= np.linalg.norm(random_vec)
                projection = embedding @ random_vec
                labels = (projection > 0).astype(np.int8)
                cut = compute_cut(labels)
                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels

            if best_labels is not None:
                return best_labels
        except Exception:
            pass
        return np.random.randint(0, 2, size=n, dtype=np.int8)

    def fiedler_partition():
        """Partition using the Fiedler vector (2nd eigenvector of Laplacian)."""
        try:
            D = np.diag(degrees.astype(np.float64))
            L = D - adj.astype(np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            fiedler = eigenvectors[:, 1]
            return (fiedler > 0).astype(np.int8)
        except Exception:
            return np.random.randint(0, 2, size=n, dtype=np.int8)

    def greedy_partition():
        """Greedy partition: assign vertices one by one to maximize cut."""
        labels = np.zeros(n, dtype=np.int8)
        order = np.argsort(-degrees)
        for v in order:
            count_0 = 0
            count_1 = 0
            for u in neighbors[v]:
                if labels[u] == 0:
                    count_0 += 1
                else:
                    count_1 += 1
            if count_0 < count_1:
                labels[v] = 1
            elif count_1 < count_0:
                labels[v] = 0
            else:
                labels[v] = random.randint(0, 1)
        return labels

    best_cut = -1
    best_labels = None

    def try_solution(labels):
        nonlocal best_cut, best_labels
        # 1-flip greedy
        labels1 = flip_greedy(labels, max_passes=60)
        cut1 = compute_cut(labels1)
        if cut1 > best_cut:
            best_cut = cut1
            best_labels = labels1.copy()

        # Cerny tabu (the key escape mechanism)
        if time.time() < deadline:
            labels2 = cerny_tabu(labels1, max_iters=4000, tabu_tenure=18)
            cut2 = compute_cut(labels2)
            if cut2 > best_cut:
                best_cut = cut2
                best_labels = labels2.copy()

        # 2-swap
        if time.time() < deadline:
            labels3 = two_swap_search(labels2, max_passes=10)
            cut3 = compute_cut(labels3)
            if cut3 > best_cut:
                best_cut = cut3
                best_labels = labels3.copy()

        # Another round: 1-flip + tabu + 2-swap on the result
        if time.time() < deadline:
            labels4 = flip_greedy(labels3, max_passes=40)
            labels5 = cerny_tabu(labels4, max_iters=2500, tabu_tenure=14)
            labels6 = two_swap_search(labels5, max_passes=8)
            cut6 = compute_cut(labels6)
            if cut6 > best_cut:
                best_cut = cut6
                best_labels = labels6.copy()

    # Generate diverse initial solutions
    init_solutions = []
    init_solutions.append(spectral_partition(k=32, num_random=5))
    init_solutions.append(spectral_partition(k=16, num_random=5))
    init_solutions.append(spectral_partition(k=8, num_random=5))
    init_solutions.append(fiedler_partition())
    init_solutions.append(greedy_partition())
    init_solutions.append(np.random.randint(0, 2, size=n, dtype=np.int8))
    init_solutions.append(np.arange(n) % 2)

    for init in init_solutions:
        if time.time() > deadline:
            break
        try_solution(init)

    # Additional random restarts
    num_restarts = 12
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        if r % 3 == 0:
            labels = np.random.randint(0, 2, size=n, dtype=np.int8)
        elif r % 3 == 1:
            labels = spectral_partition(k=16, num_random=1)
        else:
            labels = greedy_partition()

        try_solution(labels)

    # Final polish: multiple rounds of local search on best solution
    if best_labels is not None and time.time() < deadline:
        for _ in range(3):
            if time.time() > deadline:
                break
            labels = flip_greedy(best_labels, max_passes=40)
            labels = cerny_tabu(labels, max_iters=2000, tabu_tenure=12)
            labels = two_swap_search(labels, max_passes=10)
            cut = compute_cut(labels)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()

    if best_labels is None:
        return np.arange(n) % 2
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END

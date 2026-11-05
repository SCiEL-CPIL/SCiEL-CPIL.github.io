import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters: greedy set cover
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            ns = {i}
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    nw = list(w)
                    nw[pos] = val
                    ns.add(word_to_idx[tuple(nw)])
            neighbors.append(np.array(list(ns), dtype=int))
        uncovered = np.ones(N, dtype=bool)
        C = []
        t0 = time.time()
        while np.any(uncovered):
            best_i, best_c = -1, -1
            for i in range(N):
                c = int(np.sum(uncovered[neighbors[i]]))
                if c > best_c:
                    best_c, best_i = c, i
            if best_c == 0:
                break
            C.append(best_i)
            uncovered[neighbors[best_i]] = False
            if time.time() - t0 > 100:
                break
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill

    # Precompute all words and neighbors
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Build neighbor arrays
    neighbors = []
    for i, w in enumerate(all_words):
        ns = {i}
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                ns.add(word_to_idx[tuple(nw)])
        neighbors.append(np.array(sorted(ns), dtype=int))

    # Build neighbor sets for fast intersection
    neighbor_sets = [set(arr.tolist()) for arr in neighbors]

    def reduce_code(C, counts):
        """Iteratively remove codewords whose neighbors all have count >= 2."""
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(counts[neighbors[c]]) >= 2:
                    counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return C, counts

    def greedy_set_cover(uncovered, rng=None):
        """Greedy set cover: repeatedly pick the word covering the most uncovered points."""
        C = []
        while np.any(uncovered):
            # Compute coverage counts for all candidates
            # Vectorized approach: for each i, count uncovered[neighbors[i]]
            # Use numpy broadcasting trick
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = int(np.sum(uncovered[neighbors[i]]))
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0:
                break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit - 2:
                break
        return C

    def local_search(C, counts, time_budget):
        """Swap-based local search to improve the code."""
        best_C = C.copy()
        best_counts = counts.copy()
        best_size = len(C)
        local_start = time.time()

        while time.time() - local_start < time_budget:
            # Pass 1: Peeling
            C, counts = reduce_code(C, counts)
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()

            if time.time() - local_start > time_budget:
                break

            # Pass 2: Try swaps
            improved = False
            indices = list(range(len(C)))
            random.shuffle(indices)

            for idx in indices:
                if time.time() - local_start > time_budget:
                    break
                c = C[idx]
                u_indices = neighbors[c]
                unique_mask = counts[u_indices] == 1
                unique_points = u_indices[unique_mask]

                if len(unique_points) == 0:
                    C.remove(c)
                    counts[neighbors[c]] -= 1
                    improved = True
                    break

                if len(unique_points) > 3:
                    continue

                # Find candidates that cover all unique points
                if len(unique_points) == 1:
                    candidates_set = set(neighbor_sets[unique_points[0]])
                else:
                    candidates_set = set(neighbor_sets[unique_points[0]])
                    for u in unique_points[1:]:
                        candidates_set &= neighbor_sets[u]
                        if not candidates_set:
                            break

                if not candidates_set:
                    continue

                candidates_set.difference_update(C)
                if not candidates_set:
                    continue

                c_prime = random.choice(list(candidates_set))
                old_C = C.copy()
                old_counts = counts.copy()

                C.remove(c)
                C.append(c_prime)
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1

                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue

                new_C, new_counts = reduce_code(C, counts)
                if len(new_C) < len(old_C):
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    improved = True
                    break
                else:
                    C = new_C
                    counts = new_counts
                    improved = True
                    break

            if not improved:
                # Random kicks
                for _ in range(5):
                    if time.time() - local_start > time_budget:
                        break
                    if not C:
                        break
                    c_idx = random.randint(0, len(C) - 1)
                    c = C[c_idx]
                    cand = neighbors[c]
                    c_prime = int(cand[random.randint(0, len(cand) - 1)])
                    if c_prime in C:
                        continue

                    old_C = C.copy()
                    old_counts = counts.copy()
                    C.remove(c)
                    C.append(c_prime)
                    counts[neighbors[c]] -= 1
                    counts[neighbors[c_prime]] += 1

                    if np.any(counts < 1):
                        C = old_C
                        counts = old_counts
                        continue

                    new_C, new_counts = reduce_code(C, counts)
                    if len(new_C) < len(old_C):
                        C = new_C
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        break
                    else:
                        C = new_C
                        counts = new_counts

        return best_C, best_counts

    # Main loop: deterministic greedy, then local search, then randomized restarts
    best_C = None
    best_counts = None
    best_size = float('inf')

    # First run: deterministic greedy
    uncovered = np.ones(N, dtype=bool)
    C = greedy_set_cover(uncovered)

    counts = np.zeros(N, dtype=int)
    for c in C:
        counts[neighbors[c]] += 1

    # Ensure coverage
    if np.any(counts < 1):
        while np.any(counts < 1) and time.time() - start_time < time_limit:
            uncovered_mask = counts < 1
            best_idx, best_count = -1, -1
            for i in range(N):
                cnt = int(np.sum(uncovered_mask[neighbors[i]]))
                if cnt > best_count:
                    best_count, best_idx = cnt, i
            if best_count == 0:
                break
            C.append(best_idx)
            counts[neighbors[best_idx]] += 1

    C, counts = reduce_code(C, counts)
    if len(C) < best_size:
        best_size = len(C)
        best_C = C.copy()
        best_counts = counts.copy()

    # Local search on the first solution
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        best_C, best_counts = local_search(best_C, best_counts, remaining_time * 0.6)
        if len(best_C) < best_size:
            best_size = len(best_C)

    # Multiple randomized restarts
    remaining_time = time_limit - (time.time() - start_time)
    num_restarts = min(10, max(1, int(remaining_time / 15)))
    rng = random.Random(42)

    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 5:
            break

        # Randomized greedy
        uncovered = np.ones(N, dtype=bool)
        C = []
        while np.any(uncovered):
            best_idx, best_count = -1, -1
            for i in range(N):
                cnt = int(np.sum(uncovered[neighbors[i]]))
                if cnt > best_count:
                    best_count, best_idx = cnt, i
            if best_count == 0:
                break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit - 3:
                break

        counts = np.zeros(N, dtype=int)
        for c in C:
            counts[neighbors[c]] += 1

        if np.any(counts < 1):
            while np.any(counts < 1) and time.time() - start_time < time_limit - 3:
                uncovered_mask = counts < 1
                best_idx, best_count = -1, -1
                for i in range(N):
                    cnt = int(np.sum(uncovered_mask[neighbors[i]]))
                    if cnt > best_count:
                        best_count, best_idx = cnt, i
                if best_count == 0:
                    break
                C.append(best_idx)
                counts[neighbors[best_idx]] += 1

        C, counts = reduce_code(C, counts)
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()

        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 10 and len(C) <= best_size + 5:
            C, counts = local_search(C, counts, remaining_time * 0.5)
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()

    # Final reduction
    if best_C is not None:
        final_counts = np.zeros(N, dtype=int)
        for c in best_C:
            final_counts[neighbors[c]] += 1
        best_C, final_counts = reduce_code(best_C, final_counts)

    if best_C is None:
        best_C = list(range(N))

    return [list(all_words[i]) for i in best_C]

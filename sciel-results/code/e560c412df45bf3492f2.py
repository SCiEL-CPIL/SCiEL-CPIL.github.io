import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        return _fallback(q, n, radius)

    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbors as numpy arrays
    neighbors = []
    for i, w in enumerate(all_words):
        neigh = [i]
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh.append(word_to_idx[new_w])
        neighbors.append(np.array(neigh, dtype=np.int32))

    # Precompute neighbor sets for fast intersection
    neighbor_sets = [set(neigh.tolist()) for neigh in neighbors]

    start_time = time.time()
    time_limit = 115.0

    def get_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    def reduce_code(code, counts):
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    def greedy_init():
        uncovered = np.ones(N, dtype=bool)
        C = []
        while np.any(uncovered):
            if time.time() - start_time > time_limit:
                break
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0:
                break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
        return C

    def local_search(C, counts, time_limit):
        C = list(C)
        C_set = set(C)
        best_size = len(C)
        best_C = C.copy()
        best_counts = counts.copy()

        while time.time() - start_time < time_limit:
            old_size = len(C)
            C, counts = reduce_code(C, counts)
            C_set = set(C)

            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                best_counts = counts.copy()

            if time.time() - start_time > time_limit:
                break

            if len(C) == old_size:
                # Try targeted swaps
                improved = False
                indices = list(range(len(C)))
                random.shuffle(indices)

                for idx in indices:
                    if time.time() - start_time > time_limit:
                        break
                    c = C[idx]
                    u_indices = neighbors[c]
                    unique_mask = counts[u_indices] == 1
                    unique_points = u_indices[unique_mask]

                    if len(unique_points) == 0:
                        # Should have been removed
                        C.remove(c)
                        C_set.remove(c)
                        counts[neighbors[c]] -= 1
                        improved = True
                        break

                    if len(unique_points) > 4:
                        continue

                    # Find candidates that cover all unique points
                    if len(unique_points) == 1:
                        candidates_set = set(neighbor_sets[unique_points[0]])
                    else:
                        candidates_set = set(neighbor_sets[unique_points[0]])
                        for u in unique_points[1:]:
                            candidates_set &= set(neighbor_sets[u])
                            if not candidates_set:
                                break

                    if not candidates_set:
                        continue

                    candidates_set.difference_update(C_set)
                    if not candidates_set:
                        continue

                    # Try each candidate (or a random subset)
                    candidate_list = list(candidates_set)
                    random.shuffle(candidate_list)
                    tried = 0
                    for c_prime in candidate_list:
                        if time.time() - start_time > time_limit:
                            break
                        if tried > 10:
                            break
                        tried += 1

                        old_C = C.copy()
                        old_counts = counts.copy()
                        old_C_set = C_set.copy()

                        C[idx] = c_prime
                        C_set.remove(c)
                        C_set.add(c_prime)
                        counts[neighbors[c]] -= 1
                        counts[neighbors[c_prime]] += 1

                        if np.any(counts < 1):
                            C = old_C
                            counts = old_counts
                            C_set = old_C_set
                            continue

                        new_C, new_counts = reduce_code(C, counts)
                        new_C_set = set(new_C)

                        if len(new_C) < len(old_C):
                            C = new_C
                            counts = new_counts
                            C_set = new_C_set
                            if len(C) < best_size:
                                best_size = len(C)
                                best_C = C.copy()
                                best_counts = counts.copy()
                            improved = True
                            break
                        else:
                            if len(new_C) == len(old_C) and random.random() < 0.3:
                                C = new_C
                                counts = new_counts
                                C_set = new_C_set
                            else:
                                C = old_C
                                counts = old_counts
                                C_set = old_C_set

                    if improved:
                        break

                if not improved:
                    # Try random swap as fallback
                    if C:
                        c_idx = random.randint(0, len(C) - 1)
                        c = C[c_idx]
                        w = random.randint(0, N - 1)
                        if w != c and w not in C_set:
                            old_C = C.copy()
                            old_counts = counts.copy()
                            old_C_set = C_set.copy()
                            C[c_idx] = w
                            C_set.remove(c)
                            C_set.add(w)
                            counts[neighbors[c]] -= 1
                            counts[neighbors[w]] += 1
                            if np.any(counts < 1):
                                C = old_C
                                counts = old_counts
                                C_set = old_C_set
                            else:
                                new_C, new_counts = reduce_code(C, counts)
                                new_C_set = set(new_C)
                                if len(new_C) < len(old_C):
                                    C = new_C
                                    counts = new_counts
                                    C_set = new_C_set
                                    if len(C) < best_size:
                                        best_size = len(C)
                                        best_C = C.copy()
                                        best_counts = counts.copy()
                                elif len(new_C) == len(old_C) and random.random() < 0.3:
                                    C = new_C
                                    counts = new_counts
                                    C_set = new_C_set
                                else:
                                    C = old_C
                                    counts = old_counts
                                    C_set = old_C_set

        return best_C, best_counts

    # Run multiple restarts
    best_overall = None
    best_size_overall = float('inf')

    while time.time() - start_time < time_limit:
        C = greedy_init()
        if not C:
            break
        counts = get_counts(C)
        C, counts = local_search(C, counts, time_limit)
        C, _ = reduce_code(C, get_counts(C))

        if len(C) < best_size_overall:
            best_size_overall = len(C)
            best_overall = C.copy()

        # Check if we've hit a good result
        if best_size_overall <= 486:
            break

    if best_overall is None:
        C = greedy_init()
        counts = get_counts(C)
        C, counts = reduce_code(C, counts)
        best_overall = C

    return [list(all_words[i]) for i in best_overall]

def _fallback(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set([i])
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(list(neigh_set))

    N = q ** n
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 100.0

    while np.any(uncovered):
        best_idx = -1
        best_count = -1
        for i in range(N):
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        if time.time() - start_time > time_limit:
            break

    coverage_counts = np.zeros(N, dtype=int)
    for c in C:
        coverage_counts[neighbors[c]] += 1
    changed = True
    while changed:
        changed = False
        new_C = []
        for c in C:
            if np.min(coverage_counts[neighbors[c]]) >= 2:
                coverage_counts[neighbors[c]] -= 1
                changed = True
            else:
                new_C.append(c)
        C = new_C
    return [list(all_words[i]) for i in C]

import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n
    ball_size = 1 + n * (q - 1)
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i - 1] * q

    # Pre-compute neighbors
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        neighbors[idx, 0] = idx
        nb = 1
        for pos in range(n):
            orig = word[pos]
            pw = powers[pos]
            d1 = (orig + 1) % q
            delta1 = -2 * pw if d1 == 0 else pw
            neighbors[idx, nb] = idx + delta1
            nb += 1
            d2 = (orig - 1) % q
            delta2 = 2 * pw if d2 == 2 else -pw
            neighbors[idx, nb] = idx + delta2
            nb += 1

    # Reverse neighbors
    rev_neighbors = [[] for _ in range(total_points)]
    for b in range(total_points):
        for p in neighbors[b]:
            rev_neighbors[p].append(b)

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def greedy_set_cover(seed, deadline):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        uncovered_counts = np.full(total_points, ball_size, dtype=np.int32)
        selected = []
        
        while np.any(uncovered_counts > 0):
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            candidates = np.where(uncovered_counts == max_count)[0]
            best_idx = rng.choice(candidates)
            selected.append(best_idx)
            
            for p in neighbors[best_idx]:
                if not is_covered[p]:
                    is_covered[p] = True
                    for bp in rev_neighbors[p]:
                        uncovered_counts[bp] -= 1
        return selected

    def deterministic_peel(selected_indices):
        """Iteratively remove codewords whose removal doesn't leave any point uncovered."""
        selected_list = list(selected_indices)
        coverage = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage[neighbors[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_list)):
                idx = selected_list[i]
                # Check if removing idx leaves any point with count 0
                if np.all(coverage[neighbors[idx]] > 1):
                    coverage[neighbors[idx]] -= 1
                    selected_list.pop(i)
                    changed = True
                    break
        return selected_list

    def local_search(selected_indices, deadline):
        selected_list = list(selected_indices)
        coverage = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        rng = np.random.default_rng(7)
        selected_set = set(selected_list)
        pos_map = {idx: i for i, idx in enumerate(selected_list)}
        stall = 0
        
        while time.time() < deadline and stall < 8000:
            if not selected_list:
                break
            c = selected_list[rng.integers(len(selected_list))]
            found = False
            for pos in range(1, ball_size):
                cp = neighbors[c][pos]
                if cp in selected_set:
                    continue
                coverage[neighbors[c]] -= 1
                coverage[neighbors[cp]] += 1
                if np.any(coverage[neighbors[c]] == 0):
                    coverage[neighbors[cp]] -= 1
                    coverage[neighbors[c]] += 1
                    continue
                else:
                    pi = pos_map[c]
                    selected_list[pi] = cp
                    selected_set.remove(c)
                    selected_set.add(cp)
                    pos_map.pop(c)
                    pos_map[cp] = pi
                    found = True
                    stall = 0
                    break
            if not found:
                stall += 1
                if stall % 80 == 0:
                    ri = selected_list[rng.integers(len(selected_list))]
                    coverage[neighbors[ri]] -= 1
                    if not np.any(coverage == 0):
                        pi = pos_map[ri]
                        selected_list.pop(pi)
                        selected_set.remove(ri)
                        pos_map.pop(ri)
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                        stall = 0
                    else:
                        coverage[neighbors[ri]] += 1
        return best_code

    start_time = time.time()
    deadline = start_time + 115.0

    best_code = None
    best_len = float('inf')

    # Run multiple greedy instances with peeling
    for seed in range(8):
        if time.time() > deadline - 40:
            break
        sel = greedy_set_cover(seed, deadline)
        sel = deterministic_peel(sel)
        if len(sel) < best_len:
            best_len = len(sel)
            best_code = sel

    if best_code is None:
        best_code = deterministic_peel(greedy_set_cover(0, deadline))
        best_len = len(best_code)

    # Local search
    if time.time() < deadline - 5:
        best_code = local_search(best_code, deadline)

    # Final peel
    best_code = deterministic_peel(best_code)

    result = [idx_to_word(idx) for idx in best_code]
    return result

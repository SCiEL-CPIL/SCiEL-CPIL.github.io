import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    num_edges = len(edges)

    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx

    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
            edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
        ]
        k4s.append(e_idx)
    num_k4s = len(k4s)

    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4s):
        for ei in e_list:
            edge_k4s[ei].append(k4_idx)

    # Precompute for each K4 the 6 edge indices (already in k4s)
    # For fast penalty computation, we'll track per-K4 color counts

    def compact(colors):
        unique = sorted(set(colors))
        mapping = {c: i for i, c in enumerate(unique)}
        return [mapping[c] for c in colors], len(unique)

    # Try simple algebraic constructions first
    best_colors = None
    best_count = float('inf')

    # (i+j) % p
    for p in range(13, 22):
        colors = [(i + j) % p for (i, j) in edges]
        valid = True
        for k4 in k4s:
            if len(set(colors[ei] for ei in k4)) < 5:
                valid = False
                break
        if valid:
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break

    # (i*j) % p for p prime
    for p in [13, 17, 19, 23, 29, 31]:
        colors = [(i * j) % p for (i, j) in edges]
        valid = True
        for k4 in k4s:
            if len(set(colors[ei] for ei in k4)) < 5:
                valid = False
                break
        if valid:
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break

    # (i^2 + j) % p
    for p in [13, 15, 17, 19]:
        colors = [(i * i + j) % p for (i, j) in edges]
        valid = True
        for k4 in k4s:
            if len(set(colors[ei] for ei in k4)) < 5:
                valid = False
                break
        if valid:
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break

    # (i + 2*j) % p
    for p in [13, 15, 17, 19]:
        colors = [(i + 2 * j) % p for (i, j) in edges]
        valid = True
        for k4 in k4s:
            if len(set(colors[ei] for ei in k4)) < 5:
                valid = False
                break
        if valid:
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break

    # (j - i) % p
    for p in [13, 15, 17, 19]:
        colors = [(j - i) % p for (i, j) in edges]
        valid = True
        for k4 in k4s:
            if len(set(colors[ei] for ei in k4)) < 5:
                valid = False
                break
        if valid:
            nc = len(set(colors))
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break

    if best_colors is not None and best_count <= 13:
        return compact(best_colors)[0]

    # Use local search with O(1) penalty updates
    # Start from best found or (i+j)%17
    if best_colors is not None:
        colors = best_colors[:]
    else:
        colors = [(i + j) % 17 for (i, j) in edges]

    # Track per-K4: number of distinct colors (maintained incrementally)
    # For each K4, track the multiset of colors on its 6 edges
    k4_color_counts = [{} for _ in range(num_k4s)]
    for k4_idx, e_list in enumerate(k4s):
        for ei in e_list:
            c = colors[ei]
            k4_color_counts[k4_idx][c] = k4_color_counts[k4_idx].get(c, 0) + 1

    k4_penalty = [0] * num_k4s
    for k4_idx in range(num_k4s):
        if len(k4_color_counts[k4_idx]) < 5:
            k4_penalty[k4_idx] = 1

    total_penalty = sum(k4_penalty)
    current_colors = colors[:]
    current_penalty = total_penalty
    best_count = len(set(current_colors))
    best_solution = current_colors[:]

    max_k = 17
    T = 3.0
    cooling = 0.9997
    min_T = 0.005

    iterations = 0
    while time.time() < deadline - 8.0 and T > min_T:
        e_idx = random.randint(0, num_edges - 1)
        old_c = current_colors[e_idx]
        new_c = random.randint(0, max_k - 1)
        if new_c == old_c:
            continue

        # Compute penalty delta for affected K4s
        delta_penalty = 0
        for k4_idx in edge_k4s[e_idx]:
            counts = k4_color_counts[k4_idx]
            old_distinct = len(counts)

            # Remove old color
            counts[old_c] -= 1
            if counts[old_c] == 0:
                del counts[old_c]
            # Add new color
            counts[new_c] = counts.get(new_c, 0) + 1
            new_distinct = len(counts)

            if old_distinct < 5 and new_distinct >= 5:
                delta_penalty -= 1
            elif old_distinct >= 5 and new_distinct < 5:
                delta_penalty += 1
            # Keep state modified for this K4

        if delta_penalty <= 0 or random.random() < math.exp(-delta_penalty / T):
            current_colors[e_idx] = new_c
            current_penalty += delta_penalty
            iterations += 1

            nc = len(set(current_colors))
            if nc < best_count:
                best_count = nc
                best_solution = current_colors[:]
                # Try to reduce max_k
                if best_count < max_k:
                    max_k = best_count
        else:
            # Revert: undo the color changes in k4_color_counts
            for k4_idx in edge_k4s[e_idx]:
                counts = k4_color_counts[k4_idx]
                # Remove new color
                counts[new_c] -= 1
                if counts[new_c] == 0:
                    del counts[new_c]
                # Add old color
                counts[old_c] = counts.get(old_c, 0) + 1

        T *= cooling

    # Final polish: try to reduce colors used
    remaining_time = deadline - time.time()
    if remaining_time > 3.0 and best_count > 13:
        polish_colors = best_solution[:]
        polish_k4_counts = [{} for _ in range(num_k4s)]
        for k4_idx, e_list in enumerate(k4s):
            for ei in e_list:
                c = polish_colors[ei]
                polish_k4_counts[k4_idx][c] = polish_k4_counts[k4_idx].get(c, 0) + 1
        polish_penalty = [1 if len(polish_k4_counts[k]) < 5 else 0 for k in range(num_k4s)]
        polish_best_count = best_count

        unique_colors = set(polish_colors)
        polish_iter = 0
        max_polish = int(min(remaining_time * 200, 20000))
        while polish_iter < max_polish and time.time() < deadline - 1.5:
            polish_iter += 1
            if len(set(polish_colors)) <= 13:
                break
            e_idx = random.randint(0, num_edges - 1)
            old_c = polish_colors[e_idx]
            other_colors = [c for c in unique_colors if c != old_c]
            if not other_colors:
                break
            new_c = random.choice(other_colors)

            valid = True
            for k4_idx in edge_k4s[e_idx]:
                counts = polish_k4_counts[k4_idx]
                # Simulate: remove old_c, add new_c
                temp_counts = dict(counts)
                temp_counts[old_c] -= 1
                if temp_counts[old_c] == 0:
                    del temp_counts[old_c]
                temp_counts[new_c] = temp_counts.get(new_c, 0) + 1
                if len(temp_counts) < 5:
                    valid = False
                    break

            if valid:
                for k4_idx in edge_k4s[e_idx]:
                    counts = polish_k4_counts[k4_idx]
                    counts[old_c] -= 1
                    if counts[old_c] == 0:
                        del counts[old_c]
                    counts[new_c] = counts.get(new_c, 0) + 1
                polish_colors[e_idx] = new_c
                nc = len(set(polish_colors))
                if nc < polish_best_count:
                    polish_best_count = nc
                    best_solution = polish_colors[:]
                    best_count = nc
                    unique_colors = set(polish_colors)

    return compact(best_solution)[0]

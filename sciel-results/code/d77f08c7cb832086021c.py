import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit

    n = 16
    num_edges = n * (n - 1) // 2  # 120
    num_k4 = 1820  # C(16, 4)

    # Precompute edge indices
    edge_indices = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_indices[(i, j)] = idx
            idx += 1

    # Precompute K4s
    k4_list = []
    edge_to_k4s = [[] for _ in range(num_edges)]

    k_idx = 0
    for quad in itertools.combinations(range(n), 4):
        i, j, k, l = quad
        edges = [
            edge_indices[(i, j)],
            edge_indices[(i, k)],
            edge_indices[(i, l)],
            edge_indices[(j, k)],
            edge_indices[(j, l)],
            edge_indices[(k, l)]
        ]
        k4_list.append(edges)
        for e in edges:
            edge_to_k4s[e].append(k_idx)
        k_idx += 1

    # Precompute for each K4, the position of each edge within that K4
    # k4_edge_pos[k4_idx][edge_idx_in_k4] = edge_index
    # Actually we already have k4_list[k4_idx] = list of 6 edge indices

    # Precompute: for each edge e, for each k4 containing e, what's the position of e in that k4
    # edge_pos_in_k4[e] = list of (k4_idx, position)
    edge_pos_in_k4 = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_list):
        for pos, e in enumerate(edges):
            edge_pos_in_k4[e].append((k4_idx, pos))

    # Color count per K4: k4_color_counts[k4_idx] = [count_c0, count_c1, ..., count_cK-1]
    # But K varies. We'll use a fixed max color value.
    MAX_C = 16  # colors 0..15

    def init_color_counts(colors, K):
        """Initialize k4_color_counts for given colors and K."""
        counts = []
        for edges in k4_list:
            cnt = [0] * K
            for e in edges:
                c = colors[e]
                if c < K:
                    cnt[c] += 1
            counts.append(cnt)
        return counts

    def count_violations_from_counts(counts, K):
        """Count K4s with < 5 distinct colors."""
        v = 0
        for cnt in counts:
            distinct = sum(1 for x in cnt if x > 0)
            if distinct < 5:
                v += 1
        return v

    def is_valid(colors):
        for edges in k4_list:
            if len({colors[e] for e in edges}) < 5:
                return False
        return True

    def count_violations_full(colors):
        v = 0
        for edges in k4_list:
            if len({colors[e] for e in edges}) < 5:
                v += 1
        return v

    # Seed solution: (i+j) % 17
    seed_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            seed_colors.append((i + j) % 17)

    if is_valid(seed_colors):
        best_result = seed_colors
        best_k = 17
    else:
        # Shouldn't happen, but fallback
        best_result = [0] * num_edges
        best_k = 1

    # Try to find better solutions
    # Strategy: For each target K from 13 upward, try to find valid coloring with K colors
    # Use incremental local search with color penalty

    for target_K in range(13, best_k):
        if time.time() > deadline - 15:
            break

        found_valid = False
        max_restarts = 30
        restarts = 0

        while restarts < max_restarts and not found_valid and time.time() < deadline - 10:
            # Initialize: random colors in 0..target_K-1
            colors = [random.randint(0, target_K - 1) for _ in range(num_edges)]

            # Initialize violation counts
            k4_counts = init_color_counts(colors, target_K)
            current_violations = count_violations_from_counts(k4_counts, target_K)
            best_local_violations = current_violations
            best_local_colors = colors[:]

            # Simulated annealing with incremental updates
            T = 2.0
            cooling = 0.9995
            step = 0
            max_steps = 50000

            while T > 0.005 and step < max_steps and time.time() < deadline - 5:
                # Pick random edge
                e = random.randint(0, num_edges - 1)
                old_c = colors[e]
                new_c = random.randint(0, target_K - 1)

                if old_c == new_c:
                    step += 1
                    T *= cooling
                    continue

                # Calculate delta violations incrementally
                delta_v = 0
                for k4_idx, pos in edge_pos_in_k4[e]:
                    cnt = k4_counts[k4_idx]
                    old_cnt = cnt[old_c]
                    new_cnt = cnt[new_c]

                    # Before change: distinct = number of non-zero entries
                    # After change: decrement old_c count, increment new_c count
                    # If old_cnt == 1, distinct decreases by 1
                    # If new_cnt == 0, distinct increases by 1

                    distinct_before = sum(1 for x in cnt if x > 0)
                    # Apply change
                    cnt[old_c] -= 1
                    cnt[new_c] += 1
                    distinct_after = sum(1 for x in cnt if x > 0)

                    was_violated = distinct_before < 5
                    is_violated = distinct_after < 5

                    if was_violated and not is_violated:
                        delta_v -= 1
                    elif not was_violated and is_violated:
                        delta_v += 1

                new_violations = current_violations + delta_v

                # Also compute color usage penalty to encourage using fewer colors
                # Count how many colors are actually used
                # This is expensive, so we'll use a simpler approach:
                # Just minimize violations first, then in a second phase minimize colors

                if new_violations < current_violations:
                    current_violations = new_violations
                    if new_violations < best_local_violations:
                        best_local_violations = new_violations
                        best_local_colors = colors[:]
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < pow(2.718281828, -delta_v / T):
                        current_violations = new_violations
                    else:
                        # Revert
                        for k4_idx, pos in edge_pos_in_k4[e]:
                            cnt = k4_counts[k4_idx]
                            cnt[old_c] += 1
                            cnt[new_c] -= 1
                        colors[e] = old_c
                        step += 1
                        T *= cooling
                        continue

                colors[e] = new_c
                step += 1
                T *= cooling

                if best_local_violations == 0:
                    # Found valid coloring!
                    distinct = len(set(best_local_colors))
                    if distinct < best_k:
                        best_k = distinct
                        best_result = best_local_colors[:]
                    found_valid = True
                    break

            restarts += 1

        if found_valid:
            if best_k <= target_K:
                # We found a valid coloring with <= target_K colors
                # Can we do better? Try to reduce colors further
                # For now, continue to next target_K (which is larger, so we won't improve)
                # Actually, if we found valid with target_K, we should break since we're going up
                # Wait, we're iterating target_K from 13 upward. If we found valid at 13, that's best.
                if best_k == 13:
                    break
                # If found at 14, try 13 next? No, 13 was already tried.
                # The loop goes 13, 14, 15, 16. If 13 failed but 14 succeeded, we have 14.
                # We could try to reduce the 14-color solution to 13 colors in a post-processing step.
                # For now, accept it.
                continue

    # Post-processing: try to reduce colors in the best solution
    if best_result is not None and best_k > 13:
        if time.time() < deadline - 10:
            # Try to eliminate one color at a time
            current_colors = best_result[:]
            current_k = best_k

            for remove_c in range(current_k - 1, -1, -1):
                if time.time() > deadline - 5:
                    break

                # Try to recolor all edges with color remove_c to other colors
                # and check if still valid
                # Use local search to fix violations
                new_colors = current_colors[:]
                # First, recolor all edges with remove_c to random other colors
                other_colors = [c for c in range(current_k) if c != remove_c]
                for e in range(num_edges):
                    if new_colors[e] == remove_c:
                        new_colors[e] = random.choice(other_colors)

                # Check validity
                if is_valid(new_colors):
                    # Great, we reduced by 1
                    current_colors = new_colors
                    current_k -= 1
                    if current_k < best_k:
                        best_k = current_k
                        best_result = current_colors[:]
                    break
                else:
                    # Try local search to fix
                    k4_counts = init_color_counts(new_colors, current_k)
                    v = count_violations_from_counts(k4_counts, current_k)
                    T = 1.0
                    for step in range(20000):
                        if time.time() > deadline - 3:
                            break
                        if v == 0:
                            break
                        e = random.randint(0, num_edges - 1)
                        old_c = new_colors[e]
                        new_c = random.randint(0, current_k - 1)
                        if old_c == new_c:
                            continue

                        delta_v = 0
                        for k4_idx, pos in edge_pos_in_k4[e]:
                            cnt = k4_counts[k4_idx]
                            distinct_before = sum(1 for x in cnt if x > 0)
                            cnt[old_c] -= 1
                            cnt[new_c] += 1
                            distinct_after = sum(1 for x in cnt if x > 0)
                            was_violated = distinct_before < 5
                            is_violated = distinct_after < 5
                            if was_violated and not is_violated:
                                delta_v -= 1
                            elif not was_violated and is_violated:
                                delta_v += 1

                        new_v = v + delta_v
                        if new_v < v:
                            v = new_v
                            new_colors[e] = new_c
                        else:
                            if random.random() < pow(2.718281828, -delta_v / T):
                                v = new_v
                                new_colors[e] = new_c
                            else:
                                for k4_idx, pos in edge_pos_in_k4[e]:
                                    cnt = k4_counts[k4_idx]
                                    cnt[old_c] += 1
                                    cnt[new_c] -= 1

                        T *= 0.9995

                    if v == 0:
                        current_colors = new_colors
                        current_k -= 1
                        if current_k < best_k:
                            best_k = current_k
                            best_result = current_colors[:]
                        break

    # Final check: ensure best_result is valid
    if best_result is not None and not is_valid(best_result):
        # Fallback to seed
        if is_valid(seed_colors):
            best_result = seed_colors
            best_k = 17

    return best_result

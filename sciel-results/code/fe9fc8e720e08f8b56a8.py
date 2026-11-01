import time
import itertools
import random

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"

    # Standard Graham-Lagarias construction to get perms of 1..n-1
    def get_word_and_perms(m):
        if m == 1:
            return [1], [(1,)]
        if m == 2:
            return [1, 2, 1], [(1, 2), (2, 1)]

        prev_word, prev_perms = get_word_and_perms(m - 1)

        new_word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]

        # Extract unique perms of length m from new_word
        perms = []
        seen = set()
        for i in range(len(new_word) - m + 1):
            window = tuple(new_word[i:i + m])
            if len(set(window)) == m and window not in seen:
                seen.add(window)
                perms.append(window)

        return new_word, perms

    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin

    # Get perms of 1..n-1
    _, perms = get_word_and_perms(n - 1)
    num_perms = len(perms)

    # Precompute overlap matrix
    max_k = n - 2
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]

    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            for k in range(max_k, -1, -1):
                if pi[-k:] == pj[:k]:
                    overlap_matrix[i][j] = k
                    break

    # Block length for each permutation block
    block_len = 2 * (n - 1) + 1

    # Initial order
    order = list(range(num_perms))

    # Compute initial length
    def calc_total_length(ord_list):
        total = block_len
        for i in range(len(ord_list) - 1):
            u = ord_list[i]
            v = ord_list[i + 1]
            total += block_len - overlap_matrix[u][v]
        return total

    best_length = calc_total_length(order)
    best_order = order[:]

    # Incremental length calculation for swaps
    def get_swap_delta(order, i):
        """Return length change if we swap positions i and i+1"""
        u, v = order[i], order[i + 1]
        delta = 0

        # Remove edges (prev, u), (u, v), (v, next)
        # Add edges (prev, v), (v, u), (u, next)

        # Edge before u
        if i > 0:
            prev = order[i - 1]
            delta += overlap_matrix[prev][u]  # remove
            delta -= overlap_matrix[prev][v]  # add

        # Edge u->v becomes v->u
        delta += overlap_matrix[u][v]  # remove
        delta -= overlap_matrix[v][u]  # add

        # Edge after v
        if i + 2 < len(order):
            nxt = order[i + 2]
            delta += overlap_matrix[v][nxt]  # remove
            delta -= overlap_matrix[u][nxt]  # add

        # delta is change in total overlap, so length change is -delta
        return -delta

    # Local search with adjacent swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(order) - 1):
            delta = get_swap_delta(order, i)
            if delta < 0:  # length decreases
                # Perform swap
                order[i], order[i + 1] = order[i + 1], order[i]
                best_length += delta
                best_order = order[:]
                improved = True
            # If delta >= 0, no swap needed (order unchanged)

    # 2-opt: try reversing segments
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(order) - 2):
                for j in range(i + 2, min(i + 50, len(order))):  # Limit segment length
                    # Calculate length change for reversing order[i..j]
                    # Before: ... a, order[i], order[i+1], ..., order[j], b ...
                    # After:  ... a, order[j], order[j-1], ..., order[i], b ...

                    a = order[i - 1] if i > 0 else None
                    b = order[j + 1] if j + 1 < len(order) else None

                    old_overlap = 0
                    new_overlap = 0

                    # Old: edges within segment + boundary edges
                    if a is not None:
                        old_overlap += overlap_matrix[a][order[i]]
                    for k in range(i, j):
                        old_overlap += overlap_matrix[order[k]][order[k + 1]]
                    if b is not None:
                        old_overlap += overlap_matrix[order[j]][b]

                    # New: reversed segment
                    if a is not None:
                        new_overlap += overlap_matrix[a][order[j]]
                    for k in range(i, j):
                        new_overlap += overlap_matrix[order[j - k + i]][order[j - k + i + 1]]
                    if b is not None:
                        new_overlap += overlap_matrix[order[i]][b]

                    delta = new_overlap - old_overlap  # positive means better (more overlap)
                    if delta > 0:
                        # Reverse segment
                        order[i:j + 1] = order[i:j + 1][::-1]
                        best_length -= delta
                        best_order = order[:]
                        improved = True

    # Construct final string
    final_word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(final_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if final_word[len(final_word) - o:] == block[:o]:
                overlap = o
                break
        final_word += block[overlap:]

    return "".join(map(str, final_word))

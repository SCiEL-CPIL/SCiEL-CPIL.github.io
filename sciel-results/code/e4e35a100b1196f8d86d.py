"""Ternary covering code K_3(7,1) – vectorized greedy + aggressive local polish."""
import time
import numpy as np


def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 140.0  # safety margin under 150s

    total = q ** n

    # Precompute strides
    strides = [q ** (n - 1 - p) for p in range(n)]

    # Precompute balls: for each word index, list of indices within Hamming distance <= radius
    # For radius 1: ball[i] = {i} union {i + (v - a)*strides[p] for p in range(n), a = (i//strides[p])%q, v != a}
    ball_list = []
    for i in range(total):
        neighbors = [i]
        for p in range(n):
            s = strides[p]
            a = (i // s) % q
            for v in range(q):
                if v == a:
                    continue
                neighbors.append(i + (v - a) * s)
        ball_list.append(np.array(neighbors, dtype=np.int32))

    # Build ball_matrix as 2D array (all balls have same size for uniform q, n, radius)
    ball_size = len(ball_list[0])
    ball_matrix = np.zeros((total, ball_size), dtype=np.int32)
    for v in range(total):
        ball_matrix[v] = ball_list[v]

    # Build reverse lookup: for each index, which words' balls contain it
    # This helps with swap search
    reverse_balls = [set() for _ in range(total)]
    for v in range(total):
        for c in ball_list[v]:
            reverse_balls[int(c)].add(v)

    # --- Greedy Set Cover ---
    covered = np.zeros(total, dtype=np.uint8)
    chosen = []
    chosen_set = set()

    # Initial gain: each word covers ball_size words
    init_gain = ball_size
    gain = np.full(total, init_gain, dtype=np.int32)

    # Use a simple approach: at each step, find the word with max gain
    # For efficiency, use a heap with lazy deletion
    import heapq
    heap = [(-init_gain, w) for w in range(total)]
    heapq.heapify(heap)

    while time.time() < DEADLINE - 30.0:
        if not heap:
            break

        neg_gain, w = heapq.heappop(heap)
        if w in chosen_set:
            continue

        # Verify current gain
        cur_gain = int(np.sum(covered[ball_matrix[w]] == 0))
        if cur_gain < 1:
            continue
        if cur_gain != -neg_gain:
            if cur_gain > 0:
                heapq.heappush(heap, (-cur_gain, w))
            continue

        # Commit word w
        chosen.append(w)
        chosen_set.add(w)
        covered[ball_matrix[w]] = 1

        # Update gains for all unchosen words
        # For each unchosen word v, gain[v] = count of uncovered words in ball[v]
        unchosen_mask = np.ones(total, dtype=bool)
        for c in chosen:
            unchosen_mask[c] = False

        # Recompute gains vectorized
        cov_expanded = covered[ball_matrix]  # shape (total, ball_size)
        new_gain = np.sum(cov_expanded == 0, axis=1)

        # Rebuild heap with updated gains
        heap = []
        for v in range(total):
            if v not in chosen_set and new_gain[v] > 0:
                heap.append((-int(new_gain[v]), v))
        heapq.heapify(heap)

    # Safety: complete cover if greedy didn't finish
    for w in range(total):
        if covered[w] == 0 and w not in chosen_set:
            chosen.append(w)
            chosen_set.add(w)
            covered[ball_matrix[w]] = 1

    # --- Local Search: Removal ---
    def _try_remove(chosen_list):
        if time.time() > DEADLINE - 5.0:
            return chosen_list

        cov_count = np.zeros(total, dtype=np.int32)
        for w in chosen_list:
            cov_count[ball_matrix[w]] += 1

        improved = True
        while improved and time.time() < DEADLINE - 5.0:
            improved = False
            for i in range(len(chosen_list) - 1, -1, -1):
                w = chosen_list[i]
                # Temporarily remove
                cov_count[ball_matrix[w]] -= 1
                if np.min(cov_count) >= 1:
                    chosen_list.pop(i)
                    improved = True
                    break
                else:
                    cov_count[ball_matrix[w]] += 1
        return chosen_list

    # --- Local Search: Swap ---
    def _try_swap(chosen_list):
        if time.time() > DEADLINE - 5.0:
            return chosen_list

        improved = True
        while improved and time.time() < DEADLINE - 5.0:
            improved = False

            cov_count = np.zeros(total, dtype=np.int32)
            for w in chosen_list:
                cov_count[ball_matrix[w]] += 1

            for i in range(len(chosen_list)):
                if time.time() > DEADLINE - 5.0:
                    break
                w_old = chosen_list[i]
                # Temporarily remove w_old
                cov_count[ball_matrix[w_old]] -= 1

                # Find uncovered words
                uncov = np.where(cov_count == 0)[0]

                if len(uncov) == 0:
                    # w_old is redundant
                    chosen_list.pop(i)
                    improved = True
                    break

                if len(uncov) <= 2:
                    # Try to find a replacement that covers all uncov words
                    # For each uncov word, get candidates from reverse_balls
                    candidates = set()
                    for u in uncov:
                        candidates.update(reverse_balls[int(u)])

                    # Filter out chosen words (except w_old which we're replacing)
                    chosen_set_local = set(chosen_list)
                    for w_new in candidates:
                        if w_new in chosen_set_local and w_new != w_old:
                            continue
                        # Check if w_new covers all uncov words
                        covers_all = all(u in ball_list[w_new] for u in uncov)
                        if covers_all:
                            chosen_list[i] = w_new
                            improved = True
                            break

                # Restore w_old
                cov_count[ball_matrix[w_old]] += 1

                if improved:
                    break

        return chosen_list

    # Apply local search
    chosen_list = list(chosen)

    # Multiple polish passes
    for _ in range(10):
        if time.time() > DEADLINE - 5.0:
            break
        before = len(chosen_list)
        chosen_list = _try_remove(chosen_list)
        chosen_list = _try_swap(chosen_list)
        chosen_list = _try_remove(chosen_list)
        if len(chosen_list) == before:
            break

    # Build output
    out = []
    seen = set()
    for w in chosen_list:
        codeword = list(np.unravel_index(w, (q,) * n))
        t = tuple(codeword)
        if t not in seen:
            seen.add(t)
            out.append(codeword)

    return out

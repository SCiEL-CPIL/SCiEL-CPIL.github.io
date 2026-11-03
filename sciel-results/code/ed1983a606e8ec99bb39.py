import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Constructs a covering code using greedy set cover, trimming, and
    simulated annealing local search with swap moves.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    total_words = q ** n

    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute balls using a tight Python loop
    balls_list = [None] * total_words
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                neighbors.append(digits_to_int(d, q))
            d[pos] = orig
        balls_list[i] = neighbors

    balls = np.array(balls_list, dtype=np.int32)

    # Precompute ball sets for fast subset checking
    ball_sets = [set(balls[i].tolist()) for i in range(total_words)]

    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    cover_count = np.zeros(total_words, dtype=np.int32)

    while np.any(uncovered) and time.time() < deadline:
        gains = uncovered[balls].sum(axis=1)
        best_center = int(np.argmax(gains))
        max_gain = gains[best_center]

        if max_gain == 0:
            break

        code.append(best_center)
        uncovered[balls[best_center]] = False
        cover_count[balls[best_center]] += 1

    # Trimming: remove redundant codewords
    def trim_code():
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                if np.any(cover_count[balls[c]] == 1):
                    continue
                to_remove.append(i)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True

    trim_code()

    # Simulated annealing local search
    def get_unique_words(c):
        return balls[c][cover_count[balls[c]] == 1]

    def local_search(seed, max_rounds=200):
        nonlocal code, cover_count
        random.seed(seed)
        code_set = set(code)
        all_indices = set(range(total_words))
        
        improved = True
        swap_round = 0
        temp = 1.0  # temperature for simulated annealing

        while improved and time.time() < deadline and swap_round < max_rounds:
            improved = False
            swap_round += 1
            temp *= 0.99  # cool down

            # Try removing each codeword first
            for ci in range(len(code) - 1, -1, -1):
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    code.pop(ci)
                    cover_count[balls[i]] -= 1
                    code_set.discard(i)
                    improved = True
                    break

            if improved:
                continue

            # Try swaps with simulated annealing
            code_indices = list(range(len(code)))
            random.shuffle(code_indices)

            for ci in code_indices:
                if time.time() > deadline:
                    return
                i = code[ci]
                unique = get_unique_words(i)
                if len(unique) == 0:
                    continue

                # Find candidates that cover all unique words
                non_code = list(all_indices - code_set)
                if not non_code:
                    break

                # Try a limited number of candidates
                n_candidates = min(300, len(non_code))
                random.shuffle(non_code)
                non_code = non_code[:n_candidates]

                best_swap = None
                best_gain = -1

                for j in non_code:
                    if time.time() > deadline:
                        return
                    j_ball_set = ball_sets[j]
                    if all(w in j_ball_set for w in unique):
                        # Calculate gain: how many words does j uniquely cover after swap?
                        # After removing i and adding j, check if j covers words not covered by others
                        new_cover = cover_count.copy()
                        new_cover[balls[i]] -= 1
                        new_cover[balls[j]] += 1
                        j_unique = balls[j][new_cover[balls[j]] == 1]
                        gain = len(j_unique)
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = j

                if best_swap is not None:
                    # Accept swap with probability based on gain and temperature
                    if best_gain >= 0 or random.random() < np.exp(best_gain / max(temp, 0.01)):
                        j = best_swap
                        code[ci] = j
                        code_set.discard(i)
                        code_set.add(j)
                        cover_count[balls[i]] -= 1
                        cover_count[balls[j]] += 1
                        improved = True
                        break

    # Run local search with different seeds
    for seed in [42, 123, 456]:
        if time.time() < deadline:
            local_search(seed, max_rounds=150)
            trim_code()

    # Final trimming
    trim_code()

    # Convert back to list of lists of integers
    result = []
    for c in code:
        result.append(int_to_digits(c, n, q))

    return result

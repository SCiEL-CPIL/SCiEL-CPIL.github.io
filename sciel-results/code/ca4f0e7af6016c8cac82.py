import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallbacks
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n

    # Precompute neighbors (balls) for all 6561 words
    weights = [3 ** (n - 1 - i) for i in range(n)]

    balls = np.zeros((total, 17), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Precompute ball membership matrix for O(1) lookup
    # ball_membership[w, v] = True if v is in balls[w]
    ball_membership = np.zeros((total, total), dtype=bool)
    for w in range(total):
        ball_membership[w, balls[w]] = True

    best_size = total
    best_set = None

    # Helper: Remove redundant codewords efficiently
    def remove_redundant(codewords):
        if not codewords:
            return []

        current_set = set(codewords)
        cover_count = np.zeros(total, dtype=np.int32)

        # Initial count
        for c in current_set:
            cover_count[balls[c]] += 1

        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            # Find a codeword c such that all words in its ball have count > 1
            # i.e., no word in ball(c) has count == 1.

            # Optimization: Check if there are any words with count == 1 globally.
            # If not, we can remove any codeword.
            min_count = np.min(cover_count)
            if min_count >= 2:
                c_to_remove = next(iter(current_set))
                current_set.remove(c_to_remove)
                cover_count[balls[c_to_remove]] -= 1
                changed = True
                continue

            # If min_count == 1, we need to find a codeword c that does NOT cover any word with count 1.
            unique_words_mask = (cover_count == 1)

            candidates = list(current_set)
            random.shuffle(candidates)

            for c in candidates:
                num_unique_covered = np.sum(cover_count[balls[c]] == 1)
                if num_unique_covered == 0:
                    current_set.remove(c)
                    cover_count[balls[c]] -= 1
                    changed = True
                    break

        return list(current_set)

    # Helper: Local Search with Swap
    def local_search(codewords):
        current_set = set(remove_redundant(codewords))
        if not current_set:
            return []

        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            cover_count[balls[c]] += 1

        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False

            candidates = list(current_set)
            random.shuffle(candidates)

            for c in candidates:
                if time.time() > deadline - 1.0:
                    break

                c_ball = balls[c]
                U_c = c_ball[cover_count[c_ball] == 1]

                if len(U_c) == 0:
                    current_set.remove(c)
                    cover_count[c_ball] -= 1
                    improved = True
                    break

                found_swap = False
                U_c_list = list(U_c)

                # Generate candidate c' values that could cover the critical words
                # A candidate c' must have all critical_words in balls[c']
                # This is equivalent to c' being in the intersection of balls[cw] for cw in U_c
                # We can use the precomputed ball_membership matrix for this.
                # candidate_mask[c'] is True if for all w in U_c, ball_membership[c', w] is True.

                candidate_mask = np.ones(total, dtype=bool)
                for w in U_c_list:
                    candidate_mask &= ball_membership[:, w]

                valid_candidates = np.where(candidate_mask)[0]
                # Exclude c itself
                valid_candidates = valid_candidates[valid_candidates != c]

                if len(valid_candidates) > 0:
                    # Pick a random candidate from the valid ones
                    c_prime = valid_candidates[np.random.randint(len(valid_candidates))]

                    # Perform the swap
                    current_set.remove(c)
                    cover_count[c_ball] -= 1

                    current_set.add(c_prime)
                    cover_count[balls[c_prime]] += 1

                    improved = True
                    found_swap = True
                    break

                if found_swap:
                    break

        return list(current_set)

    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        # Greedy Initialization
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)

        while np.any(uncovered_mask):
            if time.time() > deadline - 5.0:
                break

            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]

            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)

        if np.any(uncovered_mask):
            continue

        # Local Search
        refined_set = local_search(selected)

        if len(refined_set) < best_size:
            best_size = len(refined_set)
            best_set = refined_set

    # Fallback
    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)

    return result

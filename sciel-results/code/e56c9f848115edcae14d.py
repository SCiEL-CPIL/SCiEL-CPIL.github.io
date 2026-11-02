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
    N = q ** n  # 6561

    # Precompute balls: balls[i] is list of neighbors of i (including i)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = []
    balls_matrix = np.zeros((N, 1 + 2 * n), dtype=np.int32)

    for w in range(N):
        neighbors = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                neighbors.append(new_w)
        balls.append(neighbors)
        balls_matrix[w] = neighbors

    def idx_to_list(idx):
        digits = []
        temp = idx
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        return digits[::-1]

    best_size = N
    best_set = None

    # Main search loop with multiple restarts
    while time.time() < deadline - 10.0:
        # 1. Greedy Construction
        selected = []
        cover_count = np.zeros(N, dtype=np.int32)
        uncovered = np.ones(N, dtype=bool)

        while np.any(uncovered):
            if time.time() > deadline - 10.0:
                break

            gains = uncovered[balls_matrix].sum(axis=1)

            if np.max(gains) == 0:
                break

            max_gain = np.max(gains)
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[np.random.randint(len(candidates))]

            selected.append(best_idx)
            ball = balls[best_idx]
            cover_count[ball] += 1
            uncovered[ball] = False

        if np.any(uncovered):
            continue

        # 2. Initial Pruning
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            i = 0
            while i < len(selected):
                c = selected[i]
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    selected.pop(i)
                    changed = True
                else:
                    i += 1

        # 3. Targeted 1-Swap: for each codeword, find replacement covering all critical words
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Find critical words (covered only by c)
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]

                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Find candidates that cover all critical words
                # Start with neighbors of first critical word, intersect with others
                candidates = set(balls[critical_words[0]].tolist())
                for cw in critical_words[1:]:
                    candidates.intersection_update(balls[cw].tolist())
                    if not candidates:
                        break

                candidates.discard(c)

                if candidates:
                    c_prime = random.choice(list(candidates))
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved = True
                else:
                    new_selected.append(c)

            selected = new_selected

        # 4. 2-Exchange: try to replace two codewords with one
        if len(selected) > 480 and time.time() < deadline - 8.0:
            improved_2ex = True
            while improved_2ex and time.time() < deadline - 5.0:
                improved_2ex = False
                num_trials = min(3000, len(selected) * (len(selected) - 1) // 2)

                for _ in range(num_trials):
                    if time.time() > deadline - 5.0:
                        break

                    idx1 = random.randint(0, len(selected) - 1)
                    idx2 = random.randint(0, len(selected) - 1)
                    if idx1 == idx2:
                        continue

                    c1 = selected[idx1]
                    c2 = selected[idx2]

                    # Temporarily remove c1 and c2
                    cover_count[balls[c1]] -= 1
                    cover_count[balls[c2]] -= 1

                    uncovered_after = (cover_count == 0)

                    if np.any(uncovered_after):
                        uncovered_words = np.where(uncovered_after)[0]

                        # Find c3 that covers all uncovered words
                        candidates = set(balls[uncovered_words[0]].tolist())
                        for uw in uncovered_words[1:]:
                            candidates.intersection_update(balls[uw].tolist())
                            if not candidates:
                                break

                        candidates.discard(c1)
                        candidates.discard(c2)

                        if candidates:
                            c3 = random.choice(list(candidates))
                            cover_count[balls[c3]] += 1
                            # Remove c1, c2 from selected
                            new_selected = []
                            for c in selected:
                                if c != c1 and c != c2:
                                    new_selected.append(c)
                            new_selected.append(c3)
                            selected = new_selected
                            improved_2ex = True
                            break
                        else:
                            # Revert
                            cover_count[balls[c1]] += 1
                            cover_count[balls[c2]] += 1
                    else:
                        # Both can be removed
                        new_selected = []
                        for c in selected:
                            if c != c1 and c != c2:
                                new_selected.append(c)
                        selected = new_selected
                        improved_2ex = True
                        break

        # 5. Final prune after 2-exchanges
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            i = 0
            while i < len(selected):
                c = selected[i]
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    selected.pop(i)
                    changed = True
                else:
                    i += 1

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

    if best_set is None:
        return [idx_to_list(w) for w in range(N)]

    return [idx_to_list(w) for w in best_set]

import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)

    # Fallback for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Precompute balls efficiently
    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Build balls array: balls[w] = [w, neighbors...]
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1

    # Precompute digit representations for Hamming distance
    digit_rep = np.zeros((total, n), dtype=np.int8)
    for w in range(total):
        temp = w
        for i in range(n - 1, -1, -1):
            digit_rep[w, i] = temp % 3
            temp //= 3

    # Precompute frozensets for fast intersection in local search
    ball_sets = [frozenset(balls[w].tolist()) for w in range(total)]

    # Build inverse mapping
    inv_lists = [None] * total
    for c in range(total):
        for w in balls[c]:
            if inv_lists[w] is None:
                inv_lists[w] = []
            inv_lists[w].append(c)

    best_size = total
    best_set = None

    def hamming_dist(a, b):
        return np.sum(digit_rep[a] != digit_rep[b])

    def latin_hypercube_init(num_points):
        """
        Generate a diverse initial set using Latin Hypercube sampling.
        """
        selected = set()
        remaining = num_points
        while remaining > 0:
            word = []
            for pos in range(n):
                word.append(random.randint(0, 2))
            w = 0
            for i, d in enumerate(word):
                w += d * weights[i]
            if w not in selected:
                selected.add(w)
                remaining -= 1
            if time.time() > deadline - 100.0:
                break
        return list(selected)

    def structured_linear_init():
        """
        Generate a structured initial set based on linear code structure.
        """
        codewords = set()
        for trial in range(6):
            P = np.random.randint(0, 3, size=(4, 4))
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        for d in range(3):
                            info = np.array([a, b, c, d], dtype=np.int64)
                            parity = (P @ info) % 3
                            word = np.concatenate([info, parity])
                            w = 0
                            for i, val in enumerate(word):
                                w += val * weights[i]
                            codewords.add(w)
        return list(codewords)

    def greedy_construct(init_set=None):
        """Randomized greedy set cover construction with optional initial set"""
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        uncovered_count = total

        if init_set:
            for c in init_set:
                for w in balls[c]:
                    if uncovered_mask[w]:
                        uncovered_count -= 1
                        uncovered_mask[w] = False
                    cover_count[w] += 1
                selected.append(c)

        cand_count = np.zeros(total, dtype=np.int32)
        for c in range(total):
            if c in selected:
                continue
            cand_count[c] = np.sum(uncovered_mask[balls[c]])

        active = np.ones(total, dtype=bool)
        for c in selected:
            active[c] = False

        while uncovered_count > 0:
            if time.time() > deadline - 10.0:
                break

            active_counts = np.where(active, cand_count, -1)
            max_count = np.max(active_counts)

            if max_count <= 0:
                break

            candidates = np.where((active_counts == max_count) & active)[0]
            if len(candidates) == 0:
                break
            
            # Diversity bias: prefer candidates far from recently selected
            if selected and len(candidates) > 1:
                recent = selected[-min(15, len(selected)):]
                min_dists = np.array([min(hamming_dist(cand, s) for s in recent) for cand in candidates])
                max_d = np.max(min_dists)
                weights_arr = np.exp((min_dists - max_d) / 1.5)
                weights_arr /= weights_arr.sum()
                best_idx = candidates[np.random.choice(len(candidates), p=weights_arr)]
            else:
                best_idx = candidates[np.random.randint(len(candidates))]

            selected.append(best_idx)
            active[best_idx] = False

            for w in balls[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                    uncovered_mask[w] = False
                    for c in inv_lists[w]:
                        if active[c]:
                            cand_count[c] -= 1
                cover_count[w] += 1

        return selected, uncovered_count

    def prune_redundant(selected, cover_count):
        """Remove codewords that are not needed"""
        if not selected:
            return selected, cover_count

        changed = True
        iterations = 0
        while changed and time.time() < deadline - 3.0:
            changed = False
            iterations += 1
            if iterations > 50:
                break
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected

        return selected, cover_count

    def vectorized_local_search(selected, cover_count):
        """
        Vectorized local search using boolean masks.
        Phase 1: Redundancy removal
        Phase 2: 1-Swap (Remove 1, Add 1)
        """
        if not selected:
            return selected, cover_count

        # Phase 1: Remove redundant codewords
        for _pass in range(10):
            if time.time() > deadline - 5.0:
                break
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            if not changed:
                break

        # Phase 2: 1-Swap (Remove 1, Add 1)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]

                if len(critical_words) == 0:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Use frozenset intersection for speed
                candidates = set(ball_sets[critical_words[0]])
                for cw in critical_words[1:]:
                    candidates = candidates & ball_sets[cw]
                    if not candidates:
                        break

                candidates.discard(c)

                if candidates:
                    # Prefer candidates that cover many other words
                    # Simple heuristic: random choice or max coverage
                    c_prime = random.choice(list(candidates))
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved = True
                else:
                    new_selected.append(c)

            selected = new_selected

        return selected, cover_count

    def two_exchange_local_search(selected, cover_count):
        """
        2-exchange local search: try removing 2 codewords and adding 1.
        """
        if not selected or len(selected) < 2:
            return selected, cover_count

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            n_selected = len(selected)
            
            # Try random pairs
            max_pairs = min(n_selected * (n_selected - 1) // 2, 500)
            pairs_tried = 0

            while pairs_tried < max_pairs and time.time() < deadline - 2.0:
                i = random.randint(0, n_selected - 1)
                j = random.randint(0, n_selected - 1)
                if i == j:
                    continue
                pairs_tried += 1

                c1, c2 = selected[i], selected[j]

                # Temporarily remove
                cover_count[balls[c1]] -= 1
                cover_count[balls[c2]] -= 1

                uncovered_after = (cover_count == 0)

                if np.any(uncovered_after):
                    uncovered_words = np.where(uncovered_after)[0]

                    if len(uncovered_words) <= 4:
                        candidates = set(ball_sets[uncovered_words[0]])
                        for uw in uncovered_words[1:]:
                            candidates = candidates & ball_sets[uw]
                            if not candidates:
                                break

                        candidates.discard(c1)
                        candidates.discard(c2)

                        if candidates:
                            c3 = random.choice(list(candidates))
                            cover_count[balls[c3]] += 1
                            # Remove c1, c2 from selected
                            new_selected = [c for c in selected if c != c1 and c != c2]
                            new_selected.append(c3)
                            selected = new_selected
                            improved = True
                            break
                        else:
                            # Restore
                            cover_count[balls[c1]] += 1
                            cover_count[balls[c2]] += 1
                    else:
                        # Restore
                        cover_count[balls[c1]] += 1
                        cover_count[balls[c2]] += 1
                else:
                    # Both can be removed
                    new_selected = [c for c in selected if c != c1 and c != c2]
                    selected = new_selected
                    improved = True
                    break

            # Final pruning
            if time.time() < deadline - 2.0:
                selected, cover_count = prune_redundant(selected, cover_count)

        return selected, cover_count

    # Main loop
    restart_count = 0
    while time.time() < deadline - 5.0:
        restart_count += 1
        seed = restart_count * 12345 + 67890
        np.random.seed(seed)
        random.seed(seed)

        # Choose initialization strategy
        init_strategy = random.random()
        init_set = None

        if init_strategy < 0.4:
            init_size = random.randint(100, 200)
            init_set = latin_hypercube_init(init_size)
        elif init_strategy < 0.6:
            init_set = structured_linear_init()
        elif init_strategy < 0.8:
            init_size = random.randint(50, 100)
            init_set = random.sample(range(total), init_size)

        selected, uncovered_count = greedy_construct(init_set)

        if uncovered_count > 0:
            continue

        # Pruning
        selected, cover_count = prune_redundant(selected, None if cover_count is None else cover_count)
        if cover_count is None:
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[balls[c]] += 1

        # Local Search: Vectorized optimization
        selected, cover_count = vectorized_local_search(selected, cover_count)

        # 2-exchange local search
        selected, cover_count = two_exchange_local_search(selected, cover_count)

        # Final pruning
        selected, cover_count = prune_redundant(selected, cover_count)

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

        if best_size <= 486 and time.time() > deadline - 15.0:
            break
        if time.time() > deadline - 8.0:
            break

    if best_set is None:
        best_set = list(range(total))

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

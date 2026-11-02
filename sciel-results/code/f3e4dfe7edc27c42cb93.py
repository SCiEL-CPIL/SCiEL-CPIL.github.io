import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallbacks for other parameters
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561

    # Precompute balls: balls[w] = [w, w+e1, ..., w+e_{2n}] (17 entries)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
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

    # Precompute frozensets for fast intersection in local search
    ball_sets = [frozenset(balls[w].tolist()) for w in range(total)]

    # Precompute digit representations
    digit_rep = np.zeros((total, n), dtype=np.int8)
    for w in range(total):
        temp = w
        for i in range(n - 1, -1, -1):
            digit_rep[w, i] = temp % 3
            temp //= 3

    best_size = total
    best_set = None

    # --- Partition-based recursive construction ---
    # Decompose 8-dim ternary space into 3^4 = 81 sub-cubes by fixing first 4 coords.
    # For each sub-cube, use the optimal 9-word tetracode (covering radius 1 in 4-dim).
    # Then apply cross-subcube pruning.
    def _partition_construct():
        """Build code by fixing first 4 coords, using tetracode for last 4 coords."""
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # For each of the 81 prefixes (first 4 coords), place 9 tetracode words
        # Total: 81 * 9 = 729 words. Then prune.
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        for prefix in itertools.product(range(3), repeat=4):
            for tc in tetracode:
                word = list(prefix) + list(tc)
                # Convert to integer
                w = 0
                for d in word:
                    w = w * 3 + d
                selected.append(w)
                cover_count[balls[w]] += 1

        # Cross-subcube pruning: remove any codeword whose ball is fully covered
        # by the union of others (i.e., min coverage > 1)
        for _pass in range(10):
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

        return selected, cover_count

    # --- Randomized Greedy Construction ---
    def _greedy_construct(seed):
        np.random.seed(seed)
        random.seed(seed)
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)

        while np.any(uncovered_mask):
            if time.time() > deadline - 15.0:
                break

            counts = uncovered_mask[balls].sum(axis=1)
            if np.max(counts) == 0:
                break

            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]

            # Diversity bias: prefer candidates far from recently selected
            if selected and len(candidates) > 1:
                recent = selected[-min(20, len(selected)):]
                min_dists = np.array([min(np.sum(digit_rep[cand] != digit_rep[s]) for s in recent) for cand in candidates])
                max_d = np.max(min_dists)
                weights_arr = np.exp((min_dists - max_d) / 1.5)
                weights_arr /= weights_arr.sum()
                best_idx = candidates[np.random.choice(len(candidates), p=weights_arr)]
            else:
                best_idx = candidates[np.random.randint(len(candidates))]

            selected.append(int(best_idx))
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)

        return selected, cover_count

    # --- Local Search: 1-Swap with smooth surrogate ---
    def _local_search(selected, cover_count, time_budget):
        """Perform 1-swap local search using a smooth surrogate for redundancy."""
        end_time = time.time() + time_budget
        improved = True
        while improved and time.time() < end_time:
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

                # Find codewords that can replace c (cover all critical words)
                candidates = set(ball_sets[int(critical_words[0])])
                for cw in critical_words[1:]:
                    candidates = candidates & ball_sets[int(cw)]
                    if not candidates:
                        break

                candidates.discard(c)

                if candidates:
                    # Smooth surrogate: use log-sum-exp to softly weight candidates
                    # by how many words they additionally cover (redundancy measure)
                    cand_list = list(candidates)
                    if len(cand_list) > 1:
                        # Measure: for each candidate, how many words in its ball
                        # are already covered by >= 1 other codeword (redundancy)
                        redundancies = np.array([
                            np.sum(cover_count[balls[cp]] >= 1) for cp in cand_list
                        ], dtype=np.float64)
                        # Soft-max: prefer candidates with higher redundancy
                        # (they replace c with less loss of unique coverage)
                        max_r = np.max(redundancies)
                        # Log-sum-exp for numerical stability
                        shifted = redundancies - max_r
                        log_sum_exp = np.log(np.sum(np.exp(shifted)))
                        weights_arr = np.exp(shifted - log_sum_exp)
                        c_prime = int(cand_list[np.random.choice(len(cand_list), p=weights_arr)])
                    else:
                        c_prime = int(cand_list[0])

                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    new_selected.append(c_prime)
                    improved = True
                else:
                    new_selected.append(c)

            selected = new_selected

        return selected, cover_count

    # --- 2-Exchange: Remove 2, Add 1 ---
    def _two_exchange(selected, cover_count, time_budget):
        end_time = time.time() + time_budget
        improved = True
        while improved and time.time() < end_time:
            improved = False
            num_trials = min(5000, len(selected) * (len(selected) - 1) // 2)

            for _ in range(num_trials):
                if time.time() > end_time:
                    break

                idx1 = random.randint(0, len(selected) - 1)
                idx2 = random.randint(0, len(selected) - 1)
                if idx1 == idx2:
                    continue

                c1 = selected[idx1]
                c2 = selected[idx2]

                # Temporarily remove
                cover_count[balls[c1]] -= 1
                cover_count[balls[c2]] -= 1

                uncovered_after = (cover_count == 0)

                if np.any(uncovered_after):
                    uncovered_words = np.where(uncovered_after)[0]

                    if len(uncovered_words) <= 6:
                        candidates = set(ball_sets[int(uncovered_words[0])])
                        for uw in uncovered_words[1:]:
                            candidates = candidates & ball_sets[int(uw)]
                            if not candidates:
                                break

                        candidates.discard(c1)
                        candidates.discard(c2)

                        if candidates:
                            c3 = random.choice(list(candidates))
                            cover_count[balls[c3]] += 1
                            new_selected = [c for c in selected if c != c1 and c != c2]
                            new_selected.append(c3)
                            selected = new_selected
                            improved = True
                            break
                        else:
                            cover_count[balls[c1]] += 1
                            cover_count[balls[c2]] += 1
                    else:
                        cover_count[balls[c1]] += 1
                        cover_count[balls[c2]] += 1
                else:
                    # Both can be removed
                    new_selected = [c for c in selected if c != c1 and c != c2]
                    selected = new_selected
                    improved = True
                    break

                if not improved:
                    cover_count[balls[c1]] += 1
                    cover_count[balls[c2]] += 1

        return selected, cover_count

    # --- Final high-precision polish ---
    def _polish(selected, cover_count, time_budget):
        end_time = time.time() + time_budget
        for _round in range(5):
            if time.time() > end_time:
                break

            # Removal pass
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

            # 1-swap pass
            improved = True
            while improved and time.time() < end_time:
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

                    candidates = set(ball_sets[int(critical_words[0])])
                    for cw in critical_words[1:]:
                        candidates = candidates & ball_sets[int(cw)]
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

        return selected, cover_count

    # --- Main loop: multiple restarts with both construction methods ---
    iteration = 0
    while time.time() < deadline - 10.0:
        iteration += 1
        seed = iteration * 12345 + 67890
        np.random.seed(seed)
        random.seed(seed)

        # Alternate between partition and greedy construction
        if iteration % 3 == 0:
            # Partition-based construction
            selected, cover_count = _partition_construct()
        else:
            # Randomized greedy
            selected, cover_count = _greedy_construct(seed)

        if not selected or np.any(cover_count == 0):
            continue

        # Local search phases
        # Phase 1: Redundancy removal
        for _pass in range(5):
            if time.time() > deadline - 10.0:
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

        # Phase 2: 1-Swap with smooth surrogate
        time_left = deadline - time.time() - 5.0
        if time_left > 2.0:
            selected, cover_count = _local_search(selected, cover_count, min(time_left * 0.4, 30.0))

        # Phase 3: 2-Exchange
        time_left = deadline - time.time() - 5.0
        if time_left > 2.0 and len(selected) > 486:
            selected, cover_count = _two_exchange(selected, cover_count, min(time_left * 0.3, 20.0))

        # Phase 4: Final polish
        time_left = deadline - time.time() - 3.0
        if time_left > 1.0:
            selected, cover_count = _polish(selected, cover_count, min(time_left * 0.5, 15.0))

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()

            if best_size <= 403:
                break

    # Convert best_set to output format
    if best_set is None:
        # Fallback: pure greedy
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = int(candidates[0])
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected

    result = []
    for w in best_set:
        digits = []
        temp = int(w)
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)

    return result

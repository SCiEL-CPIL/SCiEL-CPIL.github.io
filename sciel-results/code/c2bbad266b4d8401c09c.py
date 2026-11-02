import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s wall-clock kill

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

    # Precompute balls: balls[w] = [w, 16 neighbors]
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

    # Frozensets for fast intersection in local search
    ball_sets = [frozenset(balls[w].tolist()) for w in range(total)]

    best_size = total
    best_set = None

    # Digit reps for Hamming distance
    digit_rep = np.zeros((total, n), dtype=np.int8)
    for w in range(total):
        temp = w
        for i in range(n - 1, -1, -1):
            digit_rep[w, i] = temp % 3
            temp //= 3

    def hamming_dist(a, b):
        return int(np.sum(digit_rep[a] != digit_rep[b]))

    # Fast update helpers using numpy advanced indexing (much faster than Python loops)
    def add_cover(cc, c):
        cc[balls[c]] += 1

    def remove_cover(cc, c):
        cc[balls[c]] -= 1

    def add_cover_batch(cc, codes):
        if len(codes) == 0:
            return
        idx = balls[codes]
        np.add.at(cc, idx, 1)

    def remove_cover_batch(cc, codes):
        if len(codes) == 0:
            return
        idx = balls[codes]
        np.subtract.at(cc, idx, 1)

    def remove_redundant(selected, cc):
        """Remove codewords whose entire ball is covered by others. Returns (new_selected, changed)."""
        new_selected = []
        changed = False
        for c in selected:
            if np.min(cc[balls[c]]) > 1:
                cc[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        return new_selected, changed

    def one_swap_pass(selected, cc):
        """Try to replace each non-redundant codeword with another that covers its critical words."""
        improved = False
        new_selected = []
        for c in selected:
            if np.min(cc[balls[c]]) > 1:
                cc[balls[c]] -= 1
                improved = True
                continue
            critical_mask = (cc[balls[c]] == 1)
            critical_words = balls[c][critical_mask]
            if len(critical_words) == 0:
                cc[balls[c]] -= 1
                improved = True
                continue
            cands = ball_sets[critical_words[0]]
            for cw in critical_words[1:]:
                cands = cands & ball_sets[cw]
                if not cands:
                    break
            cands.discard(c)
            if cands:
                cp = random.choice(tuple(cands))
                cc[balls[c]] -= 1
                cc[balls[cp]] += 1
                new_selected.append(cp)
                improved = True
            else:
                new_selected.append(c)
        return new_selected, improved

    def two_ex_one_pass(selected, cc, num_trials):
        """Randomly try to remove 2 codewords and add 1 that covers all newly-uncovered words."""
        for _ in range(num_trials):
            if time.time() > deadline - 4.0:
                return selected, False
            idx1 = random.randint(0, len(selected) - 1)
            idx2 = random.randint(0, len(selected) - 1)
            if idx1 == idx2:
                continue
            c1 = selected[idx1]
            c2 = selected[idx2]
            cc[balls[c1]] -= 1
            cc[balls[c2]] -= 1
            uncovered_after = (cc == 0)
            if np.any(uncovered_after):
                uncovered_words = np.where(uncovered_after)[0]
                if len(uncovered_words) <= 6:
                    cands = ball_sets[uncovered_words[0]]
                    for uw in uncovered_words[1:]:
                        cands = cands & ball_sets[uw]
                        if not cands:
                            break
                    cands.discard(c1)
                    cands.discard(c2)
                    if cands:
                        c3 = random.choice(tuple(cands))
                        cc[balls[c3]] += 1
                        new_selected = [c for c in selected if c != c1 and c != c2]
                        new_selected.append(c3)
                        return new_selected, True
                    else:
                        cc[balls[c1]] += 1
                        cc[balls[c2]] += 1
                else:
                    cc[balls[c1]] += 1
                    cc[balls[c2]] += 1
            else:
                new_selected = [c for c in selected if c != c1 and c != c2]
                return new_selected, True
        return selected, False

    def one_ex_two_pass(selected, cc, num_trials):
        """Randomly try to remove 1 codeword and add 2 that cover its critical words.
        This is the key new exchange: it can reduce size when two new codewords
        together cover more than the one old codeword did."""
        for _ in range(num_trials):
            if time.time() > deadline - 4.0:
                return selected, False
            idx1 = random.randint(0, len(selected) - 1)
            c1 = selected[idx1]
            cc[balls[c1]] -= 1
            uncovered_after = (cc == 0)
            if np.any(uncovered_after):
                uncovered_words = np.where(uncovered_after)[0]
                if len(uncovered_words) <= 12:
                    # Find c2 that covers a subset, then c3 that covers the rest
                    # Strategy: pick c2 from intersection of first few uncovered balls,
                    # then c3 from intersection of remaining
                    uw = list(uncovered_words)
                    random.shuffle(uw)
                    half = len(uw) // 2 + 1
                    first_half = uw[:half]
                    second_half = uw[half:]
                    cands2 = ball_sets[first_half[0]]
                    for f in first_half[1:]:
                        cands2 = cands2 & ball_sets[f]
                        if not cands2:
                            break
                    cands2.discard(c1)
                    if not cands2:
                        cc[balls[c1]] += 1
                        continue
                    # Try a few random c2
                    for _attempt in range(3):
                        c2 = random.choice(tuple(cands2))
                        cc[balls[c2]] += 1
                        # Check what's still uncovered
                        still_uncovered = (cc == 0)
                        if not np.any(still_uncovered):
                            # c2 alone covers everything - just swap
                            new_selected = [c for c in selected if c != c1]
                            new_selected.append(c2)
                            return new_selected, True
                        else:
                            su = np.where(still_uncovered)[0]
                            if len(su) <= 8:
                                cands3 = ball_sets[su[0]]
                                for s in su[1:]:
                                    cands3 = cands3 & ball_sets[s]
                                    if not cands3:
                                        break
                                cands3.discard(c1)
                                cands3.discard(c2)
                                if cands3:
                                    c3 = random.choice(tuple(cands3))
                                    cc[balls[c3]] += 1
                                    new_selected = [c for c in selected if c != c1]
                                    new_selected.append(c2)
                                    new_selected.append(c3)
                                    return new_selected, True
                        # Revert c2
                        cc[balls[c2]] -= 1
                    cc[balls[c1]] += 1
                else:
                    cc[balls[c1]] += 1
            else:
                # c1 was redundant, just remove it
                new_selected = [c for c in selected if c != c1]
                return new_selected, True
        return selected, False

    # Run multiple restarts
    iteration = 0
    while time.time() < deadline - 6.0:
        iteration += 1
        seed = iteration * 12345 + 67890
        np.random.seed(seed)
        random.seed(seed)

        # 1. Randomized Greedy Construction with diversity bias
        selected = []
        cc = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)

        while np.any(uncovered_mask):
            if time.time() > deadline - 12.0:
                break
            counts = uncovered_mask[balls].sum(axis=1)
            if np.max(counts) == 0:
                break
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]

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
            cc[balls[best_idx]] += 1
            uncovered_mask = (cc == 0)

        if np.any(uncovered_mask):
            continue

        # 2. Local Search
        # Phase 1: Remove redundant (multiple passes)
        for _pass in range(5):
            if time.time() > deadline - 5.0:
                break
            selected, changed = remove_redundant(selected, cc)
            if not changed:
                break

        # Phase 2: 1-Swap
        improved = True
        while improved and time.time() < deadline - 5.0:
            selected, improved = one_swap_pass(selected, cc)

        # Phase 3: 2-Exchange (Remove 2, Add 1)
        if len(selected) > 403 and time.time() < deadline - 6.0:
            improved_2ex = True
            while improved_2ex and time.time() < deadline - 5.0:
                num_trials = min(8000, len(selected) * (len(selected) - 1) // 2)
                selected, improved_2ex = two_ex_one_pass(selected, cc, num_trials)

        # Phase 4: 1-Exchange-2 (Remove 1, Add 2) -- NEW
        if len(selected) > 403 and time.time() < deadline - 6.0:
            improved_1x2 = True
            while improved_1x2 and time.time() < deadline - 5.0:
                num_trials = min(6000, len(selected))
                selected, improved_1x2 = one_ex_two_pass(selected, cc, num_trials)

        # Phase 5: Interleave swaps and 2-exchanges for more thorough search
        if time.time() < deadline - 4.0:
            for _round in range(3):
                if time.time() > deadline - 4.0:
                    break
                selected, _ = remove_redundant(selected, cc)
                improved = True
                while improved and time.time() < deadline - 4.0:
                    selected, improved = one_swap_pass(selected, cc)
                if len(selected) > 403:
                    selected, _ = two_ex_one_pass(selected, cc, 4000)
                    selected, _ = one_ex_two_pass(selected, cc, 3000)

        # Phase 6: Final polish - aggressive pass with tighter budget
        if time.time() < deadline - 2.0:
            for _polish in range(5):
                if time.time() > deadline - 2.0:
                    break
                selected, changed = remove_redundant(selected, cc)
                if not changed:
                    break
                selected, _ = one_swap_pass(selected, cc)

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            if best_size <= 403:
                break

    # Convert best_set to output format
    if best_set is None:
        # Fallback: greedy
        selected = []
        cc = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cc[balls[best_idx]] += 1
            uncovered_mask = (cc == 0)
        best_set = selected

    # Deduplicate and validate
    seen = set()
    unique_best = []
    for w in best_set:
        if w not in seen:
            seen.add(w)
            unique_best.append(w)
    best_set = unique_best

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

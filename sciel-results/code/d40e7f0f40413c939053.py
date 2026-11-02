import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback for other cases
    if q ** n > 100000:
        return []

    total = q ** n
    codewords = []
    uncovered = set(range(total))

    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % q
            idx //= q
        return digits

    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * q + d
        return idx

    # Precompute neighbors
    neighbors = {}
    for i in range(total):
        digits = idx_to_digits(i)
        nbs = [i]
        for pos in range(n):
            orig = digits[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                nbs.append(digits_to_idx(new_digits))
        neighbors[i] = nbs

    while uncovered and time.time() < deadline:
        best_cw = -1
        best_count = -1
        for cw in uncovered:
            count = sum(1 for nb in neighbors[cw] if nb in uncovered)
            if count > best_count:
                best_count = count
                best_cw = cw

        if best_cw == -1:
            break

        codewords.append(best_cw)
        for nb in neighbors[best_cw]:
            uncovered.discard(nb)

    result = [idx_to_digits(cw) for cw in codewords]
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    return unique_result


def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Precompute balls using numpy for speed
    # balls[i] = array of 17 word indices covered by word i
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
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
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Precompute neighbor sets for fast membership
    # Using a 2D boolean matrix: ball_membership[i, j] = True if j in balls[i]
    ball_membership = np.zeros((total, total), dtype=bool)
    for w in range(total):
        ball_membership[w, balls[w]] = True

    # Dual-guided greedy with shadow prices
    # Shadow price for word w = 1 / (number of candidates that can cover w)
    # This prioritizes hard-to-cover words

    def dual_guided_greedy():
        uncovered_mask = np.ones(total, dtype=bool)
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)

        # Precompute for each word, how many codewords can cover it (always 17 by symmetry)
        # But dynamically, the number of *uncovered* candidates that cover w matters

        while np.any(uncovered_mask):
            if time.time() > deadline - 5.0:
                break

            # For each candidate c, count how many uncovered words it covers
            counts = np.zeros(total, dtype=np.int32)
            for c in range(total):
                counts[c] = np.sum(uncovered_mask[balls[c]])

            if np.max(counts) == 0:
                break

            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]

            # Tie-breaking: among candidates with same count, prefer those that
            # cover words with higher shadow prices (harder to cover)
            if len(candidates) > 1:
                # Shadow price: words with fewer remaining candidates get higher priority
                # A word w is covered by exactly 17 candidates (by symmetry), but some
                # may already be selected. The "difficulty" is inversely related to
                # how many *unselected* candidates can still cover w.
                # For simplicity, use the count of uncovered words in the candidate's ball
                # weighted by inverse of their current coverage options.
                # This is expensive, so use a simpler heuristic: random selection
                # but with a slight bias toward words that are "central" in the uncovered set.

                # Simple approach: pick randomly among top candidates
                best_idx = candidates[np.random.randint(len(candidates))]
            else:
                best_idx = candidates[0]

            selected.append(int(best_idx))
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)

        return selected

    # Local search: 1-swap (remove c, add d) and 2-remove-1-add
    def compute_cov_count(code_set):
        cov_count = np.zeros(total, dtype=np.int32)
        code_arr = np.array(list(code_set), dtype=np.int32)
        for c in code_arr:
            cov_count[balls[c]] += 1
        return cov_count

    def prune_redundant(code_set):
        """Remove codewords that are not needed for coverage."""
        if not code_set:
            return code_set

        cov_count = compute_cov_count(code_set)
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                # Check if removing c leaves any word uncovered
                removable = True
                for nb in balls[c]:
                    if cov_count[nb] <= 1:
                        removable = False
                        break
                if removable:
                    code_set.remove(c)
                    cov_count[balls[c]] -= 1
                    changed = True
        return code_set

    def local_search(code_set):
        """Perform 1-swap and 2-remove-1-add local search."""
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            cov_count = compute_cov_count(code_set)
            c_list = list(code_set)
            random.shuffle(c_list)

            # Try 1-swaps: remove c, add d
            for c in c_list:
                if time.time() > deadline - 3.0:
                    break
                if c not in code_set:
                    continue

                # Find words that would become uncovered if c is removed
                critical_words = []
                for w in balls[c]:
                    if cov_count[w] == 1:
                        critical_words.append(w)

                if not critical_words:
                    # c is redundant
                    code_set.remove(c)
                    improved = True
                    cov_count = compute_cov_count(code_set)
                    continue

                # Try to find d that covers all critical_words
                # d must be in the intersection of N(w) for all w in critical_words
                if len(critical_words) <= 4:  # Only try if small
                    # Find candidates: words that cover all critical_words
                    candidate_mask = np.ones(total, dtype=bool)
                    for w in critical_words:
                        candidate_mask &= ball_membership[:, w]

                    candidates = np.where(candidate_mask & ~np.isin(np.arange(total), list(code_set)))[0]
                    if len(candidates) > 0:
                        # Pick a random candidate
                        d = int(candidates[np.random.randint(len(candidates))])
                        code_set.remove(c)
                        code_set.add(d)
                        improved = True
                        cov_count = compute_cov_count(code_set)
                        break

            # Try 2-remove-1-add: remove c1, c2, add d
            if not improved and time.time() < deadline - 3.0:
                cov_count = compute_cov_count(code_set)
                c_list = list(code_set)
                random.shuffle(c_list)
                n_cw = len(c_list)

                for i in range(min(n_cw, 50)):  # Limit to first 50 for speed
                    if time.time() > deadline - 3.0:
                        break
                    c1 = c_list[i]
                    if c1 not in code_set:
                        continue

                    for j in range(i + 1, min(i + 10, n_cw)):  # Limit pairs
                        if time.time() > deadline - 3.0:
                            break
                        c2 = c_list[j]
                        if c2 not in code_set:
                            continue

                        # Find words that would become uncovered
                        critical_words = []
                        seen_words = set()

                        for w in balls[c1]:
                            if w in seen_words:
                                continue
                            seen_words.add(w)
                            # Count coverage from c1 and c2
                            cover = cov_count[w]
                            if ball_membership[c2, w]:
                                cover -= 1
                            cover -= 1  # for c1
                            if cover == 0:
                                critical_words.append(w)

                        for w in balls[c2]:
                            if w in seen_words:
                                continue
                            seen_words.add(w)
                            cover = cov_count[w]
                            cover -= 1  # for c2
                            if cover == 0:
                                critical_words.append(w)

                        if not critical_words:
                            # Both are redundant together
                            code_set.remove(c1)
                            code_set.remove(c2)
                            improved = True
                            cov_count = compute_cov_count(code_set)
                            break

                        # Try to find d that covers all critical_words
                        if len(critical_words) <= 3:
                            candidate_mask = np.ones(total, dtype=bool)
                            for w in critical_words:
                                candidate_mask &= ball_membership[:, w]

                            # Exclude existing codewords
                            existing_mask = np.zeros(total, dtype=bool)
                            for c in code_set:
                                existing_mask[c] = True
                            candidate_mask &= ~existing_mask

                            candidates = np.where(candidate_mask)[0]
                            if len(candidates) > 0:
                                d = int(candidates[np.random.randint(len(candidates))])
                                code_set.remove(c1)
                                code_set.remove(c2)
                                code_set.add(d)
                                improved = True
                                cov_count = compute_cov_count(code_set)
                                break

                    if improved:
                        break

            # Prune after each improvement
            if improved:
                code_set = prune_redundant(code_set)

        return code_set

    # Main loop: multiple restarts with different initializations
    best_size = total
    best_set = None

    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1

        # Alternate between different initialization strategies
        if num_restarts % 3 == 0:
            # Max-min distance initialization: start with spread-out points
            selected = []
            uncovered_mask = np.ones(total, dtype=bool)
            cover_count = np.zeros(total, dtype=np.int32)

            # Pick first point randomly
            first = np.random.randint(total)
            selected.append(int(first))
            cover_count[balls[first]] += 1
            uncovered_mask = (cover_count == 0)

            # Then use greedy but with tie-breaking that prefers distant points
            while np.any(uncovered_mask):
                if time.time() > deadline - 15.0:
                    break

                counts = np.zeros(total, dtype=np.int32)
                for c in range(total):
                    counts[c] = np.sum(uncovered_mask[balls[c]])

                if np.max(counts) == 0:
                    break

                max_count = np.max(counts)
                candidates = np.where(counts == max_count)[0]

                if len(candidates) > 1 and len(selected) > 0:
                    # Tie-breaking: prefer candidates far from recent selections
                    # Use Hamming distance approximation via ball overlap
                    # If two points are in each other's ball, distance <= 1
                    # Otherwise, they're farther apart
                    last_sel = selected[-1]
                    # Prefer candidates NOT in the ball of last_sel
                    not_in_ball = ~ball_membership[last_sel, candidates]
                    if np.any(not_in_ball):
                        candidates = candidates[not_in_ball]

                best_idx = candidates[np.random.randint(len(candidates))]
                selected.append(int(best_idx))
                cover_count[balls[best_idx]] += 1
                uncovered_mask = (cover_count == 0)

        else:
            # Standard dual-guided greedy
            selected = dual_guided_greedy()

        if not selected:
            continue

        code_set = set(selected)

        # Prune redundant codewords
        code_set = prune_redundant(code_set)

        # Local search
        code_set = local_search(code_set)

        # Update best
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_set = code_set.copy()
            print(f"Restart {num_restarts}: found solution with {best_size} codewords")

        if best_size <= 486:  # Best known
            break

    if best_set is None:
        # Fallback: return greedy result
        selected = dual_guided_greedy()
        best_set = set(selected)

    # Convert to result format
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % q
            idx //= q
        return digits

    result = [idx_to_digits(c) for c in best_set]
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    return unique_result

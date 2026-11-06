"""Backtracking search with bitmask diff tracking for minimal-diameter Sidon set."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Known valid candidates (verified Sidon sets covering 1..30)
    # We want to minimize the largest element (diameter)
    known_candidates = [
        [0, 1, 3, 7, 12, 21, 25],
        [0, 1, 3, 7, 12, 21, 28],
        [0, 1, 3, 7, 12, 25, 28],
        [0, 2, 5, 10, 18, 23, 28],
        [0, 1, 4, 10, 18, 23, 28],
        [0, 1, 2, 7, 12, 22, 28],
        [0, 1, 3, 8, 12, 22, 26],
        [0, 1, 4, 10, 12, 28, 32, 45, 63],
        [0, 1, 4, 10, 12, 28, 32, 45],
        [0, 1, 4, 10, 12, 28, 32, 45, 47],
        [0, 1, 4, 10, 12, 28, 32, 45, 49],
    ]

    def is_valid(a):
        diffs = set()
        n = len(a)
        for i in range(n):
            for j in range(i + 1, n):
                d = a[j] - a[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return all(d in diffs for d in range(1, 31))

    best_result = None
    for c in known_candidates:
        c = sorted(set(c))
        if 1 <= len(c) <= 32 and is_valid(c):
            if best_result is None or max(c) < max(best_result):
                best_result = c

    # Backtracking: build Sidon set in {0,...,D} containing 0, covering diffs 1..30.
    # Use bitmask for diff tracking (bits 0..29 represent diffs 1..30)
    # Key: efficient pruning and ordering

    def search(D):
        """Find a valid Sidon set in {0,...,D} with 0, covering 1..30. Return None if not found."""
        found = [None]
        FULL_MASK = (1 << 30) - 1  # bits 0..29 set

        def dfs(current, diff_mask, last, depth):
            if time.time() > deadline:
                return True  # timeout
            if found[0] is not None:
                return True

            # Check completion
            if diff_mask == FULL_MASK:
                found[0] = list(current)
                return True

            # Pruning: count remaining uncovered diffs
            remaining = FULL_MASK & ~diff_mask
            remaining_count = bin(remaining).count('1')

            # Max new diffs from adding elements: each new element x adds diffs (x - elem) for all elem in current
            # With at most (32 - len(current)) more elements, and each can add at most len(current) + (remaining elements - 1) new diffs
            # Simpler bound: total possible diffs from k more elements is k*len(current) + k*(k-1)//2
            k_max = 32 - len(current)
            if k_max <= 0:
                return False
            max_new = k_max * len(current) + k_max * (k_max - 1) // 2
            if max_new < remaining_count:
                return False

            # Try each candidate x from last+1 to D
            # Order: prefer x that covers the most uncovered small diffs
            for x in range(last + 1, D + 1):
                if time.time() > deadline:
                    return True
                if found[0] is not None:
                    return True

                # Check Sidon property: all (x - elem) must not be in diff_mask
                ok = True
                new_mask = 0
                for elem in current:
                    d = x - elem
                    if d < 1 or d > 30:
                        # This diff doesn't matter for coverage, but still must be unique
                        # Actually, all positive differences must be distinct, not just 1..30
                        # So we need to track ALL differences, not just 1..30
                        # This is a problem with bitmask approach for large D
                        pass
                    bit = 1 << (d - 1)
                    if diff_mask & bit:
                        ok = False
                        break
                    new_mask |= bit

                # We also need to check that new diffs among themselves don't conflict
                # But x - elem1 and x - elem2 are always distinct for elem1 != elem2
                # However, we need ALL differences to be distinct, not just those in 1..30
                # So we need a full diff set, not just bitmask

                if not ok:
                    continue

                current.append(x)
                if dfs(current, diff_mask | new_mask, x, depth + 1):
                    return True
                current.pop()

            return False

        dfs([0], 0, 0, 0)
        return found[0]

    # The bitmask approach above only tracks diffs 1..30, but we need ALL diffs to be distinct.
    # Let's use a proper set-based approach with better pruning.

    def search_full(D):
        """Find a valid Sidon set in {0,...,D} with 0, covering 1..30. Return None if not found."""
        found = [None]
        FULL_MASK = (1 << 30) - 1

        def dfs(current, diff_set, diff_mask, last):
            if time.time() > deadline:
                return True
            if found[0] is not None:
                return True

            if diff_mask == FULL_MASK:
                found[0] = list(current)
                return True

            remaining_count = 30 - bin(diff_mask).count('1')

            # Pruning: can we cover remaining diffs?
            k_max = 32 - len(current)
            if k_max <= 0:
                return False
            max_new = k_max * len(current) + k_max * (k_max - 1) // 2
            if max_new < remaining_count:
                return False

            # Additional pruning: the largest uncovered diff must be <= D
            for d in range(30, 0, -1):
                if not (diff_mask & (1 << (d - 1))):
                    if d > D:
                        return False
                    break

            # Try candidates, ordered by how many new small diffs they cover
            candidates = []
            for x in range(last + 1, D + 1):
                ok = True
                new_diffs = []
                for elem in current:
                    d = x - elem
                    if d in diff_set:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue
                # Score: number of new diffs in 1..30
                score = sum(1 for d in new_diffs if 1 <= d <= 30 and not (diff_mask & (1 << (d - 1))))
                candidates.append((score, x, new_diffs))

            # Sort by score descending (cover more small diffs first)
            candidates.sort(key=lambda t: -t[0])

            for score, x, new_diffs in candidates:
                if time.time() > deadline:
                    return True
                if found[0] is not None:
                    return True

                current.append(x)
                new_diff_set = diff_set | set(new_diffs)
                new_mask = diff_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                if dfs(current, new_diff_set, new_mask, x):
                    return True
                current.pop()

            return False

        dfs([0], set(), 0, 0)
        return found[0]

    # Try diameters from lower bound upward
    # Lower bound: need at least 30 distinct diffs, k(k-1)/2 >= 30 => k >= 8
    # Also need diff 30, so D >= 30
    # But with Sidon constraint, D is likely larger
    for D in range(28, 100):
        if time.time() > deadline:
            break
        result = search_full(D)
        if result is not None:
            if best_result is None or max(result) < max(best_result):
                best_result = result
            return result

    # Fallback: use best known or greedy
    if best_result is not None:
        return best_result

    # Greedy fallback
    def greedy():
        a = [0]
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            cands = sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a))
            found = False
            for x in cands:
                trial = a + [x]
                diffs = set()
                ok = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        dd = trial[j] - trial[i]
                        if dd in diffs:
                            ok = False
                            break
                        diffs.add(dd)
                    if not ok:
                        break
                if ok and d in diffs:
                    a = sorted(trial)
                    covered = diffs
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = a + [y, y + d]
                    diffs = set()
                    ok = True
                    for i in range(len(trial)):
                        for j in range(i + 1, len(trial)):
                            dd = trial[j] - trial[i]
                            if dd in diffs:
                                ok = False
                                break
                            diffs.add(dd)
                        if not ok:
                            break
                    if ok:
                        a = sorted(trial)
                        covered = diffs
                        break
                    y += 1
        return a

    return greedy()
# EVOLVE-BLOCK-END

"""Backtracking search with bitmask diff tracking for minimal-diameter Sidon set."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Try known candidates first
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

    for c in known_candidates:
        c = sorted(set(c))
        if 1 <= len(c) <= 32 and is_valid(c):
            return c

    # Backtracking: build Sidon set in {0,...,D} containing 0, covering diffs 1..30.
    # Use a set for all differences (correctness first, then optimize).
    # Key insight: for small D (< 100), the search space is manageable with good pruning.

    def search(D):
        """Find a valid Sidon set in {0,...,D} with 0, covering 1..30. Return None if not found."""
        found = [None]

        def dfs(current, diff_set, last):
            if time.time() > deadline:
                return True  # timeout
            if found[0] is not None:
                return True

            # Check completion
            if all(d in diff_set for d in range(1, 31)):
                found[0] = list(current)
                return True

            # Pruning: if we can't cover remaining diffs with remaining elements
            # Max additional diffs from adding k more elements: k * len(current) + k*(k-1)/2
            remaining_diffs_needed = sum(1 for d in range(1, 31) if d not in diff_set)
            max_possible_new = (D - last) * len(current) + (D - last) * (D - last - 1) // 2
            if max_possible_new < remaining_diffs_needed:
                return False

            # Try each candidate
            for x in range(last + 1, D + 1):
                if time.time() > deadline:
                    return True
                # Check Sidon property
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
                # Also check new_diffs among themselves (they're all distinct since 
                # x - elem1 != x - elem2 when elem1 != elem2)
                # So just need to check against existing diff_set
                current.append(x)
                new_diff_set = diff_set | set(new_diffs)
                if dfs(current, new_diff_set, x):
                    return True
                current.pop()

            return False

        dfs([0], set(), 0)
        return found[0]

    # Try diameters from lower bound upward
    for D in range(28, 80):
        if time.time() > deadline:
            break
        result = search(D)
        if result is not None:
            return result

    # Fallback: greedy
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

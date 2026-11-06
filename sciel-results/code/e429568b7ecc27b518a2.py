"""Rewritten program: pick a dense ambient set whose largest Sidon subset is small."""
import time

# EVOLVE-BLOCK-START
import itertools

_DEADLINE = 90.0
_T0 = time.time()


def _time_left():
    return _DEADLINE - 15.0 - (time.time() - _T0)  # 15s safety margin


def _candidate_sets():
    """Yield candidate 12-sets (tuples) to evaluate. Dense sets first."""
    seen = set()

    def emit(s):
        t = tuple(sorted(s))
        if len(t) == 12 and len(set(t)) == 12 and all(0 <= x <= 128 for x in t) and t not in seen:
            seen.add(t)
            yield t

    # 1) consecutive blocks starting at small offsets (densest)
    for start in range(0, 20):
        yield from emit(range(start, start + 12))
    # 2) low-value clusters in [0,25]
    for s in [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14],
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16],
        [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17],
        [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18],
        [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
        [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18],
        [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19],
        [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16],
        [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17],
        [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18],
        [2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 19],
        [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20],
        [2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16],
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 17],
        [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 18],
        [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 19],
        [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 20],
        [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 16],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 17],
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 19],
        [0, 1, 2, 3, 4, 5, 7, 9, 12, 15, 18, 21],
        [0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 20, 23],
        [0, 1, 2, 3, 5, 7, 10, 13, 16, 19, 22, 25],
        [0, 1, 2, 4, 6, 9, 12, 15, 18, 21, 24, 27],
        [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16],
        [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 17],
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 18],
        [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 19],
        [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20],
        [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21],
    ]:
        yield from emit(s)
    # 3) a few structured / spread sets (for safety, in case dense fails)
    for s in [
        [0, 1, 4, 10, 12, 23, 25, 36, 38, 49, 51, 62],
        [0, 2, 3, 7, 12, 18, 25, 34, 44, 55, 67, 80],
        [0, 1, 3, 7, 12, 20, 30, 42, 56, 72, 90, 110],
    ]:
        yield from emit(s)


def _upper_bound(rem, used, limit):
    """Max size of a Sidon subset of `rem` given already-chosen `used`.
    Returns (bound, tight)."""
    if not rem:
        return 0, True
    if len(rem) + len(used) <= limit:
        return len(rem), False
    # count elements of rem that can still be added
    c = 0
    for x in rem:
        if all(x + y != z + w for y in used for z in used for w in used):
            c += 1
    b = c + len(used)
    if b <= limit:
        return b, False
    return len(rem) + len(used), True


def _max_sidon(arr):
    """Exact largest subset with all unordered pair sums (incl. diagonals) unique."""
    n = len(arr)
    best = [0]

    # Precompute sum values
    sums = [arr[i] + arr[j] for i in range(n) for j in range(n)]
    # For each pair (i,j), precompute which other pairs have the same sum
    # sum_conflict[i][j] = set of (k,l) where arr[k]+arr[l] == arr[i]+arr[j] and (k,l) != (i,j)
    # Actually, we need: when we add element x to a used set S, we need all sums x+y for y in S to be unique among themselves AND not collide with any existing sums in S.
    # Better approach: track the set of used sums.

    def rec(rem_mask, used_mask, current_size, used_sums):
        if rem_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        # Upper bound: can we possibly beat best?
        if current_size + bin(rem_mask).count('1') <= best[0]:
            return
        # Try to find a good ordering of remaining elements
        # Pick element with most conflicts first
        best_idx = -1
        best_conflicts = -1
        for i in range(n):
            if rem_mask & (1 << i):
                # Count how many used sums would be created
                conflicts = 0
                x = arr[i]
                for j in range(n):
                    if used_mask & (1 << j):
                        s = x + arr[j]
                        if s in used_sums:
                            conflicts += 1
                if conflicts > best_conflicts:
                    best_conflicts = conflicts
                    best_idx = i
                    if conflicts > 0:
                        break  # Can't add this element, try next
        if best_idx == -1:
            # No elements can be added
            if current_size > best[0]:
                best[0] = current_size
            return

        # Branch: skip best_idx
        rec(rem_mask & ~(1 << best_idx), used_mask, current_size, used_sums)

        # Branch: take best_idx if possible
        x = arr[best_idx]
        new_sums = set()
        can_add = True
        for j in range(n):
            if used_mask & (1 << j):
                s = x + arr[j]
                if s in used_sums or s in new_sums:
                    can_add = False
                    break
                new_sums.add(s)
        if can_add:
            # Also need x+x to be unique
            s_diag = x + x
            if s_diag not in used_sums and s_diag not in new_sums:
                new_sums.add(s_diag)
                rec(rem_mask & ~(1 << best_idx), used_mask | (1 << best_idx), current_size + 1, used_sums | new_sums)

    rec((1 << n) - 1, 0, 0, set())
    return best[0]


def construct():
    best_set = None
    best_size = 10**9
    for cand in _candidate_sets():
        if _time_left() <= 0:
            break
        s = _max_sidon(cand)
        if s < best_size:
            best_size = s
            best_set = cand
        if best_size <= 4:
            break
    if best_set is None:
        return list(range(12))
    return list(best_set)


# EVOLVE-BLOCK-END

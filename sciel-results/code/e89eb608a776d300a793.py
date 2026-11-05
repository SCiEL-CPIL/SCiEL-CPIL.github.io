"""Ambient integer sets with small largest Sidon subset.
We want a 12-element set in [0,128] whose largest 'strong Sidon' subset
(all unordered pair sums including diagonals unique) is as small as possible."""
import time
import random
from itertools import combinations
from collections import Counter


def max_sidon_size(S):
    """Exact largest k such that some k-subset of S has all unordered pair sums
    (including diagonals) unique. Returns that k."""
    S = sorted(S)
    n = len(S)
    if n <= 1:
        return n

    # Precompute pair sums for all indices
    pair_sums = {}
    for i in range(n):
        for j in range(i, n):
            pair_sums[(i, j)] = S[i] + S[j]

    # Check subsets in descending size order for early termination
    for k in range(n, 0, -1):
        for combo in combinations(range(n), k):
            # Check if all pair sums in this subset are unique
            seen_sums = set()
            valid = True
            for idx_a in range(k):
                i = combo[idx_a]
                for idx_b in range(idx_a, k):
                    j = combo[idx_b]
                    s = S[i] + S[j]
                    if s in seen_sums:
                        valid = False
                        break
                    seen_sums.add(s)
                if not valid:
                    break
            if valid:
                return k
    return 0


def _valid(c):
    if len(c) != 12 or len(set(c)) != 12:
        return False
    return all(0 <= x <= 128 for x in c)


def _count_duplicate_sums(S):
    """Count the number of duplicate unordered pair sums (including diagonals)."""
    S = sorted(S)
    n = len(S)
    sum_counts = Counter()
    for i in range(n):
        for j in range(i, n):
            sum_counts[S[i] + S[j]] += 1
    duplicates = sum(v - 1 for v in sum_counts.values() if v > 1)
    return duplicates


def construct():
    start = time.time()
    deadline = start + 85.0

    best_set = None
    best_val = 99
    seen = set()

    def consider(c):
        nonlocal best_set, best_val
        if time.time() > deadline:
            return None
        key = tuple(sorted(c))
        if key in seen:
            return None
        seen.add(key)
        if not _valid(c):
            return None
        v = max_sidon_size(c)
        if v < best_val:
            best_val = v
            best_set = list(c)
        return v

    rng = random.Random(12345)

    # Phase 1: Try consecutive sets (known to have small Sidon subsets)
    for lo in range(0, 117):
        if time.time() > deadline:
            break
        c = list(range(lo, lo + 12))
        consider(c)

    # Phase 2: Structured sets - two interleaved APs with small differences
    for d in range(1, 10):
        if time.time() > deadline:
            break
        for a in range(0, 100):
            for gap in range(1, 12):
                b = a + gap
                if b + 5 * d > 128:
                    continue
                c1 = [a + i * d for i in range(6)]
                c2 = [b + i * d for i in range(6)]
                c = sorted(set(c1 + c2))
                if len(c) == 12 and _valid(c):
                    consider(c)

    # Phase 3: Lattice-like structures with repeated small differences
    for base in range(0, 80):
        if time.time() > deadline:
            break
        for d1 in range(1, 8):
            for d2 in range(1, 8):
                for off in range(0, 15):
                    c1 = [base + i * d1 for i in range(6)]
                    c2 = [base + off + i * d2 for i in range(6)]
                    c = sorted(set(c1 + c2))
                    if len(c) == 12 and _valid(c):
                        consider(c)

    # Phase 4: Random sets in small windows with collision-focused sampling
    for trial in range(3000):
        if time.time() > deadline:
            break
        lo = rng.randint(0, 90)
        width = rng.randint(12, 22)
        hi = min(lo + width, 129)
        if hi - lo < 12:
            continue
        c = sorted(rng.sample(range(lo, hi), 12))
        consider(c)

    # Phase 5: Collision-maximization local search using actual Sidon size
    if best_set is not None:
        cur = list(best_set)
        cur_val = best_val
        cur_dup = _count_duplicate_sums(cur)

        improved = True
        max_iter = 400
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            if not improved:
                break
            improved = False

            best_swap = None
            best_swap_dup = cur_dup
            best_swap_val = cur_val

            for pos in range(12):
                old_val = cur[pos]
                candidates = set()
                for other in cur:
                    for delta in range(-5, 6):
                        nv = other + delta
                        if 0 <= nv <= 128 and nv != old_val:
                            candidates.add(nv)
                for _ in range(10):
                    nv = rng.randint(0, 128)
                    if nv != old_val:
                        candidates.add(nv)

                for nv in candidates:
                    if nv in cur:
                        continue
                    new_cur = cur.copy()
                    new_cur[pos] = nv
                    new_cur.sort()

                    new_dup = _count_duplicate_sums(new_cur)

                    if new_dup > best_swap_dup:
                        best_swap_dup = new_dup
                        best_swap = (pos, nv)
                        new_val = max_sidon_size(new_cur)
                        best_swap_val = new_val
                        if new_val < cur_val:
                            improved = True
                            break
                    elif new_dup == best_swap_dup:
                        new_val = max_sidon_size(new_cur)
                        if new_val < best_swap_val:
                            best_swap = (pos, nv)
                            best_swap_val = new_val
                            if new_val < cur_val:
                                improved = True
                                break
                if improved and best_swap_val < cur_val:
                    break

            if best_swap is not None:
                pos, nv = best_swap
                cur[pos] = nv
                cur.sort()
                cur_dup = best_swap_dup
                if best_swap_val < cur_val:
                    cur_val = best_swap_val
                    improved = True
                    consider(cur)

        if cur_val < best_val:
            consider(cur)

    # Phase 6: Multiple restart local search from diverse starting points
    if time.time() < deadline - 15:
        start_sets = []
        for lo in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            start_sets.append(list(range(lo, lo + 12)))
        for _ in range(20):
            lo = rng.randint(0, 90)
            width = rng.randint(12, 24)
            hi = min(lo + width, 129)
            if hi - lo >= 12:
                start_sets.append(sorted(rng.sample(range(lo, hi), 12)))

        for ss in start_sets:
            if time.time() > deadline - 5:
                break
            if not _valid(ss):
                continue
            cur = list(ss)
            cur_dup = _count_duplicate_sums(cur)
            cur_val = max_sidon_size(cur)

            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                best_swap = None
                best_swap_dup = cur_dup

                for pos in range(12):
                    old_val = cur[pos]
                    candidates = set()
                    for other in cur:
                        for delta in range(-4, 5):
                            nv = other + delta
                            if 0 <= nv <= 128 and nv != old_val:
                                candidates.add(nv)

                    for nv in candidates:
                        if nv in cur:
                            continue
                        new_cur = cur.copy()
                        new_cur[pos] = nv
                        new_cur.sort()
                        new_dup = _count_duplicate_sums(new_cur)
                        if new_dup > best_swap_dup:
                            best_swap_dup = new_dup
                            best_swap = (pos, nv)

                if best_swap is None:
                    break

                pos, nv = best_swap
                cur[pos] = nv
                cur.sort()
                cur_dup = best_swap_dup
                cur_val = max_sidon_size(cur)

            consider(cur)

    if best_set is None:
        best_set = list(range(12))

    return best_set

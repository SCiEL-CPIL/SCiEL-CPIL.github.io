"""Ambient integer sets with small largest strong-Sidon subset.
We want a 12-element set in [0,128] whose largest strong-Sidon subset
(all unordered pair sums including diagonals unique) is as small as possible.

Strategy: maximize the number of duplicate unordered pair sums (including
diagonals) via a greedy collision-maximizing local search, then verify the
exact largest strong-Sidon subset size with a fast DFS.
"""
import random
import time


def max_sidon_size(S):
    """Exact largest k such that some k-subset of S has all unordered pair sums
    (including diagonals) unique. Returns that k."""
    S = list(S)
    n = len(S)
    if n <= 1:
        return n

    # Precompute pair sums for each i: sums of S[i] + S[j] for j >= i
    sums_by_i = [
        [S[i] + S[j] for j in range(i, n)]
        for i in range(n)
    ]

    # Map each distinct sum to an ID
    allsums = set()
    for lst in sums_by_i:
        allsums.update(lst)
    sum_to_id = {s: k for k, s in enumerate(allsums)}
    m = len(allsums)

    best = 1
    cnt = [0] * m

    def dfs(start, size):
        nonlocal best
        if size > best:
            best = size
        if size + (n - start) <= best:
            return
        if best == n:
            return
        for i in range(start, n):
            ok = True
            touched = []
            for s in sums_by_i[i]:
                sid = sum_to_id[s]
                if cnt[sid] >= 1:
                    ok = False
                    break
                cnt[sid] += 1
                touched.append(sid)
            if ok:
                dfs(i + 1, size + 1)
            for sid in touched:
                cnt[sid] -= 1

    dfs(0, 0)
    return best


def _pair_sum_collisions(S):
    """Count the number of duplicate unordered pair sums (including diagonals).
    A larger count means more collisions, which tends to shrink the largest
    strong-Sidon subset."""
    from collections import Counter
    S = list(S)
    n = len(S)
    sums = []
    for i in range(n):
        for j in range(i, n):
            sums.append(S[i] + S[j])
    c = Counter(sums)
    # Number of collisions = total pairs - number of distinct sums
    return len(sums) - len(c)


def _valid(c):
    if len(c) != 12 or len(set(c)) != 12:
        return False
    return all(0 <= x <= 128 for x in c)


def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s

    best_set = None
    best_val = 99
    seen = set()

    def consider(c):
        nonlocal best_set, best_val
        key = tuple(sorted(c))
        if key in seen:
            return
        seen.add(key)
        if not _valid(c):
            return
        v = max_sidon_size(c)
        if v < best_val:
            best_val = v
            best_set = list(c)
        return v

    rng = random.Random(42)

    # Phase 1: Greedy collision-maximizing local search
    # Start from several random seeds and greedily swap to maximize collisions.
    for seed_trial in range(50):
        if time.time() > deadline:
            break
        # Random starting point: dense set in a small window
        lo = rng.randint(0, 110)
        width = rng.randint(14, 25)
        hi = min(lo + width, 129)
        if hi - lo < 12:
            continue
        cur = sorted(rng.sample(range(lo, hi), 12))
        cur_coll = _pair_sum_collisions(cur)

        # Local search: try swaps that increase collisions
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all single-element moves
            for pos in range(12):
                if time.time() > deadline:
                    break
                old_val = cur[pos]
                # Try nearby values
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    nv = old_val + delta
                    if nv < 0 or nv > 128:
                        continue
                    if nv in cur:
                        continue
                    c2 = list(cur)
                    c2[pos] = nv
                    c2.sort()
                    coll = _pair_sum_collisions(c2)
                    if coll > cur_coll:
                        cur = c2
                        cur_coll = coll
                        improved = True
                        break
                if improved:
                    break

        consider(cur)

    # Phase 2: Try structured sets with many collisions
    # Arithmetic progressions with small common difference
    for d in range(1, 6):
        if time.time() > deadline:
            break
        for a in range(0, 129 - d * 11):
            c = [a + i * d for i in range(12)]
            if _valid(c):
                consider(c)

    # Phase 3: Random sampling with collision-based filtering
    for trial in range(3000):
        if time.time() > deadline:
            break
        lo = rng.randint(0, 100)
        width = rng.randint(13, 35)
        hi = min(lo + width, 129)
        if hi - lo < 12:
            continue
        c = sorted(rng.sample(range(lo, hi), 12))
        coll = _pair_sum_collisions(c)
        # Only evaluate if collision count is high (likely small Sidon subset)
        if coll >= 25:  # threshold for promising candidates
            consider(c)

    # Phase 4: Local search from best found, optimizing actual Sidon size
    if best_set is not None:
        cur = list(best_set)
        cur_val = best_val
        for _ in range(300):
            if time.time() > deadline:
                break
            c2 = list(cur)
            pos = rng.randrange(12)
            delta = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            nv = c2[pos] + delta
            if nv < 0 or nv > 128:
                continue
            if nv in c2:
                continue
            c2[pos] = nv
            c2.sort()
            v = max_sidon_size(c2)
            if v < cur_val:
                cur = c2
                cur_val = v
            elif v == cur_val and rng.random() < 0.2:
                cur = c2
        consider(cur)

    # Phase 5: Targeted search for very small windows
    for lo in range(0, 120):
        if time.time() > deadline:
            break
        hi = lo + 14
        if hi > 129:
            break
        # Sample within this small window
        for _ in range(20):
            c = sorted(rng.sample(range(lo, hi), 12))
            coll = _pair_sum_collisions(c)
            if coll >= 20:
                consider(c)

    if best_set is None:
        best_set = list(range(12))

    return best_set

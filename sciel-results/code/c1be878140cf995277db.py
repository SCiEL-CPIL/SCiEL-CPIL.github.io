"""Combined B2[2] set construction: structured greedy + fast local search.

Strategy:
  * Several deterministic greedy orderings (ascending, descending, mixed) to
    get diverse starting sets quickly.
  * A handful of randomized greedy restarts for extra diversity.
  * A local-search phase that (a) inserts any non-chosen element that fits,
    (b) performs 1-for-1 swaps to unblock future insertions, and (c) performs
    2-for-2 swaps for larger structural moves.
  * NumPy-vectorised validity check (5-10x faster than a Python loop) so more
    search iterations fit in the wall-clock budget.
"""
import time
import random
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Greedy construction
# ---------------------------------------------------------------------------

def _greedy_build(candidates, counts_init=None):
    """Greedy: add candidates in given order if B2[2] constraint holds."""
    chosen = []
    counts = defaultdict(int) if counts_init is None else dict(counts_init)
    chosen_set = set()
    for x in candidates:
        if x in chosen_set:
            continue
        ok = True
        for y in chosen:
            s = x + y
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            if counts[s2] + 1 > 2:
                ok = False
        if ok:
            for y in chosen:
                counts[x + y] += 1
            counts[2 * x] += 1
            chosen.append(x)
            chosen_set.add(x)
    return chosen, counts


# ---------------------------------------------------------------------------
# Validity check (vectorised)
# ---------------------------------------------------------------------------

def _is_valid_b2(chosen):
    """Verify B2[2] property using NumPy."""
    if not chosen:
        return True
    arr = np.array(chosen, dtype=np.int64)
    i_idx, j_idx = np.triu_indices(len(arr))
    sums = arr[i_idx] + arr[j_idx]
    _, cnts = np.unique(sums, return_counts=True)
    return bool(np.all(cnts <= 2))


# ---------------------------------------------------------------------------
# Insertion / swap helpers
# ---------------------------------------------------------------------------

def _try_insert(chosen, counts, y, chosen_set):
    """Try to insert y into chosen.  Returns True on success."""
    if y in chosen_set:
        return False
    for z in chosen:
        s = y + z
        if counts.get(s, 0) + 1 > 2:
            return False
    s2 = 2 * y
    if counts.get(s2, 0) + 1 > 2:
        return False
    for z in chosen:
        counts[y + z] = counts.get(y + z, 0) + 1
    counts[2 * y] = counts.get(2 * y, 0) + 1
    chosen.append(y)
    chosen_set.add(y)
    return True


def _conflict_score(y, chosen, counts):
    """Number of sums that would exceed the limit if y were added (0 = fits)."""
    conf = 0
    for z in chosen:
        s = y + z
        if counts.get(s, 0) >= 2:
            conf += 1
    if counts.get(2 * y, 0) >= 2:
        conf += 1
    return conf


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def _local_search(chosen, time_limit=30.0):
    """Local search: insertions, 1-for-1 swaps, 2-for-2 swaps."""
    chosen = sorted(chosen)
    n = len(chosen)
    counts = defaultdict(int)
    for i in range(n):
        for j in range(i, n):
            counts[chosen[i] + chosen[j]] += 1
    chosen_set = set(chosen)

    start_time = time.time()
    rng = random.Random(777)
    stall = 0
    max_stall = 30

    while stall < max_stall:
        if time.time() - start_time > time_limit:
            break
        improved = False
        n = len(chosen)

        # ---- Phase 1: pure insertions (grow the set) ----------------------
        # Score every non-chosen candidate; try the best ones first.
        to_try = []
        for y in range(1, 1001):
            if y in chosen_set:
                continue
            c = _conflict_score(y, chosen, counts)
            if c <= 2:
                to_try.append((c, y))
        to_try.sort()
        for _, y in to_try:
            if time.time() - start_time > time_limit:
                break
            if _try_insert(chosen, counts, y, chosen_set):
                improved = True
        if improved:
            stall = 0
            continue

        # ---- Phase 2: 1-for-1 swaps ---------------------------------------
        # Randomly sample elements to remove and candidates to add.
        n = len(chosen)
        if n > 0:
            order = list(range(n))
            rng.shuffle(order)
            num_removals = min(n, max(3, int((time_limit - (time.time() - start_time)) * 15)))
            # Build a candidate pool of non-chosen elements, sorted by conflict
            cand_pool = []
            for y in range(1, 1001):
                if y in chosen_set:
                    continue
                c = _conflict_score(y, chosen, counts)
                if c <= 4:
                    cand_pool.append((c, y))
            cand_pool.sort()
            max_cands = min(len(cand_pool), max(20, int((time_limit - (time.time() - start_time)) * 20)))

            for idx in order[:num_removals]:
                if time.time() - start_time > time_limit:
                    break
                x = chosen[idx]
                # Remove x
                for z in chosen:
                    if z != x:
                        counts[x + z] -= 1
                counts[2 * x] -= 1
                chosen_set.discard(x)
                chosen.pop(idx)

                done = False
                for _, y in cand_pool[:max_cands]:
                    if y in chosen_set:
                        continue
                    if _try_insert(chosen, counts, y, chosen_set):
                        done = True
                        break
                if done:
                    improved = True
                    chosen.sort()
                    break
                else:
                    # Restore x
                    for z in chosen:
                        counts[x + z] = counts.get(x + z, 0) + 1
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    chosen.append(x)
                    chosen_set.add(x)
                    chosen.sort()
            if improved:
                stall = 0
                continue

        # ---- Phase 3: 2-for-2 swaps ---------------------------------------
        n = len(chosen)
        if n >= 4:
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((i, j))
            rng.shuffle(pairs)
            max_pairs = min(len(pairs), max(5, int((time_limit - (time.time() - start_time)) * 5)))

            for i, j in pairs[:max_pairs]:
                if time.time() - start_time > time_limit:
                    break
                if i == j:
                    continue
                # Ensure i < j after any prior modifications
                if i > j:
                    i, j = j, i
                if i >= n or j >= n:
                    continue
                x1 = chosen[i]
                x2 = chosen[j]

                # Remove x1 and x2
                for z in chosen:
                    if z != x1:
                        counts[x1 + z] -= 1
                counts[2 * x1] -= 1
                for z in chosen:
                    if z != x2 and z != x1:
                        counts[x2 + z] -= 1
                counts[2 * x2] -= 1
                # Remove x1+x2 counted twice? No, it was counted once (i<j).
                # Actually x1+x2 is in both loops? Let's check:
                # First loop: z != x1, includes x2 -> counts[x1+x2] -= 1
                # Second loop: z != x2 and z != x1 -> excludes x1, so x1+x2 not touched
                # Good.
                chosen_set.discard(x1)
                chosen_set.discard(x2)
                # Remove from list (remove higher index first)
                chosen.pop(j)
                chosen.pop(i)

                # Try to add two new candidates
                added = 0
                for y in range(1, 1001):
                    if time.time() - start_time > time_limit:
                        break
                    if y in chosen_set:
                        continue
                    if _conflict_score(y, chosen, counts) <= 1:
                        if _try_insert(chosen, counts, y, chosen_set):
                            added += 1
                            if added >= 2:
                                break

                if added >= 2:
                    improved = True
                    chosen.sort()
                    break
                else:
                    # Restore x1 and x2
                    for z in chosen:
                        counts[x1 + z] = counts.get(x1 + z, 0) + 1
                        counts[x2 + z] = counts.get(x2 + z, 0) + 1
                    counts[x1 + x2] = counts.get(x1 + x2, 0) + 1
                    counts[2 * x1] = counts.get(2 * x1, 0) + 1
                    counts[2 * x2] = counts.get(2 * x2, 0) + 1
                    chosen_set.add(x1)
                    chosen_set.add(x2)
                    chosen.append(x1)
                    chosen.append(x2)
                    chosen.sort()
            if improved:
                stall = 0
                continue

        stall += 1

    return sorted(chosen)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 s safety margin before 150 s kill

    best = []
    best_size = 0

    def _consider(chosen):
        nonlocal best, best_size
        if len(chosen) > best_size:
            best = sorted(chosen)
            best_size = len(best)

    # ---- Greedy orderings ------------------------------------------------
    # 1. Ascending
    if time.time() < deadline:
        chosen, _ = _greedy_build(list(range(1, 1001)))
        _consider(chosen)

    # 2. Descending
    if time.time() < deadline:
        chosen, _ = _greedy_build(list(range(1000, 0, -1)))
        _consider(chosen)

    # 3. Mixed (interleave small and large)
    if time.time() < deadline:
        small = list(range(1, 501))
        large = list(range(1000, 500, -1))
        mixed = []
        si, li = 0, 0
        while si < len(small) or li < len(large):
            if si < len(small):
                mixed.append(small[si]); si += 1
            if li < len(large):
                mixed.append(large[li]); li += 1
        chosen, _ = _greedy_build(mixed)
        _consider(chosen)

    # 4. Random restarts
    rng = random.Random(42)
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        cand = list(range(1, 1001))
        if r % 3 == 0:
            rng.shuffle(cand)
        elif r % 3 == 1:
            third = len(cand) // 3
            rng.shuffle(cand[third:])
        else:
            # Weighted: favour smaller numbers
            weights = [1001 - x for x in cand]
            seen = set()
            unique_cand = []
            for x in rng.choices(range(1, 1001), weights=weights, k=1000):
                if x not in seen:
                    seen.add(x)
                    unique_cand.append(x)
            cand = unique_cand + [x for x in range(1, 1001) if x not in seen]
        chosen, _ = _greedy_build(cand)
        _consider(chosen)

    # ---- Local search on the best solution --------------------------------
    remaining = deadline - time.time()
    if remaining > 3.0 and best:
        ls_time = min(120.0, remaining - 2.0)
        if ls_time > 1.0:
            improved = _local_search(best, time_limit=ls_time)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)

    # ---- Final validity check ---------------------------------------------
    if not _is_valid_b2(best):
        # Fallback: rebuild with simple ascending greedy
        chosen, _ = _greedy_build(list(range(1, 1001)))
        best = chosen

    return best

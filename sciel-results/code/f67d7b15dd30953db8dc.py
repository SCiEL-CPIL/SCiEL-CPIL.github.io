import time
import random

_DEADLINE = 80.0  # safety margin before 90s limit

def _is_mean_of_subset(C, i):
    """
    Check if C[i] is the mean of any subset of at least 2 OTHER elements in C.
    Uses centered bitset DP for speed.
    Returns True if violation found.
    """
    a = C[i]
    others = C[:i] + C[i+1:]
    n = len(others)
    if n < 2:
        return False

    zs = [y - a for y in others]

    max_abs = sum(abs(z) for z in zs)
    if max_abs == 0:
        return False

    offset = max_abs
    mask = (1 << (2 * max_abs + 1)) - 1
    zero_bit = 1 << offset

    # dp[k]: bitset, bit s set if sum s achievable with exactly k elements
    dp = [0] * (n + 1)
    dp[0] = zero_bit

    for z in zs:
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if z >= 0:
                shifted = dp[k-1] << z
            else:
                shifted = dp[k-1] >> (-z)
            dp[k] |= (shifted & mask)
            if k >= 2 and (dp[k] & zero_bit):
                return True

    for k in range(2, n + 1):
        if dp[k] & zero_bit:
            return True

    return False

def _full_check(C):
    """Check if any element in C is the mean of a subset of >= 2 others."""
    for i in range(len(C)):
        if _is_mean_of_subset(C, i):
            return True
    return False

def _fast_pair_check(C, x):
    """
    Quick check for 2-element subset violations involving x and set C.
    Returns True if a pair violation exists.
    """
    C_set = set(C)

    # 1. x is mean of a pair in C: 2x = a + b
    for a in C:
        b = 2 * x - a
        if b in C_set and b != a:
            return True

    # 2. Some a in C is mean of pair (x, b): 2a = x + b => b = 2a - x
    for a in C:
        b = 2 * a - x
        if b in C_set and b != a and b != x:
            return True

    return False

def _is_mean_of_subset_others(a, others):
    """Check if a is the mean of any subset of >= 2 elements in others (list)."""
    n = len(others)
    if n < 2:
        return False
    zs = [y - a for y in others]
    max_abs = sum(abs(z) for z in zs)
    if max_abs == 0:
        return False
    offset = max_abs
    mask = (1 << (2 * max_abs + 1)) - 1
    zero_bit = 1 << offset
    dp = [0] * (n + 1)
    dp[0] = zero_bit
    for z in zs:
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if z >= 0:
                shifted = dp[k-1] << z
            else:
                shifted = dp[k-1] >> (-z)
            dp[k] |= (shifted & mask)
            if k >= 2 and (dp[k] & zero_bit):
                return True
    for k in range(2, n + 1):
        if dp[k] & zero_bit:
            return True
    return False

def _incremental_add_check(C, x):
    """
    Given a valid set C, check whether C + [x] is still valid.
    Only x's own validity and pair-level changes of existing elements can
    introduce a violation. Returns True if C + [x] is valid.
    """
    # 1. Is x the mean of any subset of >= 2 of C?
    if _is_mean_of_subset_others(x, C):
        return False

    # 2. Any existing element newly a mean of a PAIR involving x?
    C_set = set(C)
    for a in C:
        b = 2 * a - x
        if b in C_set and b != a and b != x:
            return False

    return True

def _try_local_improvement(S, candidates, time_budget):
    """Improve S via insertions and swaps. Returns improved set."""
    start = time.time()
    S = list(S)
    S_set = set(S)
    improved = True
    max_size = 48

    while improved and time.time() - start < time_budget:
        improved = False

        # --- Insertions: add any candidate that keeps the set valid ---
        if len(S) < max_size:
            for cand in candidates:
                if time.time() - start > time_budget:
                    break
                if cand in S_set:
                    continue
                if _incremental_add_check(S, cand):
                    S.append(cand)
                    S_set.add(cand)
                    improved = True
                    if len(S) >= max_size:
                        break

        # --- Swaps: replace an element with a candidate not in S ---
        for i in range(len(S)):
            if time.time() - start > time_budget:
                break
            old_val = S[i]
            for cand in candidates:
                if time.time() - start > time_budget:
                    break
                if cand in S_set:
                    continue
                if cand == old_val:
                    continue
                temp_set = S[:i] + S[i+1:]
                # New element must be valid against the rest
                if _is_mean_of_subset_others(cand, temp_set):
                    continue
                # Any element in temp_set newly a pair-mean with cand?
                tset = set(temp_set)
                ok = True
                for a in temp_set:
                    b = 2 * a - cand
                    if b in tset and b != a and b != cand:
                        ok = False
                        break
                if not ok:
                    continue
                # Full check of the swapped set
                new_S = temp_set + [cand]
                if not _full_check(new_S):
                    S[i] = cand
                    S_set.remove(old_val)
                    S_set.add(cand)
                    improved = True
                    break

    return S

def _pair_conflict_count(x, universe_set):
    """Number of distinct y in universe (y != x) such that 2x - y is in universe.
    Fewer = x is less likely to be a pair-mean. Used to order candidates."""
    cnt = 0
    for y in universe_set:
        if y == x:
            continue
        if (2 * x - y) in universe_set:
            cnt += 1
    return cnt

def construct():
    start_time = time.time()
    best_set = []

    candidates = list(range(1, 513))
    cand_set = set(candidates)

    rng = random.Random(42)

    # Precompute pair-conflict counts to bias ordering toward "safe" extremes
    # (extremes have fewer in-range 3-term-AP partners).
    conflict = {x: _pair_conflict_count(x, cand_set) for x in candidates}

    def _elapsed():
        return time.time() - start_time

    orders = []

    # 1. Descending (large numbers first)
    orders.append(candidates[::-1])
    # 2. Ascending
    orders.append(candidates[:])
    # 3. Interleaved high-low
    inter = []
    lo, hi = 1, 512
    while lo <= hi:
        inter.append(hi)
        if lo != hi:
            inter.append(lo)
        lo += 1
        hi -= 1
    orders.append(inter)
    # 4. Sort by ascending pair-conflict count (fewest AP partners first)
    orders.append(sorted(candidates, key=lambda x: (conflict[x], -x)))
    # 5. Sort by ascending conflict, ties by closeness to extremes
    orders.append(sorted(candidates, key=lambda x: (conflict[x], -abs(x - 256))))

    # 6. Many random orders
    for _ in range(250):
        if _elapsed() > _DEADLINE - 8:
            break
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)

    # 7. Weighted random biased toward extremes (extremes are safer)
    for _ in range(60):
        if _elapsed() > _DEADLINE - 8:
            break
        weighted = []
        for x in candidates:
            w = abs(x - 256) / 256.0 + 0.1
            weighted.extend([x] * max(1, int(w * 3)))
        rng.shuffle(weighted)
        seen = set()
        unique_weighted = []
        for x in weighted:
            if x not in seen:
                seen.add(x)
                unique_weighted.append(x)
        orders.append(unique_weighted)

    for order in orders:
        if _elapsed() > _DEADLINE - 5:
            break

        selected = []
        sel_set = set()

        for x in order:
            if _elapsed() > _DEADLINE - 3:
                break
            if len(selected) >= 48:
                break

            # Fast pair check prunes most candidates
            if _fast_pair_check(selected, x):
                continue

            # Incremental full validity check (fast)
            if _incremental_add_check(selected, x):
                selected.append(x)
                sel_set.add(x)

        if len(selected) > len(best_set):
            best_set = selected[:]

        if len(best_set) >= 48:
            break

    # Local improvement / fill pass
    if len(best_set) < 48 and _elapsed() < _DEADLINE - 6:
        remaining = _DEADLINE - 6 - _elapsed()
        if remaining > 1:
            best_set = _try_local_improvement(best_set, candidates, remaining)

    # Final verification (defensive)
    if best_set and _full_check(best_set):
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    elif best_set:
        return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

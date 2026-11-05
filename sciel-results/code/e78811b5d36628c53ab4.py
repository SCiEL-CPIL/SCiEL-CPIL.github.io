"""
Mean-free subset construction: maximize cardinality of distinct integers in [1,512]
such that no element is the arithmetic mean of any subset of >=2 other elements.

Strategy:
  - Use a fast incremental validity check with precomputed DP bitsets.
  - Greedy construction with multiple orderings (ascending, descending, random, structured).
  - Local search: swap-out / swap-in to improve.
  - Structured candidates (2^i * 3^j) plus random fill.
"""
import time
import random

def _valid_mean_free(selected):
    """Full verification: no element is mean of any subset of >=2 others."""
    n = len(selected)
    if n < 3:
        return True
    selected = list(set(selected))
    n = len(selected)
    if n < 3:
        return True
    for i, a in enumerate(selected):
        others = [x for j, x in enumerate(selected) if j != i]
        m = len(others)
        if m < 2:
            continue
        max_sum = sum(others)
        dp = [0] * (m + 1)
        dp[0] = 1
        for x in others:
            for k in range(m, 0, -1):
                if dp[k - 1]:
                    dp[k] |= (dp[k - 1] << x)
        for k in range(2, m + 1):
            target = k * a
            if target > max_sum:
                break
            if dp[k] & (1 << target):
                return False
    return True


def _build_dp(lst):
    """Build subset-sum DP bitsets for list lst. dp[k] is a bitset of achievable sums by subsets of size k."""
    n = len(lst)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in lst:
        for k in range(n, 0, -1):
            if dp[k - 1]:
                dp[k] |= (dp[k - 1] << x)
    return dp


def _fast_validity_check_incr(existing, new_elem):
    """
    Check if adding new_elem to existing (valid) set preserves validity.
    Two checks:
      1. new_elem is not mean of any subset (size>=2) of existing.
      2. No existing element a becomes mean of a subset of (existing\\{a}) ∪ {new_elem}.
    """
    n = len(existing)
    if n < 2:
        # Only need to check if new_elem is mean of >=2 others, but n<2 means no such subset
        return True

    existing_list = list(existing)

    # Check 1: new_elem not mean of subset of existing
    dp = _build_dp(existing_list)
    max_sum = sum(existing_list)
    for k in range(2, n + 1):
        target = k * new_elem
        if target > max_sum:
            break
        if dp[k] & (1 << target):
            return False

    # Check 2: for each a in existing, subsets of (existing\{a}) ∪ {new_elem}
    # Case B (includes new_elem): T ⊆ existing\{a}, |T|>=1, sum(T) = (|T|+1)*a - new_elem
    for i, a in enumerate(existing_list):
        others = [x for j, x in enumerate(existing_list) if j != i]
        m = len(others)
        if m < 1:
            continue
        dp_others = _build_dp(others)
        max_sum_others = sum(others)
        for t in range(1, m + 1):
            k = t + 1
            target = k * a - new_elem
            if target < 0:
                continue
            if target > max_sum_others:
                break
            if dp_others[t] & (1 << target):
                return False
    return True


def _generate_23_candidates(max_val=512):
    """All numbers of form 2^i * 3^j <= max_val."""
    candidates = set()
    i = 0
    while (1 << i) <= max_val:
        val_i = 1 << i
        j = 0
        while val_i * (3 ** j) <= max_val:
            candidates.add(val_i * (3 ** j))
            j += 1
        i += 1
    return sorted(candidates)


def _generate_powers(max_val=512, primes=(2, 3, 5, 7)):
    """Numbers of form product of prime powers <= max_val."""
    candidates = set([1])
    for p in primes:
        new = set()
        for c in candidates:
            v = c
            while v * p <= max_val:
                v *= p
                new.add(v)
        candidates |= new
    return sorted([x for x in candidates if x <= max_val])


def _greedy_construct(order, deadline, best_len):
    """Greedy: add elements in given order if valid."""
    current = []
    current_set = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in current_set:
            continue
        if _fast_validity_check_incr(current, x):
            current.append(x)
            current_set.add(x)
            if len(current) > best_len:
                best_len = len(current)
    return current, best_len


def _expand_set(s, deadline, best_len):
    """Try to add more elements to set s."""
    s = list(s)
    s_set = set(s)
    best_len = len(s)
    remaining = [x for x in range(1, 513) if x not in s_set]
    # Prioritize structured numbers
    c23_set = set(_generate_23_candidates(512))
    remaining.sort(key=lambda x: (0 if x in c23_set else 1, x))
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in remaining[:]:
            if time.time() > deadline:
                break
            if x in s_set:
                continue
            if _fast_validity_check_incr(s, x):
                s.append(x)
                s_set.add(x)
                remaining.remove(x)
                best_len = len(s)
                improved = True
    return s, best_len


def _local_search(s, deadline, best_len):
    """Try removing 1-2 elements and re-adding to improve."""
    s = list(s)
    best_len = len(s)
    s_set = set(s)
    missing = [x for x in range(1, 513) if x not in s_set]
    c23_set = set(_generate_23_candidates(512))
    missing.sort(key=lambda x: (0 if x in c23_set else 1, x))

    # Remove 1, add up to 2
    for i in range(len(s)):
        if time.time() > deadline:
            break
        temp = s[:i] + s[i + 1:]
        temp_set = set(temp)
        added = 0
        for x in missing:
            if time.time() > deadline:
                break
            if x in temp_set:
                continue
            if _fast_validity_check_incr(temp, x):
                temp.append(x)
                temp_set.add(x)
                added += 1
                if added >= 2:
                    break
        if len(temp) > best_len:
            s = temp
            best_len = len(s)
            s_set = set(s)
            missing = [x for x in range(1, 513) if x not in s_set]
            missing.sort(key=lambda x: (0 if x in c23_set else 1, x))

    # Remove 2, add up to 3
    if time.time() < deadline:
        for i in range(len(s)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(s)):
                if time.time() > deadline:
                    break
                temp = s[:i] + s[i + 1:j] + s[j + 1:]
                temp_set = set(temp)
                added = 0
                for x in missing:
                    if time.time() > deadline:
                        break
                    if x in temp_set:
                        continue
                    if _fast_validity_check_incr(temp, x):
                        temp.append(x)
                        temp_set.add(x)
                        added += 1
                        if added >= 3:
                            break
                if len(temp) > best_len:
                    s = temp
                    best_len = len(s)
                    s_set = set(s)
                    missing = [x for x in range(1, 513) if x not in s_set]
                    missing.sort(key=lambda x: (0 if x in c23_set else 1, x))
                    break  # restart inner loop with new missing
    return s, best_len


def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s limit

    c23 = _generate_23_candidates(512)
    c23_set = set(c23)
    all_nums = list(range(1, 513))
    non_c23 = [x for x in all_nums if x not in c23_set]

    best_set = []
    best_len = 0

    # --- Strategy 1: Greedy with multiple orderings ---
    orders = [
        c23,
        sorted(c23, reverse=True),
        all_nums,
        sorted(all_nums, reverse=True),
    ]
    for _ in range(6):
        lst = all_nums[:]
        random.shuffle(lst)
        orders.append(lst)
    for _ in range(4):
        lst = c23 + random.sample(non_c23, min(40, len(non_c23)))
        random.shuffle(lst)
        orders.append(lst)

    for order in orders:
        if time.time() > deadline:
            break
        current, best_len = _greedy_construct(order, deadline, best_len)
        if len(current) > len(best_set):
            best_set = current[:]
            best_len = len(current)

    # --- Strategy 2: Expand best set ---
    if best_set:
        best_set, best_len = _expand_set(best_set, deadline, best_len)

    # --- Strategy 3: Random restarts with mixed pools ---
    for iteration in range(60):
        if time.time() > deadline:
            break
        pool = c23 + random.sample(non_c23, min(80, len(non_c23)))
        random.shuffle(pool)
        current, best_len = _greedy_construct(pool, deadline, best_len)
        if len(current) > len(best_set):
            best_set = current[:]
            best_len = len(current)

    # --- Strategy 4: Local search ---
    if best_set and time.time() < deadline:
        best_set, best_len = _local_search(best_set, deadline, best_len)

    # --- Strategy 5: Final expansion ---
    if best_set and time.time() < deadline:
        best_set, best_len = _expand_set(best_set, deadline, best_len)

    # Final local search if time permits
    if best_set and time.time() < deadline:
        best_set, best_len = _local_search(best_set, deadline, best_len)

    return best_set


if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    assert _valid_mean_free(s), "Invalid set found!"
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"

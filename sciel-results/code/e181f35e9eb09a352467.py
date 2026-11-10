# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# ---------- Fast validity check using numpy int64 bitsets ----------

def _dp_subset_sums(others, max_sum):
    """
    Return a list dp where dp[k] is an int64 numpy bitset: bit s is set iff
    some k-element subset of `others` sums to s (for s in [0, max_sum]).
    """
    m = len(others)
    if m == 0:
        return [np.zeros(max_sum + 1, dtype=np.int64)]
    dp = [np.zeros(max_sum + 1, dtype=np.int64)] * (m + 1)
    dp[0][0] = 1
    for x in others:
        for k in range(m, 0, -1):
            # shift dp[k-1] left by x, but only the part that fits in max_sum
            src = dp[k - 1]
            if x <= max_sum:
                shifted = np.zeros(max_sum + 1, dtype=np.int64)
                shifted[x:max_sum + 1] = src[0:max_sum + 1 - x]
                dp[k] = np.bitwise_or(dp[k], shifted)
    return dp


def _is_valid_fast(set_vals):
    """No element is the mean of any subset of >=2 OTHER elements."""
    n = len(set_vals)
    if n < 3:
        return True
    for a in set_vals:
        others = [x for x in set_vals if x != a]
        m = len(others)
        if m < 2:
            continue
        max_target = m * a
        if max_target > 100000:
            # Too big for numpy bitset; fall back to a Python-int check
            mask = (1 << (max_target + 1)) - 1
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k - 1] << x)
                    dp[k] &= mask
            for k in range(2, m + 1):
                target = k * a
                if (dp[k] >> target) & 1:
                    return False
            continue
        dp = _dp_subset_sums(others, max_target)
        for k in range(2, m + 1):
            target = k * a
            if target <= max_target:
                if dp[k][target]:
                    return False
    return True


def _check_addition_fast(current_set, x):
    """
    Check whether adding x to current_set creates any violation.
    Two cases:
      (A) x is the mean of some subset of >=2 elements from current_set.
      (B) For some a in current_set, a is the mean of a subset of
          (current_set + [x]) that includes x (and >=1 other).
    """
    n = len(current_set)
    if n < 2:
        return True

    # --- Case A ---
    m = n
    max_target_A = m * x
    if max_target_A <= 100000:
        dp = _dp_subset_sums(current_set, max_target_A)
        for k in range(2, m + 1):
            target = k * x
            if target <= max_target_A and dp[k][target]:
                return False

    # --- Case B ---
    for a in current_set:
        others_no_a = [y for y in current_set if y != a]
        m2 = len(others_no_a)
        if m2 < 1:
            continue
        # We need: exists t in [1, m2] such that a*(t+1) - x is achievable
        # as a t-element subset sum of others_no_a.
        # Max target for this a: a*(m2+1) - x.
        max_target_B = a * (m2 + 1) - x
        if max_target_B < 0:
            continue
        if max_target_B > 100000:
            # Fallback: check small t with Python ints
            mask = (1 << (max_target_B + 1)) - 1
            dp = [0] * (m2 + 1)
            dp[0] = 1
            for elem in others_no_a:
                for k in range(m2, 0, -1):
                    dp[k] |= (dp[k - 1] << elem)
                    dp[k] &= mask
            for t in range(1, m2 + 1):
                target = a * (t + 1) - x
                if 0 <= target <= max_target_B and (dp[t] >> target) & 1:
                    return False
            continue
        dp = _dp_subset_sums(others_no_a, max_target_B)
        for t in range(1, m2 + 1):
            target = a * (t + 1) - x
            if 0 <= target <= max_target_B and dp[t][target]:
                return False

    return True


# ---------- Primes helper ----------

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]


# ---------- Construction ----------

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill

    best_set = []

    primes = _primes_upto(512)
    composites = [x for x in range(1, 513) if x not in primes]

    def try_greedy(order):
        selected = []
        for x in order:
            if time.time() > deadline:
                return selected
            if len(selected) >= 48:
                break
            if _check_addition_fast(selected, x):
                selected.append(x)
        return selected

    # Strategy 1: primes first, then ascending composites
    order1 = primes + sorted(composites)
    s = try_greedy(order1)
    if len(s) > len(best_set):
        best_set = s[:]

    # Strategy 2: ascending 1..512
    s = try_greedy(list(range(1, 513)))
    if len(s) > len(best_set):
        best_set = s[:]

    # Strategy 3: descending 512..1
    s = try_greedy(list(range(512, 0, -1)))
    if len(s) > len(best_set):
        best_set = s[:]

    # Strategy 4: primes descending, then composites descending
    order4 = primes[::-1] + sorted(composites, reverse=True)
    s = try_greedy(order4)
    if len(s) > len(best_set):
        best_set = s[:]

    # Strategy 5: randomized greedy with multiple seeds
    for seed in range(40):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        s = try_greedy(order)
        if len(s) > len(best_set):
            best_set = s[:]

    # Strategy 6: top up best_set by scanning all candidates
    if time.time() < deadline - 5 and len(best_set) < 48:
        current = best_set[:]
        for x in range(1, 513):
            if time.time() > deadline:
                break
            if len(current) >= 48:
                break
            if x in current:
                continue
            if _check_addition_fast(current, x):
                current.append(x)
        if len(current) > len(best_set):
            best_set = current[:]

    # Strategy 7: local search – swap one element out, try to add one in
    if time.time() < deadline - 10 and len(best_set) < 48:
        current = best_set[:]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for x in range(1, 513):
                    if x in current:
                        continue
                    candidate = current[:i] + [x] + current[i + 1:]
                    if _is_valid_fast(candidate):
                        # try to add one more
                        for y in range(1, 513):
                            if y in candidate:
                                continue
                            if len(candidate) >= 48:
                                break
                            if _check_addition_fast(candidate, y):
                                candidate.append(y)
                                improved = True
                                break
                        current = candidate
                        if len(current) > len(best_set):
                            best_set = current[:]
                        break
                if improved:
                    break

    # Final top-up
    if len(best_set) < 48 and time.time() < deadline - 2:
        current = best_set[:]
        for x in range(1, 513):
            if time.time() > deadline:
                break
            if len(current) >= 48:
                break
            if x in current:
                continue
            if _check_addition_fast(current, x):
                current.append(x)
        if len(current) > len(best_set):
            best_set = current[:]

    return best_set[:48]
# EVOLVE-BLOCK-END

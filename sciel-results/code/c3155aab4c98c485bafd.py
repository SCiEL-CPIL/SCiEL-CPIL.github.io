import time
import random
from math import isqrt

MAX_N = 12
MAX_VAL = 10000
DEADLINE = 70.0  # safety margin under 90s wall-clock

def _sieve(limit):
    """Return smallest prime factor array up to limit."""
    spf = list(range(limit + 1))
    spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, isqrt(limit) + 1):
        if spf[i] == i:
            step = i
            start = i * i
            for j in range(start, limit + 1, step):
                if spf[j] == j:
                    spf[j] = i
    return spf

_SPF_LIMIT = 2 * MAX_VAL + 100
_SPF = _sieve(_SPF_LIMIT)

def _prime_factors(num, spf=None):
    """Return set of distinct prime factors of num."""
    if spf is None:
        spf = _SPF
    if num <= 1:
        return set()
    factors = set()
    while num > 1:
        p = spf[num]
        if p <= 1:
            p = num
        factors.add(p)
        while num % p == 0:
            num //= p
    return factors

def _smoothness(num, prime_set, spf=None):
    """Return True if all prime factors of num are in prime_set."""
    if spf is None:
        spf = _SPF
    while num > 1:
        p = spf[num]
        if p <= 1:
            p = num
        if p not in prime_set:
            return False
        while num % p == 0:
            num //= p
    return True

def _candidate_numbers_for_primes(primes, limit=MAX_VAL):
    """Return sorted list of numbers <= limit that are S-smooth (all prime factors in primes)."""
    prime_set = set(primes)
    smooth = []
    for n in range(2, limit + 1):
        if _smoothness(n, prime_set):
            smooth.append(n)
    return smooth

def _pairwise_sums_ok(nums, prime_set, spf=None):
    """Check all pairwise sums of distinct elements are S-smooth."""
    n = len(nums)
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            if not _smoothness(ai + nums[j], prime_set, spf):
                return False
    return True

def _count_distinct_primes(nums, spf=None):
    """Count distinct primes dividing any pairwise sum."""
    if spf is None:
        spf = _SPF
    all_primes = set()
    n = len(nums)
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            all_primes |= _prime_factors(ai + nums[j], spf)
    return len(all_primes), all_primes

def _greedy_build(primes, k=MAX_N, limit=MAX_VAL, rng=None):
    """Greedy: pick k distinct S-smooth numbers such that all pairwise sums are S-smooth."""
    if rng is None:
        rng = random.Random(0)
    prime_set = set(primes)
    smooth = _candidate_numbers_for_primes(primes, limit)
    if len(smooth) < k:
        return None
    chosen = []
    for cand in smooth:
        if len(chosen) >= k:
            break
        ok = True
        for c in chosen:
            if not _smoothness(cand + c, prime_set):
                ok = False
                break
        if ok:
            chosen.append(cand)
    if len(chosen) == k:
        return chosen
    return None

def _local_search(nums, primes, rng, max_iters=200000, deadline=None):
    """Local search: swap/replace elements to keep all pairwise sums S-smooth, trying to reduce count or find valid set."""
    prime_set = set(primes)
    smooth = _candidate_numbers_for_primes(primes, MAX_VAL)
    if len(smooth) < MAX_N:
        return nums
    smooth_set = set(smooth)
    current = list(nums)
    current_set = set(current)

    def valid(arr):
        return _pairwise_sums_ok(arr, prime_set)

    best = current[:]
    best_score = len(current)
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if len(current) == MAX_N and valid(current):
            best = current[:]
            break
        # pick a random position
        pos = rng.randrange(len(current))
        # try replacing with a random smooth number
        new_val = rng.choice(smooth)
        if new_val in current_set:
            continue
        trial = current[:]
        trial[pos] = new_val
        if valid(trial):
            current = trial
            current_set = set(current)
            if len(current) == MAX_N:
                best = current[:]
                break
    return best

def _try_prime_set(primes, rng, deadline):
    """Try to build a valid set for the given prime set."""
    # greedy first
    res = _greedy_build(primes, MAX_N, MAX_VAL, rng)
    if res is not None:
        return res
    # local search starting from a partial greedy
    partial = []
    prime_set = set(primes)
    smooth = _candidate_numbers_for_primes(primes, MAX_VAL)
    for cand in smooth:
        if len(partial) >= MAX_N:
            break
        ok = True
        for c in partial:
            if not _smoothness(cand + c, prime_set):
                ok = False
                break
        if ok:
            partial.append(cand)
    if len(partial) < MAX_N:
        # extend via local search
        res = _local_search(partial, primes, rng, max_iters=50000, deadline=deadline)
        if len(res) == MAX_N and _pairwise_sums_ok(res, prime_set):
            return res
    return None

def _all_prime_sets(max_prime, size):
    """Generate prime sets of given size from primes up to max_prime."""
    primes = [p for p in range(2, max_prime + 1) if _SPF[p] == p]
    from itertools import combinations
    return list(combinations(primes, size))

def construct():
    start = time.time()
    deadline = start + DEADLINE
    best = list(range(1, MAX_N + 1))
    best_count, _ = _count_distinct_primes(best)

    # Try increasing prime set sizes; smaller is better
    # Start with small prime sets and grow
    rng = random.Random(12345)

    # Candidate prime pools (include small primes that often appear in sums)
    # Sums of two numbers up to 10000 go up to 20000, so primes up to 20000 matter
    # But we want few distinct primes, so focus on small prime sets
    prime_pool = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    # Try prime set sizes from 2 upward; the first size that yields a valid set is optimal-ish
    # But we want to minimize distinct primes, so try small sizes first
    from itertools import combinations
    for size in range(2, len(prime_pool) + 1):
        if time.time() > deadline:
            break
        # For each combination of `size` primes, try to build a valid set
        # Limit combinations to keep runtime bounded
        combos = list(combinations(prime_pool, size))
        rng.shuffle(combos)
        # Cap number of combos tried per size to respect deadline
        max_combos = 200
        for combo in combos[:max_combos]:
            if time.time() > deadline:
                break
            res = _try_prime_set(combo, rng, deadline)
            if res is not None:
                # Found a valid set with `size` primes
                cnt, _ = _count_distinct_primes(res)
                if cnt < best_count:
                    best = res
                    best_count = cnt
                # Try to improve with local search
                improved = _local_search(res, combo, rng, max_iters=20000, deadline=deadline)
                if len(improved) == MAX_N and _pairwise_sums_ok(improved, set(combo)):
                    cnt2, _ = _count_distinct_primes(improved)
                    if cnt2 < best_count:
                        best = improved
                        best_count = cnt2
        # If we found any valid set at this size, we can stop (smaller size is better)
        # But only if we actually found one
        # (We break out of the size loop if we found a valid set at this size)
        # Actually, we should continue to try to find even smaller? No, size is increasing.
        # So if we found a valid set at size `size`, that's the minimum for this pool.
        # But maybe a different combination at same size has fewer actual distinct primes?
        # The count is exactly `size` if all primes are used, but could be less if some aren't needed.
        # For safety, continue trying other combos at same size to see if we can get fewer.
        # But to save time, stop after first success at this size.
        if best_count <= size:
            break

    # Final polish: local search on best to try to reduce distinct primes further
    if time.time() < deadline - 5:
        # Get current prime set
        _, cur_primes = _count_distinct_primes(best)
        improved = _local_search(best, list(cur_primes), rng, max_iters=50000, deadline=deadline)
        if len(improved) == MAX_N:
            cnt, _ = _count_distinct_primes(improved)
            if cnt < best_count:
                best = improved
                best_count = cnt

    return best

# EVOLVE-BLOCK-END

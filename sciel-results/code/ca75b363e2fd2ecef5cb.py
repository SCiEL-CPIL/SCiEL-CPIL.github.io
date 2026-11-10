"""Optimized program: precomputed SPF table + incremental local search + multiple restarts."""
import time
import random
from collections import Counter

_START_TIME = time.time()
_DEADLINE_SECONDS = 85.0

def _elapsed():
    return time.time() - _START_TIME

def _deadline_hit():
    return _elapsed() > _DEADLINE_SECONDS

# Precompute smallest prime factor table up to 20000 (max sum = 10000+10000)
_MAX_SUM = 20000
_SPF = list(range(_MAX_SUM + 1))

def _build_spf():
    for i in range(2, int(_MAX_SUM**0.5) + 1):
        if _SPF[i] == i:
            for j in range(i*i, _MAX_SUM + 1, i):
                if _SPF[j] == j:
                    _SPF[j] = i

_build_spf()

def _prime_factors_fast(n):
    """Return set of prime factors using precomputed SPF."""
    factors = set()
    while n > 1:
        p = _SPF[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _score_full(nums):
    """Compute the number of distinct prime factors across all pairwise sums."""
    factor_count = Counter()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            for p in _prime_factors_fast(s):
                factor_count[p] += 1
    return len(factor_count)

def _score_incremental(old_factor_count, old_val, new_val, nums, idx):
    """
    Compute new factor count after replacing nums[idx] with new_val.
    old_factor_count: Counter of prime factors before the change.
    Returns (new_factor_count, new_score) or None if invalid.
    """
    n = len(nums)
    # Remove contributions of old_val with all other elements
    new_fc = Counter(old_factor_count)
    for j in range(n):
        if j == idx:
            continue
        s = old_val + nums[j]
        for p in _prime_factors_fast(s):
            new_fc[p] -= 1
            if new_fc[p] == 0:
                del new_fc[p]
    # Add contributions of new_val with all other elements
    for j in range(n):
        if j == idx:
            continue
        s = new_val + nums[j]
        for p in _prime_factors_fast(s):
            new_fc[p] += 1
    return new_fc, len(new_fc)

def _local_search(nums, factor_count, score, max_time=None):
    """
    Local search: try single-element replacements to reduce score.
    Returns (best_nums, best_fc, best_score).
    """
    n = len(nums)
    best = nums[:]
    best_fc = Counter(factor_count)
    best_score = score
    improved = True
    while improved and not _deadline_hit():
        if max_time is not None and _elapsed() > max_time:
            break
        improved = False
        for idx in range(n):
            if _deadline_hit():
                return best, best_fc, best_score
            old_val = best[idx]
            # Try a range of candidate values
            # First try nearby values, then broader
            candidates = []
            for d in [-3, -2, -1, 1, 2, 3]:
                v = old_val + d
                if 1 <= v <= 10000 and v != old_val:
                    if v not in best:
                        candidates.append(v)
            # Also try some random candidates
            for _ in range(5):
                v = random.randint(1, 10000)
                if v != old_val and v not in best:
                    candidates.append(v)
            
            for new_val in candidates:
                if _deadline_hit():
                    break
                new_fc, new_score = _score_incremental(best_fc, old_val, new_val, best, idx)
                if new_score < best_score:
                    best[idx] = new_val
                    best_fc = new_fc
                    best_score = new_score
                    improved = True
                    break
            if improved:
                break
    return best, best_fc, best_score

def _random_initial(multiplier=1):
    """Generate a random initial set of 12 distinct positive integers <= 10000."""
    nums = set()
    while len(nums) < 12:
        nums.add(random.randint(1, 10000) * multiplier)
    return sorted(nums)

def _structured_initial(multiplier, base=1):
    """Generate structured initial set: multiples of multiplier starting from base."""
    nums = []
    val = base
    while len(nums) < 12:
        if val <= 10000:
            nums.append(val)
        val += multiplier
    if len(nums) < 12:
        # Fill with random multiples
        while len(nums) < 12:
            v = random.randint(1, 10000 // multiplier) * multiplier
            if v not in nums:
                nums.append(v)
    return sorted(nums)

def construct():
    random.seed(42)
    
    best_nums = None
    best_score = float('inf')
    best_fc = None
    
    # Multiple restarts with different initializations
    strategies = []
    
    # Strategy 1: multiples of 6 (guarantees 2 and 3 as factors)
    strategies.append(lambda: _structured_initial(6, 6))
    # Strategy 2: multiples of 4 (guarantees 2)
    strategies.append(lambda: _structured_initial(4, 4))
    # Strategy 3: multiples of 2 (guarantees 2)
    strategies.append(lambda: _structured_initial(2, 2))
    # Strategy 4: random
    strategies.append(lambda: _random_initial(1))
    # Strategy 5: small numbers
    strategies.append(lambda: list(range(1, 13)))
    # Strategy 6: multiples of 3
    strategies.append(lambda: _structured_initial(3, 3))
    # Strategy 7: multiples of 12
    strategies.append(lambda: _structured_initial(12, 12))
    # Strategy 8: random even
    strategies.append(lambda: _random_initial(2))
    # Strategy 9: small multiples of 6
    strategies.append(lambda: list(range(6, 79, 6)))
    # Strategy 10: arithmetic progression with diff 6
    strategies.append(lambda: list(range(1, 73, 6)))
    
    # Evaluate each strategy
    for i, strat in enumerate(strategies):
        if _deadline_hit():
            break
        nums = strat()
        if len(nums) != 12 or len(set(nums)) != 12:
            continue
        if not all(1 <= x <= 10000 for x in nums):
            continue
        
        fc = Counter()
        n = len(nums)
        for a in range(n):
            for b in range(a + 1, n):
                s = nums[a] + nums[b]
                for p in _prime_factors_fast(s):
                    fc[p] += 1
        score = len(fc)
        
        if score < best_score:
            best_score = score
            best_nums = nums[:]
            best_fc = Counter(fc)
        
        # Local search from this starting point
        remaining_time = _DEADLINE_SECONDS - _elapsed()
        if remaining_time > 2.0:
            nums_ls, fc_ls, score_ls = _local_search(nums, fc, score)
            if score_ls < best_score:
                best_score = score_ls
                best_nums = nums_ls[:]
                best_fc = Counter(fc_ls)
    
    # Additional random restarts if time permits
    attempts = 0
    while not _deadline_hit() and attempts < 50:
        attempts += 1
        nums = _random_initial(random.choice([1, 2, 3, 4, 6, 12]))
        if len(nums) != 12 or len(set(nums)) != 12:
            continue
        if not all(1 <= x <= 10000 for x in nums):
            continue
        
        fc = Counter()
        n = len(nums)
        for a in range(n):
            for b in range(a + 1, n):
                s = nums[a] + nums[b]
                for p in _prime_factors_fast(s):
                    fc[p] += 1
        score = len(fc)
        
        if score < best_score:
            best_score = score
            best_nums = nums[:]
            best_fc = Counter(fc)
        
        # Quick local search
        remaining_time = _DEADLINE_SECONDS - _elapsed()
        if remaining_time > 1.0:
            nums_ls, fc_ls, score_ls = _local_search(nums, fc, score)
            if score_ls < best_score:
                best_score = score_ls
                best_nums = nums_ls[:]
                best_fc = Counter(fc_ls)
    
    # Final polish: more aggressive local search on best solution
    if best_nums is not None and not _deadline_hit():
        remaining = _DEADLINE_SECONDS - _elapsed()
        if remaining > 3.0:
            # Try multiple local search passes
            for _ in range(3):
                if _deadline_hit():
                    break
                nums_p, fc_p, score_p = _local_search(best_nums, best_fc, best_score)
                if score_p < best_score:
                    best_score = score_p
                    best_nums = nums_p[:]
                    best_fc = Counter(fc_p)
    
    if best_nums is None:
        best_nums = list(range(1, 13))
    
    # Validate
    assert len(set(best_nums)) == 12
    assert all(1 <= x <= 10000 for x in best_nums)
    
    return best_nums

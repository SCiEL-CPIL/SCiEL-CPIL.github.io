"""Rewritten program: find 12 distinct positive integers <=10000 minimizing distinct primes in pairwise sums."""
import time
import random
from itertools import combinations

# Global deadline guard (wall-clock 90s, leave safety margin)
_START_TIME = time.time()
DEADLINE = 85.0  # seconds

def _time_ok():
    return (time.time() - _START_TIME) < DEADLINE

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors

def _distinct_prime_count(nums):
    """Count distinct primes dividing all pairwise sums of distinct-index elements."""
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            all_primes |= _prime_factors(s)
    return len(all_primes)

def _smooth_numbers(p_set, max_val, count):
    """Generate numbers <= max_val whose prime factors are all in p_set.
    Returns sorted list of distinct such numbers up to count."""
    result = set()
    # Simple generation via BFS
    result.add(1)
    queue = [1]
    i = 0
    while i < len(queue) and len(result) < count:
        v = queue[i]
        i += 1
        for p in p_set:
            nv = v * p
            if nv <= max_val and nv not in result:
                result.add(nv)
                queue.append(nv)
    return sorted(result)

def _try_p_set(p_set, max_val=10000, n_needed=12):
    """Try to find n_needed distinct numbers from p_set-smooth numbers
    such that all pairwise sums have prime factors only in p_set."""
    smooth = _smooth_numbers(p_set, max_val, 200)
    if len(smooth) < n_needed:
        return None
    
    # Try all subsets of size n_needed? Too many. Instead, try greedy/random.
    # First, check if we can find a valid subset.
    # Use random sampling first for speed.
    best = None
    for _ in range(200):
        if not _time_ok():
            break
        if len(smooth) <= n_needed:
            candidate = smooth
        else:
            candidate = random.sample(smooth, n_needed)
        # Check if all pairwise sums have only primes in p_set
        valid = True
        for i in range(n_needed):
            for j in range(i + 1, n_needed):
                s = candidate[i] + candidate[j]
                for pf in _prime_factors(s):
                    if pf not in p_set:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return candidate
    return None

def construct():
    """Return 12 distinct positive integers <=10000 minimizing distinct primes in pairwise sums."""
    best_result = list(range(1, 13))
    best_count = _distinct_prime_count(best_result)
    
    # Strategy 1: Try 2-prime sets {2,3}, {2,5}, {3,5}, etc.
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    for i in range(len(small_primes)):
        for j in range(i + 1, len(ssmall_primes)):
            p_set = frozenset({small_primes[i], small_primes[j]})
            result = _try_p_set(p_set)
            if result is not None:
                # Verify
                cnt = _distinct_prime_count(result)
                if cnt < best_count:
                    best_count = cnt
                    best_result = result
                    if best_count <= 2:
                        return best_result
            if not _time_ok():
                return best_result
    
    # Strategy 2: Try 3-prime sets
    for i in range(len(small_primes)):
        for j in range(i + 1, len(small_primes)):
            for k in range(j + 1, len(small_primes)):
                p_set = frozenset({small_primes[i], small_primes[j], small_primes[k]})
                result = _try_p_set(p_set)
                if result is not None:
                    cnt = _distinct_prime_count(result)
                    if cnt < best_count:
                        best_count = cnt
                        best_result = result
                        if best_count <= 3:
                            return best_result
                if not _time_ok():
                    return best_result
    
    # Strategy 3: Random search with local improvement
    random.seed(42)
    for iteration in range(500):
        if not _time_ok():
            break
        # Random candidate
        candidate = random.sample(range(1, 10001), 12)
        cnt = _distinct_prime_count(candidate)
        if cnt < best_count:
            best_count = cnt
            best_result = candidate
            # Local improvement
            for _ in range(100):
                if not _time_ok():
                    break
                idx = random.randrange(12)
                old_val = best_result[idx]
                new_val = random.randrange(1, 10001)
                if new_val in best_result:
                    continue
                best_result[idx] = new_val
                new_cnt = _distinct_prime_count(best_result)
                if new_cnt > best_count:
                    best_result[idx] = old_val
                elif new_cnt < best_count:
                    best_count = new_cnt
                if best_count <= 2:
                    return best_result
    
    return best_result

# Fix the typo in the code above
def construct():
    """Return 12 distinct positive integers <=10000 minimizing distinct primes in pairwise sums."""
    best_result = list(range(1, 13))
    best_count = _distinct_prime_count(best_result)
    
    # Strategy 1: Try 2-prime sets {2,3}, {2,5}, {3,5}, etc.
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    for i in range(len(small_primes)):
        for j in range(i + 1, len(small_primes)):
            p_set = frozenset({small_primes[i], small_primes[j]})
            result = _try_p_set(p_set)
            if result is not None:
                # Verify
                cnt = _distinct_prime_count(result)
                if cnt < best_count:
                    best_count = cnt
                    best_result = result
                    if best_count <= 2:
                        return best_result
            if not _time_ok():
                return best_result
    
    # Strategy 2: Try 3-prime sets
    for i in range(len(small_primes)):
        for j in range(i + 1, len(small_primes)):
            for k in range(j + 1, len(small_primes)):
                p_set = frozenset({small_primes[i], small_primes[j], small_primes[k]})
                result = _try_p_set(p_set)
                if result is not None:
                    cnt = _distinct_prime_count(result)
                    if cnt < best_count:
                        best_count = cnt
                        best_result = result
                        if best_count <= 3:
                            return best_result
                if not _time_ok():
                    return best_result
    
    # Strategy 3: Random search with local improvement
    random.seed(42)
    for iteration in range(500):
        if not _time_ok():
            break
        # Random candidate
        candidate = random.sample(range(1, 10001), 12)
        cnt = _distinct_prime_count(candidate)
        if cnt < best_count:
            best_count = cnt
            best_result = candidate
            # Local improvement
            for _ in range(100):
                if not _time_ok():
                    break
                idx = random.randrange(12)
                old_val = best_result[idx]
                new_val = random.randrange(1, 10001)
                if new_val in best_result:
                    continue
                best_result[idx] = new_val
                new_cnt = _distinct_prime_count(best_result)
                if new_cnt > best_count:
                    best_result[idx] = old_val
                elif new_cnt < best_count:
                    best_count = new_cnt
                if best_count <= 2:
                    return best_result
    
    return best_result

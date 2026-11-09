import time
import random
from collections import defaultdict

_deadline = None

def _set_deadline(t):
    global _deadline
    _deadline = t

def _time_ok():
    return _deadline is None or time.time() < _deadline

# Precompute smallest prime factor up to 20000 (max sum is 10000+9999=19999)
_MAX_SUM = 20000
_spf = list(range(_MAX_SUM + 1))
for i in range(2, int(_MAX_SUM**0.5) + 1):
    if _spf[i] == i:
        for j in range(i*i, _MAX_SUM + 1, i):
            if _spf[j] == j:
                _spf[j] = i

def _prime_factors(n):
    """Return frozenset of distinct prime factors of n."""
    if n <= 1:
        return frozenset()
    factors = set()
    while n > 1:
        p = _spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return frozenset(factors)

# Precompute prime factors for all possible sums 2..20000
_pf_cache = [None] * (_MAX_SUM + 1)
for s in range(2, _MAX_SUM + 1):
    _pf_cache[s] = _prime_factors(s)

def _count_distinct_primes(arr):
    """Count distinct primes dividing all pairwise sums of distinct elements."""
    primes = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            s = ai + arr[j]
            primes.update(_pf_cache[s])
    return len(primes)

def _is_smooth_sum(a, b, allowed):
    """Check if a+b is composed only of primes in allowed set."""
    s = a + b
    if s > _MAX_SUM:
        return False
    for p in _pf_cache[s]:
        if p not in allowed:
            return False
    return True

def _try_smooth_search(allowed_primes, max_candidates=2000):
    """Find 12 distinct integers whose pairwise sums use only allowed primes."""
    allowed = set(allowed_primes)
    # Find candidates x such that 2x is smooth
    candidates = []
    for x in range(1, 10001):
        s = 2 * x
        if s > _MAX_SUM:
            break
        smooth = True
        for p in _pf_cache[s]:
            if p not in allowed:
                smooth = False
                break
        if smooth:
            candidates.append(x)
        if len(candidates) >= max_candidates:
            break
    
    if len(candidates) < 12:
        return None
    
    # Greedy: try multiple starting points
    best_clique = []
    # Try starting from various positions
    start_points = list(range(0, min(len(candidates), 200), 3))
    random.shuffle(start_points)
    
    for si in start_points:
        if not _time_ok():
            break
        clique = [candidates[si]]
        for x in candidates[si+1:]:
            if len(clique) == 12:
                break
            ok = True
            for y in clique:
                if not _is_smooth_sum(x, y, allowed):
                    ok = False
                    break
            if ok:
                clique.append(x)
        if len(clique) > len(best_clique):
            best_clique = clique
        if len(best_clique) == 12:
            break
    
    if len(best_clique) == 12:
        return best_clique
    return None

def _local_search(arr, max_iters=5000):
    """Improve arr by swapping elements to reduce prime count."""
    current = list(arr)
    current_count = _count_distinct_primes(current)
    rng = random.Random(42)
    
    for _ in range(max_iters):
        if not _time_ok():
            break
        # Pick a random index to replace
        i = rng.randint(0, 11)
        old_val = current[i]
        # Try a few random replacements
        for _ in range(10):
            new_val = rng.randint(1, 10000)
            if new_val == old_val:
                continue
            if new_val in current:
                continue
            new_arr = current.copy()
            new_arr[i] = new_val
            new_count = _count_distinct_primes(new_arr)
            if new_count < current_count:
                current = new_arr
                current_count = new_count
                break
    
    return current, current_count

def _residue_class_search():
    """Search using arithmetic progressions with smooth moduli."""
    best_arr = None
    best_count = 99
    
    # Try moduli that are products of small primes
    small_primes = [2, 3, 5, 7, 11, 13]
    
    # Generate smooth moduli up to 500
    smooth_mods = set()
    def gen_smooth(val, idx):
        if val > 500:
            return
        if val > 1:
            smooth_mods.add(val)
        for i in range(idx, len(small_primes)):
            gen_smooth(val * small_primes[i], i)
    gen_smooth(1, 0)
    
    # Also add small moduli
    for m in range(2, 60):
        smooth_mods.add(m)
    
    for modulus in sorted(smooth_mods):
        if not _time_ok():
            break
        for residue in range(min(modulus, 15)):
            start = residue if residue != 0 else modulus
            if start > 10000:
                continue
            arr = []
            val = start
            while val <= 10000 and len(arr) < 12:
                arr.append(val)
                val += modulus
            if len(arr) < 12:
                continue
            count = _count_distinct_primes(arr)
            if count < best_count:
                best_count = count
                best_arr = arr
                if best_count <= 4:
                    return best_arr
    return best_arr

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 80  # Leave 10s margin
    
    best_arr = list(range(1, 13))
    best_count = _count_distinct_primes(best_arr)
    
    # Strategy 1: Smooth number search with various prime sets
    prime_sets = [
        [2],
        [2, 3],
        [2, 5],
        [3, 5],
        [2, 3, 5],
        [2, 3, 7],
        [2, 5, 7],
        [2, 3, 5, 7],
        [2, 3, 5, 11],
        [2, 3, 7, 11],
        [2, 5, 7, 11],
        [3, 5, 7, 11],
        [2, 3, 5, 7, 11],
    ]
    
    for ps in prime_sets:
        if not _time_ok():
            break
        if len(ps) >= best_count:
            continue
        result = _try_smooth_search(ps, max_candidates=1500)
        if result is not None:
            count = _count_distinct_primes(result)
            if count < best_count:
                best_count = count
                best_arr = result
                if best_count <= 3:
                    return best_arr
    
    # Strategy 2: Residue class search
    res = _residue_class_search()
    if res is not None:
        count = _count_distinct_primes(res)
        if count < best_count:
            best_count = count
            best_arr = res
            if best_count <= 3:
                return best_arr
    
    # Strategy 3: Local search from multiple starting points
    rng = random.Random(123)
    for attempt in range(200):
        if not _time_ok():
            break
        # Random start
        arr = rng.sample(range(1, 10001), 12)
        improved, count = _local_search(arr, max_iters=2000)
        if count < best_count:
            best_count = count
            best_arr = improved
            if best_count <= 3:
                return best_arr
    
    # Strategy 4: Structured random - pick numbers that are multiples of small numbers
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
        if not _time_ok():
            break
        multiples = [base * k for k in range(1, 10001 // base + 1)]
        if len(multiples) < 12:
            continue
        # Try a few random subsets
        for _ in range(20):
            if not _time_ok():
                break
            arr = rng.sample(multiples, 12)
            improved, count = _local_search(arr, max_iters=1000)
            if count < best_count:
                best_count = count
                best_arr = improved
                if best_count <= 3:
                    return best_arr
    
    # Strategy 5: Try to build set where all sums are even (all odd or all even)
    # All odd numbers: sums are even, so at least factor 2
    odds = list(range(1, 10001, 2))
    if len(odds) >= 12:
        for _ in range(50):
            if not _time_ok():
                break
            arr = rng.sample(odds, 12)
            improved, count = _local_search(arr, max_iters=1000)
            if count < best_count:
                best_count = count
                best_arr = improved
                if best_count <= 3:
                    return best_arr
    
    # Strategy 6: All numbers congruent mod small number
    for mod in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if not _time_ok():
            break
        for res in range(mod):
            if not _time_ok():
                break
            cls = [x for x in range(res if res != 0 else mod, 10001, mod)]
            if len(cls) < 12:
                continue
            for _ in range(10):
                if not _time_ok():
                    break
                arr = rng.sample(cls, 12)
                improved, count = _local_search(arr, max_iters=500)
                if count < best_count:
                    best_count = count
                    best_arr = improved
                    if best_count <= 3:
                        return best_arr
    
    return best_arr

import math
import time
from collections import defaultdict
from bisect import bisect_left, bisect_right

N = 210
MAX_N = 10210
MAX_K = 100
SMALL_PRIMES = [2, 3, 5, 7]

def _is_square(n):
    r = math.isqrt(n)
    return r * r == n

def _sieve_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def _factorize(n, primes):
    d = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            d[p] = e
    if n > 1:
        d[n] = 1
    return d

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0

    primes = _sieve_primes(MAX_N)
    
    # Precompute parity for all candidates
    parity = {}
    for n in range(211, MAX_N + 1):
        d = _factorize(n, primes)
        mask = 0
        for i, p in enumerate(SMALL_PRIMES):
            if d.get(p, 0) & 1:
                mask |= (1 << i)
        large = frozenset(p for p, e in d.items() if p not in SMALL_PRIMES and e & 1)
        parity[n] = (mask, large)

    TARGET_MASK = 0b1111
    TARGET_LARGE = frozenset()

    best_val = None
    best_set = None

    # Build parity dictionary
    parity_dict = defaultdict(list)
    for n in range(211, MAX_N + 1):
        m, lg = parity[n]
        parity_dict[(m, lg)].append(n)
    
    for key in parity_dict:
        parity_dict[key].sort()

    # Helper: find smallest element in sorted list not equal to any in exclude_set, >= min_val
    def find_smallest_excluding(nums, exclude_set, min_val=211):
        idx = bisect_left(nums, min_val)
        for i in range(idx, len(nums)):
            if nums[i] not in exclude_set:
                return nums[i]
        return None

    # Check k=1
    for n in range(211, MAX_N + 1):
        if time.time() > DEADLINE: break
        m, lg = parity[n]
        if m == TARGET_MASK and lg == TARGET_LARGE:
            val = n - 210
            if best_val is None or val < best_val:
                best_val = val
                best_set = (n,)

    # Check k=2 with binary search
    for a in range(211, MAX_N + 1):
        if time.time() > DEADLINE: break
        if best_val is not None and a - 210 >= best_val:
            break
        ma, la = parity[a]
        target_m = TARGET_MASK ^ ma
        target_l = TARGET_LARGE ^ la
        target_key = (target_m, target_l)
        
        if target_key in parity_dict:
            nums = parity_dict[target_key]
            b = find_smallest_excluding(nums, {a})
            if b is not None:
                val = max(a, b) - 210
                if best_val is None or val < best_val:
                    best_val = val
                    best_set = (a, b)

    # Check k=3: efficient approach
    # For each a, for each b > a, check if c exists
    # Use pruning: if best_val is set, only consider a, b < 210 + best_val
    if best_val is not None:
        limit = 210 + best_val
    else:
        limit = MAX_N
    
    for a in range(211, limit + 1):
        if time.time() > DEADLINE: break
        if best_val is not None and a - 210 >= best_val:
            break
        ma, la = parity[a]
        for b in range(a + 1, limit + 1):
            if time.time() > DEADLINE: break
            if best_val is not None and b - 210 >= best_val:
                break
            mb, lb = parity[b]
            target_m = TARGET_MASK ^ ma ^ mb
            target_l = TARGET_LARGE ^ la ^ lb
            target_key = (target_m, target_l)
            
            if target_key in parity_dict:
                nums = parity_dict[target_key]
                c = find_smallest_excluding(nums, {a, b})
                if c is not None:
                    val = max(a, b, c) - 210
                    if best_val is None or val < best_val:
                        best_val = val
                        best_set = (a, b, c)
                        limit = 210 + best_val

    # Check k=4: pair-pair matching
    # Precompute all pairs with their combined parity and max
    # Then for each pair (a,b), find pair (c,d) such that combined parity matches target
    # This is O(n^2) to build pairs, then for each pair look up complement
    
    if best_val is not None:
        limit4 = 210 + best_val
    else:
        limit4 = MAX_N
    
    # Build pair parity dictionary: key -> list of (max_val, a, b)
    pair_dict = defaultdict(list)
    
    for a in range(211, limit4 + 1):
        if time.time() > DEADLINE: break
        ma, la = parity[a]
        for b in range(a + 1, limit4 + 1):
            if time.time() > DEADLINE: break
            mb, lb = parity[b]
            pm = ma ^ mb
            pl = la ^ lb
            pair_dict[(pm, pl)].append((max(a, b), a, b))
    
    # For each pair (a,b), find complement pair (c,d) with parity = TARGET ^ pair_parity
    # and all four elements distinct
    for key, pairs in pair_dict.items():
        if time.time() > DEADLINE: break
        pm, pl = key
        comp_m = TARGET_MASK ^ pm
        comp_l = TARGET_LARGE ^ pl
        comp_key = (comp_m, comp_l)
        
        if comp_key not in pair_dict:
            continue
        
        comp_pairs = pair_dict[comp_key]
        
        for max1, a, b in pairs:
            if time.time() > DEADLINE: break
            if best_val is not None and max1 >= best_val:
                continue
            for max2, c, d in comp_pairs:
                if time.time() > DEADLINE: break
                if best_val is not None and max(max1, max2) >= best_val:
                    continue
                # Check all four distinct
                if len({a, b, c, d}) == 4:
                    val = max(max1, max2) - 210
                    if best_val is None or val < best_val:
                        best_val = val
                        best_set = (a, b, c, d)

    if best_set is not None:
        prod = N
        for x in best_set:
            prod *= x
        if _is_square(prod) and len(set(best_set)) == len(best_set) and all(
            isinstance(x, int) and not isinstance(x, bool) and 210 < x <= MAX_N for x in best_set
        ):
            return [int(x) for x in best_set]

    # Fallback
    return [840]

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result: {result}")
    print(f"Max - 210: {max(result) - 210}")
    prod = 210
    for x in result:
        prod *= x
    print(f"Product is square: {_is_square(prod)}")
    print(f"Time: {t1 - t0:.2f}s")

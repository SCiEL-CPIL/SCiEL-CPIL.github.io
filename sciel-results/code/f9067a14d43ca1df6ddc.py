import math
import time
from collections import defaultdict

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
    DEADLINE = t_start + 80.0

    primes = _sieve_primes(MAX_N)
    
    # Precompute parity for all candidates
    # parity[n] = (mask, large_set)
    # mask: 4 bits for 2,3,5,7
    # large_set: frozenset of large primes with odd exponent
    parity = {}
    for n in range(211, MAX_N + 1):
        d = _factorize(n, primes)
        mask = 0
        for i, p in enumerate(SMALL_PRIMES):
            if d.get(p, 0) & 1:
                mask |= (1 << i)
        large = frozenset(p for p, e in d.items() if p not in SMALL_PRIMES and e & 1)
        parity[n] = (mask, large)

    TARGET_MASK = 0b1111 # 210 = 2*3*5*7, all exponents 1
    TARGET_LARGE = frozenset()

    best_val = None
    best_set = None

    # Strategy: Binary search on K = max - 210
    # For a given K, check if there is a valid set with max <= 210 + K
    # Since we want to minimize max, we can iterate K from 1 upwards and stop at first valid solution.
    # However, 1-element solutions are rare. 2-element solutions are more common.
    # Let's first check 1-element solutions for all K.
    # Then 2-element solutions.
    
    # To speed up, we can group by large_set.
    # For a fixed max_limit, we only consider n <= max_limit.
    
    # Let's try to find the minimum max directly.
    # We know that if a solution exists with max M, it also exists with max M' > M (by adding more numbers? No, adding numbers changes the product).
    # Actually, the set of valid sets is not monotonic with respect to max.
    # But we are looking for the MINIMUM max.
    
    # Let's iterate possible max values from 211 upwards.
    # For each max_val, check if there is a valid set with all elements <= max_val.
    # This is expensive if done naively.
    
    # Alternative:
    # We already have a 2-element search. The current program finds the best 2-element pair.
    # Is it possible that a 3-element set has a smaller max than the best 2-element set?
    # Yes. For example, if the best 2-element pair is (1000, 1001), but there is a 3-element set (211, 212, 213).
    # So we need to check 3-element sets too.
    
    # Let's use a greedy approach for small sets (1, 2, 3, 4 elements).
    # For each k from 1 to 4:
    #   Find the set of k distinct integers in (210, 10210] whose product * 210 is a square, minimizing max.
    
    # Optimization:
    # For k=1: Check each n.
    # For k=2: Check pairs.
    # For k=3: Check triples.
    
    # The current program does k=1 and k=2.
    # Let's add k=3.
    
    # To find the best k=3 set:
    # We need a, b, c such that parity[a] ^ parity[b] ^ parity[c] == TARGET.
    # This means parity[c] == TARGET ^ parity[a] ^ parity[b].
    # We can iterate over pairs (a, b) and check if there is a c.
    # To minimize max(a, b, c), we should iterate in a smart way.
    
    # Let's precompute a dictionary: for each parity key, what are the numbers with that key?
    # key = (mask, large_set)
    # dict[key] = sorted list of numbers
    
    parity_dict = defaultdict(list)
    for n in range(211, MAX_N + 1):
        m, lg = parity[n]
        parity_dict[(m, lg)].append(n)
    
    # Sort lists for binary search
    for key in parity_dict:
        parity_dict[key].sort()
    
    # Check k=1
    for n in range(211, MAX_N + 1):
        if time.time() > DEADLINE: break
        m, lg = parity[n]
        if m == TARGET_MASK and lg == TARGET_LARGE:
            val = n - 210
            if best_val is None or val < best_val:
                best_val = val
                best_set = (n,)
    
    # Check k=2
    # We want to minimize max(a, b).
    # Iterate a from 211 upwards.
    # For each a, find b such that parity[b] == TARGET ^ parity[a].
    # We want the smallest b > 210, b != a, such that b exists.
    # To minimize max(a, b), we should pick the smallest possible b.
    # But we also need to consider that a itself might be large.
    # Let's iterate a from 211 to MAX_N.
    # For each a, compute target_parity.
    # Find the smallest b in parity_dict[target_parity] such that b != a.
    # If b exists, candidate max is max(a, b).
    # Update best.
    
    for a in range(211, MAX_N + 1):
        if time.time() > DEADLINE: break
        ma, la = parity[a]
        target_m = TARGET_MASK ^ ma
        target_l = TARGET_LARGE ^ la  # Symmetric difference for sets
        # Wait, XOR of frozensets is not symmetric difference directly in Python like that.
        # frozenset ^ frozenset is symmetric difference.
        target_key = (target_m, target_l)
        
        if target_key in parity_dict:
            nums = parity_dict[target_key]
            # Find smallest b != a
            # Since nums is sorted, we can check the first few.
            # If a is in nums, skip it.
            b = None
            for x in nums:
                if x != a:
                    b = x
                    break
            if b is not None:
                val = max(a, b) - 210
                if best_val is None or val < best_val:
                    best_val = val
                    best_set = (a, b)
    
    # Check k=3
    # We want to minimize max(a, b, c).
    # Iterate a from 211 upwards.
    # For each a, we need b, c such that parity[b] ^ parity[c] == TARGET ^ parity[a].
    # This is similar to k=2, but the target is dynamic.
    # Let target_parity_for_bc = TARGET ^ parity[a].
    # We need to find the pair (b, c) with minimum max(b, c) such that parity[b] ^ parity[c] == target_parity_for_bc.
    # This is expensive to do for every a.
    
    # Optimization:
    # Precompute the best pair for each possible target parity?
    # There are 2^4 * 2^(num_large_primes) possible parities. Too many.
    
    # Alternative:
    # Iterate a from 211 upwards.
    # For each a, iterate b from a+1 upwards.
    # Check if there exists c such that parity[c] == TARGET ^ parity[a] ^ parity[b].
    # If yes, candidate max is max(a, b, c).
    # Stop when max(a, b, c) >= best_val.
    
    if best_val is not None:
        limit = 210 + best_val
    else:
        limit = MAX_N
        
    for a in range(211, limit + 1):
        if time.time() > DEADLINE: break
        if a - 210 >= best_val: break # Since a is the smallest in the set, max >= a. If a >= 210 + best_val, no improvement.
        ma, la = parity[a]
        for b in range(a + 1, limit + 1):
            if time.time() > DEADLINE: break
            if b - 210 >= best_val: break # max >= b
            mb, lb = parity[b]
            target_m = TARGET_MASK ^ ma ^ mb
            target_l = TARGET_LARGE ^ la ^ lb
            target_key = (target_m, target_l)
            
            if target_key in parity_dict:
                nums = parity_dict[target_key]
                # Find c != a, b
                c = None
                for x in nums:
                    if x != a and x != b:
                        c = x
                        break
                if c is not None:
                    val = max(a, b, c) - 210
                    if best_val is None or val < best_val:
                        best_val = val
                        best_set = (a, b, c)
                        # Update limit
                        limit = 210 + best_val

    # Check k=4?
    # Might be too slow.
    
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

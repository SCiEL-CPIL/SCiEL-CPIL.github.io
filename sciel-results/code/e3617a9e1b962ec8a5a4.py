"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _sieve_prime_factors(max_n):
    """Precompute distinct prime factors for all numbers up to max_n using sieve."""
    factors = [set() for _ in range(max_n + 1)]
    for d in range(2, max_n + 1):
        if not factors[d]:
            for m in range(d, max_n + 1, d):
                factors[m].add(d)
    return factors

def _count_distinct_primes(arr, pf):
    """Count distinct primes dividing all pairwise sums using precomputed factors."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= pf[s]
    return len(primes)

def _find_clique(candidates, compat, target_size, deadline):
    """
    Greedy + backtracking to find a clique of size target_size.
    candidates: list of candidate numbers
    compat: list of sets, compat[i] = set of indices j where candidates[i]+candidates[j] has only allowed primes
    Returns a list of indices forming a clique, or None.
    """
    n = len(candidates)
    
    def backtrack(current, start_idx):
        if time.time() > deadline:
            return None
        if len(current) == target_size:
            return current[:]
        # Pruning: not enough candidates left
        if start_idx + (target_size - len(current)) > n:
            return None
        for i in range(start_idx, n):
            if time.time() > deadline:
                return None
            # Check compatibility with all in current
            ok = True
            for c in current:
                if i not in compat[c]:
                    ok = False
                    break
            if not ok:
                continue
            current.append(i)
            result = backtrack(current, i + 1)
            if result is not None:
                return result
            current.pop()
        return None
    
    # Greedy initialization: pick nodes with highest degree
    degrees = [len(compat[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Try greedy first
    current = []
    for i in order:
        if time.time() > deadline:
            break
        ok = True
        for c in current:
            if i not in compat[c]:
                ok = False
                break
        if ok:
            current.append(i)
            if len(current) == target_size:
                return current
    
    # Backtracking from greedy start
    if len(current) < target_size:
        result = backtrack(current, max(current) + 1 if current else 0)
        if result is not None:
            return result
    
    # Try different starting points
    for start_i in order[:50]:
        if time.time() > deadline:
            break
        current = [start_i]
        result = backtrack(current, start_i + 1)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Precompute prime factors for all sums (max sum = 20000)
    pf = _sieve_prime_factors(20000)
    
    best_arr = list(range(1, 13))
    best_count = _count_distinct_primes(best_arr, pf)
    
    # Target prime sets to try, ordered by preference (fewer primes first)
    target_prime_sets = [
        {2, 3, 5, 7},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {2, 3, 5, 11},
        {2, 3, 7, 11},
        {2, 5, 7, 11},
        {2, 3, 5, 7, 11},
    ]
    
    for target_primes in target_prime_sets:
        if time.time() > deadline:
            break
        
        # Filter candidates: numbers 1..10000 where every pairwise sum with another candidate
        # has all prime factors in target_primes
        # First, precompute which numbers have sums with only target primes
        # For a number x, we need: for all y in our set, x+y has only primes from target_primes
        
        # Build candidate list: all numbers 1..10000
        candidates = list(range(1, 10001))
        
        # For efficiency, precompute for each number x, which other numbers y make x+y
        # have only target prime factors
        # This is expensive for 10000 candidates, so let's be smarter
        
        # Key insight: if x and y are both ≡ r mod p for each p in target_primes,
        # then x+y ≡ 2r mod p. We need 2r ≡ 0 mod p for all p, OR x+y is a product of
        # only target primes.
        
        # Better approach: directly check compatibility using precomputed factors
        # But 10000 choose 2 is too many. Let's use a smarter filtering.
        
        # Strategy: group numbers by their residue pattern modulo small primes in target
        # For each prime p in target_primes, x+y must be divisible by p OR x+y has no factor p
        # Actually, we just need all prime factors of x+y to be in target_primes
        
        # Let's precompute a "compatible" check function
        def is_compatible(x, y):
            s = x + y
            if s > 20000:
                return False
            return pf[s].issubset(target_primes)
        
        # Greedy construction with backtracking
        # Start with a promising subset: numbers that are multiples of small products
        # Try numbers that are 0 mod M where M = product of some target primes
        
        target_primes_list = sorted(target_primes)
        
        # Try different modulus-based candidate subsets
        moduli_to_try = []
        # Single primes
        for p in target_primes_list:
            moduli_to_try.append(p)
        # Products of 2
        for i in range(len(target_primes_list)):
            for j in range(i+1, len(target_primes_list)):
                m = target_primes_list[i] * target_primes_list[j]
                if m <= 10000:
                    moduli_to_try.append(m)
        # Products of 3
        for i in range(len(target_primes_list)):
            for j in range(i+1, len(target_primes_list)):
                for k in range(j+1, len(target_primes_list)):
                    m = target_primes_list[i] * target_primes_list[j] * target_primes_list[k]
                    if m <= 10000:
                        moduli_to_try.append(m)
        
        found = False
        for mod in moduli_to_try:
            if time.time() > deadline:
                found = True
                break
            if mod > 10000:
                continue
            
            # Numbers ≡ 0 mod mod
            subset = [v for v in range(mod, 10001, mod)]
            if len(subset) < 12:
                continue
            
            # Build compatibility within this subset
            n_sub = len(subset)
            # Limit to first 200 candidates for tractability
            if n_sub > 200:
                subset = subset[:200]
                n_sub = 200
            
            compat = [set() for _ in range(n_sub)]
            for i in range(n_sub):
                for j in range(i+1, n_sub):
                    if is_compatible(subset[i], subset[j]):
                        compat[i].add(j)
                        compat[j].add(i)
            
            clique = _find_clique(subset, compat, 12, deadline)
            if clique is not None:
                arr = [subset[i] for i in clique]
                count = _count_distinct_primes(arr, pf)
                if count < best_count:
                    best_count = count
                    best_arr = arr
                    if best_count <= 4:
                        return best_arr
                found = True
                break
        
        if found or time.time() > deadline:
            break
    
    # Fallback: try a broader search if we haven't found a good solution
    if best_count > 4 and time.time() < deadline:
        import random
        random.seed(42)
        for trial in range(20):
            if time.time() > deadline:
                break
            # Random set with structure
            mod = random.choice([2, 3, 4, 5, 6, 7, 10, 12, 14, 15, 20, 21, 30, 42, 60, 70, 105, 210])
            r = random.randint(0, mod - 1)
            start = r % mod
            if start == 0:
                start = mod
            arr = []
            val = start
            while val <= 10000 and len(arr) < 12:
                arr.append(val)
                val += mod
            if len(arr) < 12:
                while len(arr) < 12:
                    x = random.randint(1, 10000)
                    if x not in arr:
                        arr.append(x)
            arr = arr[:12]
            count = _count_distinct_primes(arr, pf)
            if count < best_count:
                best_count = count
                best_arr = arr
                if best_count <= 4:
                    return best_arr
    
    return best_arr

# EVOLVE-BLOCK-END

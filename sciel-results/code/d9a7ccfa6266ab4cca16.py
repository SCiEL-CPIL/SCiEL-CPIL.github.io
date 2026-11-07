"""Program to find maximum sum of pairwise-coprime integers from 1..256."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd
from itertools import combinations

P = {'n': 256}

def _factorize(n):
    """Return list of prime factors of n (with multiplicity)."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def _build_data():
    """Build prime list and prime powers up to 256."""
    n = P['n']
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n+1) if is_prime[i]]
    
    # For each prime, find the largest prime power <= n
    prime_powers = {}
    for p in primes:
        pk = p
        while pk * p <= n:
            pk *= p
        prime_powers[p] = pk
    
    return primes, prime_powers

def _compute_best_group_value(primes_subset, n=256):
    """For a set of primes, find the max value <= n with exactly these prime factors (each appearing at least once)."""
    if not primes_subset:
        return 0
    
    # Product of primes in subset
    prod = 1
    for p in primes_subset:
        prod *= p
        if prod > n:
            return 0  # Can't even fit the base product
    
    # Now we can multiply by additional powers of these primes
    # Find max x <= n such that x is divisible by prod and all prime factors of x are in primes_subset
    best = 0
    # x = prod * y, where y's prime factors are subset of primes_subset
    # y <= n // prod
    max_y = n // prod
    
    # Generate all possible y values composed of primes in primes_subset
    # Since primes_subset is small (usually 1-3), we can enumerate
    def gen_values(primes_list, max_val):
        if not primes_list:
            yield 1
            return
        p = primes_list[0]
        pk = 1
        while pk <= max_val:
            for y in gen_values(primes_list[1:], max_val // pk):
                val = pk * y
                if val <= max_val:
                    yield val
            pk *= p
    
    for y in gen_values(list(primes_subset), max_y):
        val = prod * y
        if val > best:
            best = val
    return best

def construct():
    """Find maximum sum of pairwise-coprime integers from 1..256."""
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    primes, prime_powers = _build_data()
    n = P['n']
    
    # Sort primes: small primes first (they have more interaction potential)
    # Group small primes (those where combining with others might be beneficial)
    # A prime p is "small" if there exists another prime q such that p*q <= n
    # and the combined value exceeds p^k + q^m for optimal individual powers
    
    # Strategy: Use DP over subsets of "small" primes
    # Small primes: those p where p^2 <= n OR p*q <= n for some q
    # Actually, let's identify which primes can be part of a composite
    
    # For each prime p, its standalone value is prime_powers[p]
    # A group G of primes has value _compute_best_group_value(G)
    # The gain of grouping G is: value(G) - sum(prime_powers[p] for p in G)
    # We only want to group if gain > 0
    
    # Determine which primes are "active" (can potentially be grouped)
    # A prime p is active if there exists a subset G containing p with gain > 0
    # Simplification: only primes p where p * (some other prime) <= n can be in composites
    # For p > sqrt(n) ≈ 16, p*q > n for q >= 2, so p can only be standalone
    # Wait: p*q <= n means q <= n/p. For p=17, q <= 256/17 ≈ 15. So q could be 2,3,5,7,11,13.
    # But p=17, q=2: 17*2=34. Value of {17,2}: max x<=256 with factors {2,17}. 
    #   17*2=34, 17*4=68, 17*8=136, 17*16=272>256. So max is 136.
    #   Standalone: 17^1=17, 2^7=128. Sum=145. Group value=136 < 145. Not beneficial.
    
    # Let's be more systematic: only consider primes p where p <= n/2 (so at least p*2 <= n)
    # That's p <= 128. But even among those, most groupings aren't beneficial.
    
    # Key insight: for most primes, the prime power dominates any composite.
    # The main beneficial groupings involve very small primes (2,3,5,7,11,13)
    
    # Let's identify "small" primes as those <= 13 (where composites can be large)
    small_primes = [p for p in primes if p <= 13]
    large_primes = [p for p in primes if p > 13]
    
    # All large primes are standalone (their prime power is just p itself, since p^2 > 256 for p > 16)
    # For p in [17, 256], prime_powers[p] = p (since p^2 > 256)
    
    # For small primes, we need to decide whether to group them or keep standalone
    # Use DP over subsets of small_primes
    
    m = len(small_primes)
    
    # Precompute value for each subset of small_primes
    subset_values = {}
    # For each subset, compute the best value
    # 2^m subsets where m = number of small primes (primes <= 13: 2,3,5,7,11,13 -> m=6)
    
    for mask in range(1 << m):
        subset = [small_primes[i] for i in range(m) if mask & (1 << i)]
        if not subset:
            subset_values[mask] = 0
        else:
            subset_values[mask] = _compute_best_group_value(subset, n)
    
    # Standalone value for each small prime
    standalone = {small_primes[i]: prime_powers[small_primes[i]] for i in range(m)}
    
    # DP: dp[mask] = max total value for the set of small primes indicated by mask
    # Base: dp[0] = 0
    # Transition: dp[mask] = max over all submasks s of mask:
    #   dp[mask ^ s] + value(s)  where value(s) = max(subset_values[s], sum of standalone for p in s)
    
    # Actually, for each subset s, the value if we keep them all standalone is sum(standalone[p] for p in s)
    # The value if we group them is subset_values[s]
    # So effective_value(s) = max(subset_values[s], sum(standalone[p] for p in s))
    
    # But we also need to consider partial groupings within s
    # The DP handles this by considering all possible submasks
    
    dp = [0] * (1 << m)
    for mask in range(1, 1 << m):
        best = 0
        # Iterate over all submasks of mask
        sub = mask
        while sub > 0:
            # Value if we take sub as a group (or standalone)
            primes_in_sub = [small_primes[i] for i in range(m) if sub & (1 << i)]
            standalone_sum = sum(standalone[p] for p in primes_in_sub)
            group_val = subset_values[sub]
            eff_val = max(standalone_sum, group_val)
            candidate = dp[mask ^ sub] + eff_val
            if candidate > best:
                best = candidate
            sub = (sub - 1) & mask
        dp[mask] = best
    
    # Total for small primes
    small_total = dp[(1 << m) - 1]
    
    # Total for large primes (all standalone)
    large_total = sum(prime_powers[p] for p in large_primes)
    
    # Add 1 (allowed once)
    total = small_total + large_total + 1
    
    # Check time
    if time.time() > deadline:
        # Return a valid fallback (primes + 1)
        return [1] + [a for a in range(2, n+1) if all(a % d for d in range(2, isqrt(a)+1))]
    
    # Now we need to reconstruct the actual integers
    # We need to trace back through the DP to find which grouping was used
    
    # Reconstruct
    groups = []
    mask = (1 << m) - 1
    while mask > 0:
        best_sub = None
        best_candidate = -1
        sub = mask
        while sub > 0:
            primes_in_sub = [small_primes[i] for i in range(m) if sub & (1 << i)]
            standalone_sum = sum(standalone[p] for p in primes_in_sub)
            group_val = subset_values[sub]
            eff_val = max(standalone_sum, group_val)
            candidate = dp[mask ^ sub] + eff_val
            if candidate > best_candidate:
                best_candidate = candidate
                best_sub = sub
            sub = (sub - 1) & mask
        
        if best_sub is None or best_sub == 0:
            break
            
        primes_in_sub = [small_primes[i] for i in range(m) if best_sub & (1 << i)]
        standalone_sum = sum(standalone[p] for p in primes_in_sub)
        group_val = subset_values[best_sub]
        
        if group_val >= standalone_sum and len(primes_in_sub) > 1:
            # Use the grouped value
            groups.append((primes_in_sub, group_val))
        else:
            # Keep standalone
            for p in primes_in_sub:
                groups.append(([p], standalone[p]))
        
        mask ^= best_sub
    
    # Build result list
    result = [1]
    for primes_in_group, val in groups:
        if len(primes_in_group) == 1:
            result.append(prime_powers[primes_in_group[0]])
        else:
            result.append(val)
    
    for p in large_primes:
        result.append(prime_powers[p])
    
    # Verify: all distinct, in 1..256, pairwise coprime
    # (Trust the construction but add a safety check)
    if time.time() < deadline - 1:
        result_set = set(result)
        if len(result_set) != len(result):
            # Duplicates found, fallback
            return [1] + [a for a in range(2, n+1) if all(a % d for d in range(2, isqrt(a)+1))]
        
        if any(x < 1 or x > n for x in result):
            return [1] + [a for a in range(2, n+1) if all(a % d for d in range(2, isqrt(a)+1))]
        
        # Check pairwise coprime
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                if gcd(result[i], result[j]) != 1:
                    return [1] + [a for a in range(2, n+1) if all(a % d for d in range(2, isqrt(a)+1))]
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END

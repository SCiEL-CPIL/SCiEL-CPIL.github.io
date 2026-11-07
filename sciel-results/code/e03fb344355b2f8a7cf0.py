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
    deadline = start_time + 80  # safety margin before 90s limit
    
    primes, prime_powers = _build_data()
    n = P['n']
    
    # Identify all primes that could potentially be grouped
    # A prime p can be grouped if there exists another prime q such that p*q <= n
    # For p > n/2 = 128, no grouping is possible (p*2 > 256)
    # But even for p <= 128, most groupings aren't beneficial
    
    # Key insight: extend the "small primes" set to include primes up to ~100
    # where grouping with very small primes (2,3,5,7) might be beneficial
    # For p > 128, prime_powers[p] = p (since p^2 > 256), and p*2 > 256, so standalone
    
    # Let's use a broader set: all primes p where p * 2 <= n, i.e., p <= 128
    # But that's too many for subset DP. Instead, use a two-level approach:
    # Level 1: Very small primes (2,3,5,7) can be grouped with many primes
    # Level 2: For each combination of very small primes, decide which larger primes to include
    
    # Alternative: Use ILP for the full problem
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Build all candidate integers and their prime factor sets
        # An integer x from 1..256 is a candidate if it's >= 1
        # Two candidates conflict if gcd(x,y) > 1
        # We want to select a subset of non-conflicting candidates to maximize sum
        
        # Actually, let's think differently. Each selected integer has a set of prime factors.
        # Two integers conflict if they share any prime factor.
        # So we need to assign each prime to at most one selected integer.
        
        # Candidate integers: for each subset S of primes, the best value using exactly those primes
        # But there are too many subsets. Instead, enumerate all integers 1..256 as candidates.
        
        # For each integer x in 1..256, compute its set of prime factors
        # Build conflict graph: x and y conflict if gcd(x,y) > 1
        # Find maximum weight independent set
        
        # Since 256 is small, we can use branch and bound or ILP
        
        # ILP formulation:
        # Binary variable x_i for each integer i in 1..256
        # Maximize sum(i * x_i)
        # Subject to: for each prime p, sum(x_i for i where p divides i) <= 1
        # x_i in {0,1}
        
        # Number of primes up to 256: 54 primes
        # Number of variables: 256 (or 255 if we exclude 1, but 1 is always beneficial)
        # Number of constraints: 54 (one per prime)
        
        # This is a small ILP! Let's solve it.
        
        # Include 1: 1 has no prime factors, so it never conflicts. Always include it.
        # So we solve for 2..256 and add 1 to the result.
        
        candidates = list(range(2, n+1))
        num_candidates = len(candidates)
        
        # Get all primes up to n
        all_primes = [p for p in primes]
        num_primes = len(all_primes)
        
        # Build constraint matrix A: num_primes x num_candidates
        # A[j][i] = 1 if prime j divides candidate i
        A = np.zeros((num_primes, num_candidates))
        for i, x in enumerate(candidates):
            for j, p in enumerate(all_primes):
                if x % p == 0:
                    A[j][i] = 1
        
        # Objective: maximize sum of candidates
        c = np.array(candidates, dtype=float)  # milp minimizes, so negate
        c = -c
        
        # Constraints: A @ x <= 1 for each prime
        constraints = LinearConstraint(A, lb=0, ub=1)
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Solve
        res = milp(c, constraints=constraints, bounds=bounds, integrality=np.ones(num_candidates))
        
        if res.status == 0 and res.x is not None:
            selected = [candidates[i] for i in range(num_candidates) if res.x[i] > 0.5]
            result = [1] + selected
            
            # Verify
            result_set = set(result)
            if len(result_set) != len(result):
                # Duplicates - shouldn't happen but fallback
                return [1] + [a for a in range(2, n+1) if all(a % d for d in range(2, isqrt(a)+1))]
            
            if any(x < 1 or x > n for x in result):
                return [1] + [a for a in range(2, n+1) if all(a % d for d in range(2, isqrt(a)+1))]
            
            for i in range(len(result)):
                for j in range(i+1, len(result)):
                    if gcd(result[i], result[j]) != 1:
                        return [1] + [a for a in range(2, n+1) if all(a % d for d in range(2, isqrt(a)+1))]
            
            return [int(x) for x in result]
        else:
            # ILP failed, fall through to DP approach
            pass
    except Exception:
        pass
    
    # Fallback: DP approach with extended prime set
    # Use primes up to 100 that can form composites
    # For the DP, we need a manageable number of "active" primes
    
    # Let's try: include all primes p where p <= 50 in the DP
    # Primes <= 50: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47 -> 15 primes
    # 2^15 = 32768 subsets, which is manageable
    
    active_primes = [p for p in primes if p <= 50]
    inactive_primes = [p for p in primes if p > 50]
    
    m = len(active_primes)
    
    # Precompute value for each subset
    subset_values = {}
    for mask in range(1 << m):
        subset = [active_primes[i] for i in range(m) if mask & (1 << i)]
        if not subset:
            subset_values[mask] = 0
        else:
            subset_values[mask] = _compute_best_group_value(subset, n)
    
    standalone = {active_primes[i]: prime_powers[active_primes[i]] for i in range(m)}
    
    # DP
    dp = [0] * (1 << m)
    for mask in range(1, 1 << m):
        best = 0
        sub = mask
        while sub > 0:
            primes_in_sub = [active_primes[i] for i in range(m) if sub & (1 << i)]
            standalone_sum = sum(standalone[p] for p in primes_in_sub)
            group_val = subset_values[sub]
            eff_val = max(standalone_sum, group_val)
            candidate = dp[mask ^ sub] + eff_val
            if candidate > best:
                best = candidate
            sub = (sub - 1) & mask
        dp[mask] = best
    
    small_total = dp[(1 << m) - 1]
    large_total = sum(prime_powers[p] for p in inactive_primes)
    
    # Check time
    if time.time() > deadline:
        return [1] + [a for a in range(2, n+1) if all(a % d for d in range(2, isqrt(a)+1))]
    
    # Reconstruct
    groups = []
    mask = (1 << m) - 1
    while mask > 0:
        best_sub = None
        best_candidate = -1
        sub = mask
        while sub > 0:
            primes_in_sub = [active_primes[i] for i in range(m) if sub & (1 << i)]
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
            
        primes_in_sub = [active_primes[i] for i in range(m) if best_sub & (1 << i)]
        standalone_sum = sum(standalone[p] for p in primes_in_sub)
        group_val = subset_values[best_sub]
        
        if group_val >= standalone_sum and len(primes_in_sub) > 1:
            groups.append((primes_in_sub, group_val))
        else:
            for p in primes_in_sub:
                groups.append(([p], standalone[p]))
        
        mask ^= best_sub
    
    result = [1]
    for primes_in_group, val in groups:
        if len(primes_in_group) == 1:
            result.append(prime_powers[primes_in_group[0]])
        else:
            result.append(val)
    
    for p in inactive_primes:
        result.append(prime_powers[p])
    
    # Verify
    if time.time() < deadline - 1:
        result_set = set(result)
        if len(result_set) != len(result):
            return [1] + [a for a in range(2, n+1) if all(a % d for d in range(2, isqrt(a)+1))]
        
        if any(x < 1 or x > n for x in result):
            return [1] + [a for a in range(2, n+1) if all(a % d for d in range(2, isqrt(a)+1))]
        
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                if gcd(result[i], result[j]) != 1:
                    return [1] + [a for a in range(2, n+1) if all(a % d for d in range(2, isqrt(a)+1))]
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END

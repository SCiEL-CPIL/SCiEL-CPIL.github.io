"""
Constructs a maximal set of distinct integers in [2, 40] with distinct subset products.
Uses a constructive approach based on prime power decomposition:
For each prime base p <= 40, select a subset of powers {p^k} such that the exponents
have distinct subset sums. Since different primes are orthogonal in the product space,
the union of these sets across all primes yields a valid set with distinct subset products.
"""

def construct():
    limit = 40
    
    # Find all primes up to limit
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(limit)
    
    # For a given set of exponents, check if all subset sums are distinct
    # For small sets (size <= 4), we can enumerate all 2^k subsets
    def has_distinct_subset_sums(exponents):
        n = len(exponents)
        if n == 0:
            return True
        # Generate all subset sums
        sums = set()
        for mask in range(1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += exponents[i]
            if s in sums:
                return False
            sums.add(s)
        return True
    
    # For each prime, find the largest subset of powers p^1, p^2, ... <= limit
    # such that the exponents have distinct subset sums.
    # We try all subsets of the available exponents, starting from largest subsets.
    
    result = []
    
    for p in primes:
        # Find all powers p^k <= limit
        powers = []
        k = 1
        val = p
        while val <= limit:
            powers.append(val)
            k += 1
            val *= p
        
        if not powers:
            continue
        
        exponents = list(range(1, len(powers) + 1))
        
        # Try to find the largest subset of exponents with distinct subset sums
        # Sort by size descending
        best_subset = None
        n = len(exponents)
        
        # Try all subsets from largest to smallest
        for size in range(n, 0, -1):
            found = False
            # Iterate over all combinations of size 'size'
            from itertools import combinations
            for combo in combinations(range(n), size):
                exponents_subset = [exponents[i] for i in combo]
                if has_distinct_subset_sums(exponents_subset):
                    best_subset = exponents_subset
                    found = True
                    break
            if found:
                break
        
        if best_subset:
            for exp in best_subset:
                result.append(p ** exp)
    
    # Ensure we return a list of Python ints
    return [int(x) for x in result]

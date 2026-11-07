"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt

def _sieve_primes(n):
    """Return list of primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _factorize(n, primes):
    """Return set of prime factors of n."""
    factors = set()
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            factors.add(p)
            while n % p == 0:
                n //= p
    if n > 1:
        factors.add(n)
    return factors

def construct():
    n = 256
    primes = _sieve_primes(n)
    
    # Compute prime factor sets for each number in 2..n
    prime_factors = {}
    for i in range(2, n + 1):
        prime_factors[i] = _factorize(i, primes)
    
    # Condition: for distinct a, b, c in set, a does not divide b*c.
    # a divides b*c iff for every prime p, v_p(a) <= v_p(b) + v_p(c).
    # To check this efficiently, we can precompute exponent vectors.
    
    # Let's use a different approach:
    # For a candidate set S, the condition is: for all distinct a,b,c in S, a ∤ b*c.
    # This is equivalent to: for all a in S, and for all distinct b,c in S\{a}, a ∤ b*c.
    # Which is: for all a in S, a does not divide the product of any two distinct elements from S\{a}.
    
    # Greedy heuristic: prioritize numbers with large prime factors.
    # Numbers that are prime have the strongest constraint: if p is prime and p is in S,
    # then for any two other elements b,c in S, we need p ∤ b*c.
    # This means at most one of b,c can be divisible by p.
    
    # Let's try a greedy approach:
    # 1. Start with all numbers 2..n.
    # 2. Iteratively remove numbers that violate the condition.
    # Or better: build the set greedily, adding numbers that don't create conflicts.
    
    # Alternative: The set of primes themselves satisfies the condition?
    # If S is a set of primes, then for distinct primes p, q, r: p ∤ q*r since p, q, r are distinct primes.
    # So all primes in 2..256 form a valid set.
    # Number of primes in 2..256: let's count.
    
    # But we can do better by including some composites.
    # For example, if we include 4=2^2, then for any other b,c in S, we need 4 ∤ b*c.
    # This means v_2(b) + v_2(c) <= 1, so at most one of b,c can be even.
    
    # Let's use a maximum independent set approach on a conflict graph.
    # Two numbers x, y are in conflict if there exists a third number z in the candidate pool
    # such that z divides x*y. But this depends on the set we choose.
    
    # Simpler greedy: 
    # Order numbers by some criterion (e.g., descending number of prime factors, or descending value).
    # For each number, try to add it to the current set if it doesn't violate the condition.
    
    # To check if adding a new number a to current set S is valid:
    # 1. For all distinct b,c in S: a ∤ b*c (a doesn't divide product of any two existing)
    # 2. For all b in S: a ∤ a*b... wait, a, a, b are not all distinct.
    #    The condition is for THREE DISTINCT members. So when adding a to S:
    #    - For all distinct b,c in S: check a ∤ b*c
    #    - For all b in S: check b ∤ a*c for all c in S\{a,b}... this is complex.
    
    # Let me re-read: "For every THREE DISTINCT members a,b,c, require that a does not divide b*c."
    # So the condition is symmetric in the sense that for any ordered triple of distinct elements (a,b,c), a ∤ b*c.
    # This is equivalent to: for any three distinct elements a,b,c in S, none of a,b,c divides the product of the other two.
    
    # When adding a to S, we need to check:
    # 1. For all distinct b,c in S: a ∤ b*c
    # 2. For all b in S: for all c in S\{b}: b ∤ a*c (since a,b,c distinct)
    #    Wait, this is: for each pair (b,c) in S with b≠c, check b ∤ a*c and c ∤ a*b.
    
    # This is O(|S|^2) per addition, which is feasible for |S| up to a few hundred.
    
    # Let's implement this greedy approach.
    
    def divides_product(a, b, c):
        """Check if a divides b*c."""
        return (b * c) % a == 0
    
    def can_add(a, S):
        """Check if a can be added to set S without violating the condition."""
        S_list = list(S)
        # Check 1: a does not divide b*c for any distinct b,c in S
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                b = S_list[i]
                c = S_list[j]
                if divides_product(a, b, c):
                    return False
        # Check 2: for each b in S, b does not divide a*c for any c in S\{b}
        for b in S_list:
            for c in S_list:
                if c == b:
                    continue
                if divides_product(b, a, c):
                    return False
        return True
    
    # Order numbers: prioritize those with larger prime factors
    # Sort by: descending largest prime factor, then descending value
    def sort_key(x):
        pf = prime_factors[x]
        largest_pf = max(pf) if pf else 0
        return (-largest_pf, -x)
    
    candidates = list(range(2, n + 1))
    candidates.sort(key=sort_key)
    
    S = []
    for a in candidates:
        if can_add(a, S):
            S.append(a)
    
    # Convert to list of ints
    return list(S)
# EVOLVE-BLOCK-END

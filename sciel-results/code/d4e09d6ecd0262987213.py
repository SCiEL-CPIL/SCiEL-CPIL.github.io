"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Find 1..100 distinct integers > 210, <= 10210
    # such that 210 * product(selected) is a perfect square.
    # Minimize max(selected) - 210.
    
    # Key insight: 210 = 2*3*5*7 is squarefree.
    # For the product to be a perfect square, the combined exponent
    # of each prime in 210 * prod(selected) must be even.
    # 
    # We can use the squarefree kernel approach:
    # For each number n, compute its "signature" as the squarefree part
    # after removing factors of 210's primes.
    # 
    # Actually, a cleaner approach: 
    # Let P = 210. We need P * n1 * n2 * ... * nk to be a perfect square.
    # This means for every prime p, the total exponent in P * prod(ni) is even.
    # 
    # For each integer m > 210, define its "parity vector" with respect to
    # all primes. But we can simplify: the product P * n1 * ... * nk is a
    # perfect square iff the product of the squarefree kernels of P, n1, ..., nk
    # (where we multiply them and reduce) equals 1 in the multiplicative group
    # of squarefree numbers modulo squares.
    #
    # More practically: define sf(x) as the squarefree part of x.
    # Then P * n1 * ... * nk is a perfect square iff sf(P * n1 * ... * nk) = 1.
    # This is equivalent to: the product of sf(P), sf(n1), ..., sf(nk) has
    # all prime exponents even, i.e., their product in the group of squarefree
    # numbers (mod squares) is 1.
    #
    # Since 210 is squarefree, sf(210) = 210.
    #
    # Strategy: Search for small sets. Start with pairs, then triples, etc.
    # For a pair (a, b): we need sf(210 * a * b) = 1, i.e., sf(210*a) * sf(b) is a perfect square.
    # Since both are squarefree, sf(210*a) * sf(b) is a perfect square iff sf(210*a) == sf(b).
    # So for pairs: find a, b > 210 such that sf(210*a) == sf(b), and a != b, a,b <= 10210.
    #
    # For a pair, max(a,b) - 210 is the objective. We want to minimize this.
    # 
    # Let's precompute sf(n) for n from 211 to 10210.
    
    def compute_squarefree(n, primes):
        """Compute the squarefree part of n."""
        result = 1
        temp = n
        for p in primes:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                result *= p
        if temp > 1:
            result *= temp
        return result
    
    # Generate primes up to sqrt(10210) ~ 101, but we need primes up to 10210 for sf computation
    # Actually, for sf(n) where n <= 10210, we need primes up to 10210.
    # But for factoring, we only need primes up to sqrt(10210) ~ 101.
    
    def sieve_primes(limit):
        """Generate primes up to limit."""
        if limit < 2:
            return []
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # We need primes up to 10210 to handle the leftover factor in sf computation
    max_val = 10210
    primes = sieve_primes(max_val)
    
    # Precompute sf(n) for n from 211 to 10210
    sf = {}
    for n in range(211, max_val + 1):
        sf[n] = compute_squarefree(n, primes)
    
    # Also compute sf(210 * n) for n from 211 to 10210
    # But 210 * n can be up to 210 * 10210 = 2144100, so we need to compute sf for larger numbers
    # Alternatively, sf(210 * n) = sf(210) * sf(n) / gcd(sf(210), sf(n))^2 ... no, that's not right.
    # sf(a*b) is not simply sf(a)*sf(b) because of overlapping prime factors.
    # Better to compute sf(210*n) directly.
    
    # For the pair approach: we need sf(210*a) == sf(b)
    # Let's compute sf(210*a) for a in 211..10210
    # 210*a is at most 2144100, sqrt is about 1464, so we need primes up to ~1464
    
    primes_for_210n = sieve_primes(1500)
    
    def sf_210_times_n(n):
        """Compute squarefree part of 210 * n."""
        x = 210 * n
        result = 1
        temp = x
        for p in primes_for_210n:
            if p * p > temp:
                break
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                result *= p
        if temp > 1:
            result *= temp
        return result
    
    # Build a dictionary: sf_210a -> list of a values
    # Then for each b, check if sf(b) is in the dictionary and find a != b
    
    # Actually, let's think about this differently.
    # For pairs: sf(210*a) must equal sf(b).
    # We want to minimize max(a, b) - 210.
    # 
    # Let's iterate over possible max values M from 211 upward, and check if there exist
    # a, b in (210, M] with a != b and sf(210*a) == sf(b).
    # 
    # But this could be slow. Let's instead:
    # 1. For each a in 211..10210, compute key_a = sf(210*a)
    # 2. For each b in 211..10210, compute key_b = sf(b)
    # 3. If key_a == key_b and a != b, then (a, b) is a valid pair.
    # 4. Track the minimum max(a,b) over all valid pairs.
    
    # Map from squarefree key to list of a's
    a_by_key = {}
    for a in range(211, max_val + 1):
        key = sf_210_times_n(a)
        if key not in a_by_key:
            a_by_key[key] = []
        a_by_key[key].append(a)
    
    best_pair = None
    best_max = float('inf')
    
    for b in range(211, max_val + 1):
        key_b = sf[b]
        if key_b in a_by_key:
            for a in a_by_key[key_b]:
                if a != b:
                    m = max(a, b)
                    if m < best_max:
                        best_max = m
                        best_pair = (a, b)
    
    if best_pair is not None:
        return [best_pair[0], best_pair[1]]
    
    # If no pair found, try triples.
    # For triples (a, b, c): sf(210*a*b*c) = 1
    # This is harder. Let's try a different approach.
    # 
    # Actually, let's just return a known valid solution if pairs don't work.
    # But pairs should work. Let's verify with a simple case.
    # 
    # 210 = 2*3*5*7
    # If we pick a = 210*k^2 for some k, then sf(210*a) = sf(210 * 210 * k^2) = sf(210^2 * k^2) = 1.
    # So we need sf(b) = 1, meaning b is a perfect square.
    # b > 210, b <= 10210, b is a perfect square.
    # Smallest square > 210: 15^2 = 225.
    # a = 210 * k^2 > 210, so k >= 2: a = 210*4 = 840.
    # max(840, 225) = 840, so max - 210 = 630.
    
    # But maybe there's a better pair. Let's just return the best pair we found.
    
    # Fallback: return [840, 225]
    return [840, 225]

# EVOLVE-BLOCK-END

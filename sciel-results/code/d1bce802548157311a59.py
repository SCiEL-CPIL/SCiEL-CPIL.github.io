import math

def construct():
    limit = 10**18
    
    # Generate a sufficient list of primes
    def is_prime(n):
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

    primes = []
    candidate = 2
    while len(primes) < 100:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1

    def get_divisors(n):
        """Generate sorted divisors of n"""
        divisors = [1]
        # Simple factorization for small n? No, n is large.
        # But we know the prime factors if we track them.
        # For the score function, we can just factor n if it's small enough?
        # No, n is up to 1e18.
        # But in the greedy step, we are building n from primes.
        # We can pass the list of prime factors to the score function.
        pass

    def score_from_factors(factors):
        """
        factors: list of prime factors (distinct)
        Returns the number of coprime adjacent pairs in the sorted divisor list.
        """
        # Generate all divisors
        divisors = [1]
        for p in factors:
            new_divs = []
            for d in divisors:
                new_divs.append(d * p)
            divisors.extend(new_divs)
        
        divisors.sort()
        
        count = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                count += 1
        return count

    n = 1
    factors = []
    best_score = 0
    
    # Greedy search
    # We try to add one prime at a time.
    # We consider all primes in 'primes' that are not already in 'factors'.
    # We pick the one that maximizes the score.
    
    for _ in range(20): # Max 20 steps, though k will be ~15
        best_candidate = None
        max_score = -1
        best_p = None
        
        for p in primes:
            if p in factors:
                continue
            if n * p > limit:
                continue
            
            new_factors = factors + [p]
            # Optimization: If new_factors is large, score calc is expensive.
            # But we only do this ~100 times per step.
            s = score_from_factors(new_factors)
            
            if s > max_score:
                max_score = s
                best_p = p
                best_candidate = new_factors
        
        if best_p is None:
            break
        
        # Update
        n *= best_p
        factors = best_candidate
        best_score = max_score
        
    return n

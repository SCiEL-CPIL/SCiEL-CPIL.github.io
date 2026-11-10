def construct():
    import math
    import time
    
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Current best: first 15 primes
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    best_n = 1
    for p in base_primes:
        best_n *= p
    best_score = -1
    
    def compute_tau_perp(n):
        """Compute tau_perp: number of adjacent coprime pairs in sorted divisors."""
        # Generate all divisors
        divisors = [1]
        temp = n
        i = 2
        while i * i <= temp:
            if temp % i == 0:
                # Count exponent
                exp = 0
                while temp % i == 0:
                    temp //= i
                    exp += 1
                # Multiply existing divisors by i^0, i^1, ..., i^exp
                new_divs = list(divisors)
                for e in range(1, exp + 1):
                    for d in divisors:
                        new_divs.append(d * (i ** e))
                divisors = new_divs
                i += 1
            else:
                i += 1
        if temp > 1:
            new_divs = list(divisors)
            for d in divisors:
                new_divs.append(d * temp)
            divisors = new_divs
        
        divisors.sort()
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score
    
    # Verify baseline
    best_score = compute_tau_perp(best_n)
    
    # Try replacing some primes with others to see if we can do better
    # Generate candidate primes up to ~200
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i+2) == 0:
                return False
            i += 6
        return True
    
    candidates = [p for p in range(2, 300) if is_prime(p)]
    
    # Try swapping individual primes
    for idx in range(len(base_primes)):
        if time.time() > deadline:
            break
        orig = base_primes[idx]
        for new_p in candidates:
            if new_p == orig:
                continue
            test_primes = list(base_primes)
            test_primes[idx] = new_p
            # Check product <= 10^18
            prod = 1
            for p in test_primes:
                prod *= p
                if prod > 10**18:
                    break
            if prod > 10**18:
                continue
            score = compute_tau_perp(prod)
            if score > best_score:
                best_score = score
                best_n = prod
    
    return best_n

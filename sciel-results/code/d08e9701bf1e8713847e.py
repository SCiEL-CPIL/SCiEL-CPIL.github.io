import time
from math import gcd

def _sieve(n):
    """Generate primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def _divisors_from_factors(primes, exponents):
    """Generate all divisors of a number given its prime factorization."""
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            power = 1
            for i in range(e + 1):
                new_divs.append(d * power)
                power *= p
        divisors = new_divs
    return sorted(divisors)

def _compute_score(divisors):
    """Count adjacent coprime pairs in sorted divisor list."""
    count = 0
    for i in range(len(divisors) - 1):
        if gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def _try_candidate(primes, exponents):
    """Try a candidate number, return (score, n) or (0, 0) if invalid."""
    n = 1
    for p, e in zip(primes, exponents):
        n *= p ** e
    if n < 2 or n > 10**18:
        return (0, 0)
    divs = _divisors_from_factors(primes, exponents)
    score = _compute_score(divs)
    return (score, n)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial candidate from Inspiration 1
    # 2^5 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    primes_base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    exponents_base = [5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    best_score, best_n = _try_candidate(primes_base, exponents_base)
    best_primes = primes_base[:]
    best_exponents = exponents_base[:]
    
    if best_n < 2:
        # Fallback
        best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
        best_score = _compute_score(_divisors_from_factors([2,3,5,7,11,13,17,19], [1]*8))
        best_primes = [2,3,5,7,11,13,17,19]
        best_exponents = [1]*8

    # Generate a list of primes to potentially swap in
    all_primes = _sieve(200)
    prime_to_index = {p: i for i, p in enumerate(all_primes)}
    
    # Local Search
    # We will try to modify the exponents and the set of primes.
    # Moves:
    # 1. Increase exponent of p_i
    # 2. Decrease exponent of p_i
    # 3. Replace p_i with a smaller prime not in set (if any)
    # 4. Replace p_i with a larger prime (if fits)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try modifying exponents
        for i in range(len(best_primes)):
            if time.time() > deadline:
                break
            
            # Increase exponent
            p = best_primes[i]
            if best_n * p <= 10**18:
                new_exps = best_exponents[:]
                new_exps[i] += 1
                score, n = _try_candidate(best_primes, new_exps)
                if score > best_score:
                    best_score = score
                    best_n = n
                    best_exponents = new_exps
                    improved = True
                    # Restart loop to re-evaluate other moves with new base
                    break
            
            # Decrease exponent
            if best_exponents[i] > 1:
                new_exps = best_exponents[:]
                new_exps[i] -= 1
                score, n = _try_candidate(best_primes, new_exps)
                if score > best_score:
                    best_score = score
                    best_n = n
                    best_exponents = new_exps
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try modifying primes
        # Replace a prime with another prime
        # Strategy: Try to replace the largest prime with the next larger prime
        # Or replace a prime with a smaller prime to free up space for higher exponents?
        
        # 1. Try replacing the last prime with the next prime
        if len(best_primes) > 0:
            last_p = best_primes[-1]
            if last_p in prime_to_index:
                idx = prime_to_index[last_p]
                if idx + 1 < len(all_primes):
                    new_p = all_primes[idx + 1]
                    # Check if we can swap: n / last_p * new_p <= 10^18
                    if best_n // last_p * new_p <= 10**18:
                        new_primes = best_primes[:]
                        new_primes[-1] = new_p
                        score, n = _try_candidate(new_primes, best_exponents)
                        if score > best_score:
                            best_score = score
                            best_n = n
                            best_primes = new_primes
                            improved = True
                            break

        # 2. Try replacing a prime with a smaller prime (to potentially allow higher exponents later)
        # This is less direct, but let's try swapping any prime with a smaller one not in the set
        # This is complex, so we'll stick to exponent optimization and simple prime swaps for now.
        # However, we can try to "drop" the largest prime if it allows increasing exponents of small primes significantly.
        # This is effectively covered by "Decrease exponent" if we treat the prime as having exponent 1,
        # but dropping it completely changes the structure.
        
        # Let's try dropping the largest prime and see if we can increase exponents of 2 and 3.
        if len(best_primes) > 5: # Keep at least 5 primes
            p_drop = best_primes[-1]
            new_primes = best_primes[:-1]
            new_exps = best_exponents[:-1]
            
            # Try to redistribute the "budget" saved by dropping p_drop
            # We just try to increase exponents of 2 and 3 greedily
            # Actually, let's just evaluate the dropped version first
            score, n = _try_candidate(new_primes, new_exps)
            if score > best_score:
                best_score = score
                best_n = n
                best_primes = new_primes
                best_exponents = new_exps
                improved = True
                break

    return best_n

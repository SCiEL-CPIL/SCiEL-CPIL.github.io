import time
import math
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin
    
    # 1. Generate a list of small primes
    # We need enough primes to potentially reach a high count.
    # Product of first 15 primes is ~6e17. First 16 is ~3e19 (too big).
    # So max distinct primes is likely around 15 or 16 if we pick small ones.
    # Let's generate primes up to 1000 to be safe for combinations.
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(1000)
    
    # 2. Define tau_perp calculation
    def tau_perp(n):
        # Get divisors
        divs = [1]
        # Factorize n
        temp = n
        factors = []
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                factors.append(d)
                while temp % d == 0:
                    temp //= d
            else:
                d += 1
        if temp > 1:
            factors.append(temp)
            
        # Generate divisors
        for p in factors:
            new_divs = []
            for d in divs:
                new_divs.append(d * p)
            divs.extend(new_divs)
            
        divs.sort()
        
        # Count adjacent coprime pairs
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # 3. Strategy: Search for square-free numbers with max distinct primes
    # A square-free number n = p1 * p2 * ... * pk has 2^k divisors.
    # The divisors are all products of subsets.
    # To maximize tau_perp, we want the divisors to be "well-mixed" such that
    # adjacent divisors in sorted order are coprime.
    # Actually, for square-free n, gcd(d_i, d_{i+1}) == 1 iff they share no prime factors.
    
    # Let's evaluate the score for products of the first k primes.
    # This is the "greedy" baseline.
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 # Initial seed
    
    # Check products of first k primes
    prod = 1
    for k in range(1, len(primes)):
        prod *= primes[k-1]
        if prod > 10**18:
            break
        # This is a valid candidate
        s = tau_perp(prod)
        if s > best_score:
            best_score = s
            best_n = prod
        if time.time() > deadline:
            return best_n

    # Now, try to improve by swapping primes or using non-consecutive primes?
    # Generally, smaller primes allow more primes to fit, increasing 2^k.
    # The structure of coprime adjacencies depends on the specific primes.
    # Let's try random searches around the max k.
    
    max_k = 0
    prod = 1
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1
        
    # We know max_k is the maximum number of distinct primes.
    # Let's try random subsets of size max_k from the first ~50 primes.
    # But we must ensure product <= 10^18.
    
    # Heuristic: The first max_k primes is likely optimal for count.
    # But maybe replacing a large prime with a smaller non-consecutive one helps?
    # No, replacing a large prime with a smaller one only increases the product limit slack,
    # but we are already at the limit for count.
    # Wait, if we use the first max_k primes, we are using the smallest possible set of max_k primes.
    # Any other set of max_k primes will have a larger product.
    # So the set of first max_k primes is the ONLY set of size max_k that fits?
    # Not necessarily. We could drop the largest prime and add two smaller ones? No, count is fixed at max_k.
    # If we fix count k, the minimal product is the first k primes.
    # So any other combination of k primes has a larger product.
    # Does a larger product (larger primes) change the tau_perp score?
    # The score depends on the relative ordering of divisors.
    # Larger primes create larger "gaps" in the divisor lattice.
    # It is possible that non-consecutive primes yield better adjacency properties.
    
    # Let's do a limited random search for k = max_k and k = max_k - 1.
    
    for target_k in [max_k, max_k - 1]:
        if target_k < 2: continue
        attempts = 0
        while time.time() < deadline - 5.0 and attempts < 2000:
            attempts += 1
            # Pick target_k random primes from the first 200 primes
            # Ensure they are distinct
            selected = random.sample(primes[:200], target_k)
            prod = 1
            for p in selected:
                prod *= p
                if prod > 10**18:
                    break
            if prod > 10**18:
                continue
            
            s = tau_perp(prod)
            if s > best_score:
                best_score = s
                best_n = prod
            
    return best_n
# EVOLVE-BLOCK-END

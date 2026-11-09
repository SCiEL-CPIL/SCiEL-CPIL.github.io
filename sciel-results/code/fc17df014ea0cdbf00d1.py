import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Prime list for search
    def sieve(n):
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    # Generate primes up to a reasonable limit
    primes = sieve(10000)
    # Also add some larger primes
    large_primes = []
    p = 10007
    while len(large_primes) < 200 and p < 200000:
        if all(p % q != 0 for q in primes if q * q <= p):
            large_primes.append(p)
        p += 2
    all_primes = primes + large_primes
    
    def score_prime_set(prime_set):
        """Compute tau_perp for squarefree n = product of prime_set."""
        ps = sorted(prime_set)
        n = 1
        for p in ps:
            n *= p
        if n < 2 or n > 10**18:
            return -1
        # Generate all divisors
        divs = [1]
        for p in ps:
            new_divs = [d * p for d in divs]
            divs.extend(new_divs)
        divs.sort()
        # Count coprime adjacent pairs
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count
    
    # Start with seed
    current_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    current_score = score_prime_set(current_primes)
    
    best_primes = list(current_primes)
    best_score = current_score
    
    # Local search: try adding/removing/swapping primes
    random.seed(42)
    
    while time.time() < deadline:
        improved = False
        
        # Try single additions
        for p in all_primes:
            if p in current_primes:
                continue
            n = 1
            for q in current_primes:
                n *= q
            if n * p > 10**18:
                continue
            new_primes = current_primes + [p]
            s = score_prime_set(new_primes)
            if s > current_score:
                current_primes = new_primes
                current_score = s
                improved = True
                if s > best_score:
                    best_score = s
                    best_primes = list(current_primes)
                break
        
        if improved:
            continue
        
        # Try single removals
        for i, p in enumerate(current_primes):
            if len(current_primes) <= 2:
                break
            new_primes = current_primes[:i] + current_primes[i+1:]
            s = score_prime_set(new_primes)
            if s > current_score:
                current_primes = new_primes
                current_score = s
                improved = True
                if s > best_score:
                    best_score = s
                    best_primes = list(current_primes)
                break
        
        if improved:
            continue
        
        # Try swaps
        for i, p_out in enumerate(current_primes):
            if time.time() > deadline:
                break
            for p_in in all_primes:
                if p_in in current_primes:
                    continue
                n = 1
                for q in current_primes:
                    if q != p_out:
                        n *= q
                if n * p_in > 10**18:
                    continue
                new_primes = current_primes[:i] + [p_in] + current_primes[i+1:]
                s = score_prime_set(new_primes)
                if s > current_score:
                    current_primes = new_primes
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_primes = list(current_primes)
                    break
            if improved:
                break
        
        if not improved:
            # Stuck, try random restart with perturbation
            if len(best_primes) > 3:
                # Remove a random prime and try to rebuild
                idx = random.randint(0, len(best_primes) - 1)
                current_primes = best_primes[:idx] + best_primes[idx+1:]
                current_score = score_prime_set(current_primes)
            else:
                break
    
    # Return best found
    n = 1
    for p in best_primes:
        n *= p
    if n < 2:
        n = 2
    if n > 10**18:
        n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    return n

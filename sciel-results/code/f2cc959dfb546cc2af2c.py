import time
import math
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Sieve for primes up to 200
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(200)
    
    def calc_score(factors):
        """
        factors: list of (prime, exponent)
        Returns the score tau_perp(n)
        """
        # Generate divisors
        divs = [1]
        for p, e in factors:
            # Multiply existing divisors by p^1, p^2, ..., p^e
            current_divs = list(divs)
            for _ in range(e):
                current_divs = [d * p for d in current_divs]
                divs.extend(current_divs)
        
        # Sort divisors
        divs.sort()
        
        # Count adjacent coprime pairs
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    best_score = 0
    best_factors = []
    
    # Helper to check if product fits
    def fits(factors):
        prod = 1
        for p, e in factors:
            prod *= (p ** e)
            if prod > 10**18:
                return False
        return True

    # Generate candidate structures
    candidates = []
    
    # 1. Squarefree cases
    # Try first k primes
    for k in range(10, 16):
        fac = [(primes[i], 1) for i in range(k)]
        if fits(fac):
            candidates.append(fac)
            
    # 2. One prime with higher exponent
    # Base: first k primes, increase exponent of smallest primes
    for k in range(10, 15):
        base_fac = [(primes[i], 1) for i in range(k)]
        # Try increasing exponent of primes[0] (2)
        for e2 in range(2, 6):
            fac = list(base_fac)
            fac[0] = (primes[0], e2)
            if fits(fac):
                candidates.append(fac)
        # Try increasing exponent of primes[1] (3)
        for e3 in range(2, 5):
            fac = list(base_fac)
            fac[1] = (primes[1], e3)
            if fits(fac):
                candidates.append(fac)
        
        # Try increasing exponent of both 2 and 3
        for e2 in range(2, 5):
            for e3 in range(2, 4):
                fac = list(base_fac)
                fac[0] = (primes[0], e2)
                fac[1] = (primes[1], e3)
                if fits(fac):
                    candidates.append(fac)

    # 3. Specific insight: 2^4 * prod(3..43)
    # Let's explicitly add this and variations
    # Primes 3 to 43 are indices 1 to 13
    fac_insight = [(2, 4)] + [(primes[i], 1) for i in range(1, 14)]
    if fits(fac_insight):
        candidates.append(fac_insight)
    else:
        # Try 2^3
        fac_insight3 = [(2, 3)] + [(primes[i], 1) for i in range(1, 14)]
        if fits(fac_insight3):
            candidates.append(fac_insight3)
            
    # 4. Local search on exponents for a good base
    # Start with the best candidate found so far or a known good one
    # We will do a simple hill climbing on the exponents of the first few primes
    
    # First, evaluate all initial candidates
    for fac in candidates:
        if time.time() > deadline:
            break
        s = calc_score(fac)
        if s > best_score:
            best_score = s
            best_factors = list(fac)
            
    # If we have time, do a more systematic search
    # Focus on structures with ~40000 divisors
    # We can try to optimize the set of primes and exponents
    
    # Let's try a random search for structures
    # Represent structure by (num_primes, exponents of first few)
    
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        # Randomly perturb a good structure
        if not best_factors:
            continue
            
        current_fac = list(best_factors)
        
        # Randomly change an exponent
        if current_fac:
            idx = random.randint(0, min(3, len(current_fac)-1))
            old_p, old_e = current_fac[idx]
            new_e = old_e + random.randint(-1, 1)
            if new_e >= 1:
                current_fac[idx] = (old_p, new_e)
                if fits(current_fac):
                    s = calc_score(current_fac)
                    if s > best_score:
                        best_score = s
                        best_factors = list(current_fac)
        
        # Randomly swap a prime with a smaller/larger one (keep sorted by prime value for uniqueness)
        # This is complex, so we stick to exponent changes for now
        # Or add/remove a prime
        if random.random() < 0.5:
            # Try adding a new prime (larger than current max)
            max_prime = max(p for p, e in current_fac)
            new_primes = [p for p in primes if p > max_prime]
            if new_primes:
                new_p = random.choice(new_primes[:10]) # Small range
                # Check if we can add it with exponent 1
                # We need to remove something or just check fit
                # If we just add, product might exceed 10^18
                # Let's try adding and then reducing exponents of others if needed
                trial_fac = current_fac + [(new_p, 1)]
                if not fits(trial_fac):
                    # Try reducing exponents of 2, 3
                    modified = False
                    for i in range(min(2, len(trial_fac))):
                        if trial_fac[i][1] > 1:
                            p, e = trial_fac[i]
                            trial_fac[i] = (p, e-1)
                            if fits(trial_fac):
                                modified = True
                                break
                    if modified and fits(trial_fac):
                        s = calc_score(trial_fac)
                        if s > best_score:
                            best_score = s
                            best_factors = list(trial_fac)
            # Try removing a prime
            if len(current_fac) > 10:
                idx_to_remove = random.randint(len(current_fac)//2, len(current_fac)-1)
                trial_fac = current_fac[:idx_to_remove] + current_fac[idx_to_remove+1:]
                # We have room to increase exponents of small primes
                # Try increasing exponent of 2
                if trial_fac and trial_fac[0][0] == 2:
                    p, e = trial_fac[0]
                    trial_fac[0] = (p, e+1)
                if fits(trial_fac):
                    s = calc_score(trial_fac)
                    if s > best_score:
                        best_score = s
                        best_factors = list(trial_fac)

    # Final construction
    n = 1
    for p, e in best_factors:
        n *= (p ** e)
        
    return n

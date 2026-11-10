import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Generate primes up to 200
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(200)
    
    # We focus on the first 15-18 primes.
    # If we use more primes, the product grows fast.
    # Max product of first 15 primes is ~6e17.
    # Max product of first 16 primes is ~3e19 (too big).
    # So we are likely limited to 15 primes, or fewer with higher exponents.
    
    def calc_score(factors):
        """
        factors: list of (prime, exponent)
        Returns the score tau_perp(n)
        """
        # Generate divisors
        divs = [1]
        for p, e in factors:
            if e == 0:
                continue
            # Multiply existing divisors by p^1, p^2, ..., p^e
            current_divs = list(divs)
            for _ in range(e):
                current_divs = [d * p for d in current_divs]
                divs.extend(current_divs)
        
        # Sort divisors
        divs.sort()
        
        # Count adjacent coprime pairs
        # Optimization: use numpy for gcd? No, math.gcd is C-speed.
        # But looping in Python is slow.
        # Let's try to vectorize the gcd check if possible, or just accept the loop.
        # Number of divisors for 2^4 * 3 * ... * 43 is 5 * 2^13 = 40960.
        # Looping 40960 times with math.gcd is fast (< 0.1s).
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    def fits(factors):
        prod = 1
        for p, e in factors:
            if e == 0:
                continue
            prod *= (p ** e)
            if prod > 10**18:
                return False
        return True

    # Initial candidates
    # Based on Inspiration 1: 2^4 * prod(3..43)
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43
    # Indices 0 to 13.
    
    best_score = 0
    best_factors = []
    
    # Generate initial candidates
    candidates = []
    
    # 1. Squarefree first 15 primes
    fac_sq = [(primes[i], 1) for i in range(15)]
    if fits(fac_sq):
        candidates.append(fac_sq)
        
    # 2. 2^2 * first 15 primes (replaces 2^1 with 2^2, so effectively 2^2 * 3...47)
    # Product is 2 * (product of first 15). If 2^15 fits, 2^16 might not.
    # Product first 15 is ~6e17. 2 * 6e17 = 1.2e18 (fits). 4 * 6e17 = 2.4e18 (no).
    # So 2^2 * 3...47 fits.
    fac_2sq = [(primes[0], 2)] + [(primes[i], 1) for i in range(1, 15)]
    if fits(fac_2sq):
        candidates.append(fac_2sq)
        
    # 3. 2^3 * first 14 primes (drop 47)
    fac_2cu = [(primes[0], 3)] + [(primes[i], 1) for i in range(1, 14)]
    if fits(fac_2cu):
        candidates.append(fac_2cu)
        
    # 4. 2^4 * first 13 primes (drop 43, 47)
    fac_2qu = [(primes[0], 4)] + [(primes[i], 1) for i in range(1, 13)]
    if fits(fac_2qu):
        candidates.append(fac_2qu)
        
    # 5. 2^4 * 3^2 * first 12 primes (drop 41, 43, 47)
    fac_24_32 = [(primes[0], 4), (primes[1], 2)] + [(primes[i], 1) for i in range(2, 12)]
    if fits(fac_24_32):
        candidates.append(fac_24_32)

    # 6. 3^2 * first 14 primes (drop 47)
    fac_3sq = [(primes[0], 1), (primes[1], 2)] + [(primes[i], 1) for i in range(2, 14)]
    if fits(fac_3sq):
        candidates.append(fac_3sq)

    # Evaluate initial candidates
    for fac in candidates:
        if time.time() > deadline:
            break
        s = calc_score(fac)
        if s > best_score:
            best_score = s
            best_factors = [tuple(f) for f in fac] # Store as tuples for hashing if needed, but list is fine

    if not best_factors:
        # Fallback
        best_factors = [(2, 1)]
        best_score = calc_score(best_factors)

    # Local Search on Exponents
    # We will try to adjust exponents of the first few primes.
    # We assume the set of primes is fixed to the first K primes.
    # We can also try adding/removing the last prime.
    
    current_factors = [list(f) for f in best_factors]
    current_score = best_score
    
    # Define the set of primes we are working with
    # We'll stick to the primes present in current_factors, but allow adding the next prime if space permits.
    
    improved = True
    restarts = 0
    max_restarts = 50
    
    while time.time() < deadline:
        if improved:
            improved = False
            # Systematic pass: try changing each exponent by +/- 1
            for i in range(len(current_factors)):
                if time.time() > deadline: break
                p, e = current_factors[i]
                
                # Try increasing
                if e < 10: # Cap exponent to avoid huge loops
                    trial = [list(f) for f in current_factors]
                    trial[i][1] = e + 1
                    if fits(trial):
                        s = calc_score(trial)
                        if s > current_score:
                            current_factors = trial
                            current_score = s
                            improved = True
                            
                # Try decreasing
                if e > 1:
                    trial = [list(f) for f in current_factors]
                    trial[i][1] = e - 1
                    if fits(trial):
                        s = calc_score(trial)
                        if s > current_score:
                            current_factors = trial
                            current_score = s
                            improved = True
                            
            # Try adding a new prime (the next one after the max prime in current set)
            if len(current_factors) < 20:
                max_prime = max(p for p, e in current_factors)
                next_prime_idx = primes.index(max_prime) + 1
                if next_prime_idx < len(primes):
                    new_p = primes[next_prime_idx]
                    trial = [list(f) for f in current_factors]
                    # To add a prime, we might need to reduce an exponent of a smaller prime
                    # Or just add it if it fits (unlikely if we are near limit)
                    # Let's try adding it with exponent 1, and if it doesn't fit, 
                    # try reducing the exponent of the smallest prime by 1.
                    trial.append([new_p, 1])
                    if not fits(trial):
                        # Try reducing exponent of first prime
                        if trial[0][1] > 1:
                            trial[0][1] -= 1
                            if fits(trial):
                                s = calc_score(trial)
                                if s > current_score:
                                    current_factors = trial
                                    current_score = s
                                    improved = True
                    else:
                        s = calc_score(trial)
                        if s > current_score:
                            current_factors = trial
                            current_score = s
                            improved = True

            # Try removing the largest prime and increasing exponent of a smaller one
            if len(current_factors) > 2:
                # Remove last
                removed_p, removed_e = current_factors[-1]
                trial = [list(f) for f in current_factors[:-1]]
                # Increase exponent of first prime
                if trial[0][1] < 10:
                    trial[0][1] += 1
                    if fits(trial):
                        s = calc_score(trial)
                        if s > current_score:
                            current_factors = trial
                            current_score = s
                            improved = True

            # Update best
            if current_score > best_score:
                best_score = current_score
                best_factors = [tuple(f) for f in current_factors]
                improved = True # Allow another pass

        else:
            # No improvement in systematic pass.
            # Random restart/perturbation
            restarts += 1
            if restarts > max_restarts:
                break
            
            # Perturb current best
            current_factors = [list(f) for f in best_factors]
            
            # Apply random changes
            num_changes = random.randint(1, 3)
            for _ in range(num_changes):
                idx = random.randint(0, len(current_factors)-1)
                delta = random.choice([-1, 1, 1]) # Bias towards increasing
                current_factors[idx][1] += delta
                if current_factors[idx][1] < 1:
                    current_factors[idx][1] = 1
            
            if not fits(current_factors):
                # Try to fix by reducing largest exponents
                for i in range(len(current_factors)-1, -1, -1):
                    if current_factors[i][1] > 1:
                        current_factors[i][1] -= 1
                        if fits(current_factors):
                            break
            
            if not fits(current_factors):
                # Revert
                current_factors = [list(f) for f in best_factors]
                current_score = best_score
                continue

            current_score = calc_score(current_factors)
            
            # Local climb from here
            improved_local = True
            while improved_local and time.time() < deadline:
                improved_local = False
                for i in range(len(current_factors)):
                    if time.time() > deadline: break
                    p, e = current_factors[i]
                    for delta in [1, -1]:
                        new_e = e + delta
                        if new_e < 1 or new_e > 10:
                            continue
                        trial = [list(f) for f in current_factors]
                        trial[i][1] = new_e
                        if fits(trial):
                            s = calc_score(trial)
                            if s > current_score:
                                current_factors = trial
                                current_score = s
                                improved_local = True
                                improved = True
                                break # Restart the for loop for this i? Or just continue? 
                                # Breaking inner loop, continuing outer.
            
            if current_score > best_score:
                best_score = current_score
                best_factors = [tuple(f) for f in current_factors]

    # Final construction
    n = 1
    for p, e in best_factors:
        n *= (p ** e)
        
    return n

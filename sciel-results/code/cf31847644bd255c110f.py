import math
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    LIMIT = 10**18
    
    # Primes to consider.
    # Product of first 15 primes (up to 47) is ~6.1e17.
    # Product of first 16 primes (up to 53) is ~3.2e19 > 10^18.
    # So we can have at most 15 distinct prime factors if we use the smallest ones.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    k = len(primes)
    
    def calc_score_from_exponents(exps):
        """
        Calculate tau_perp(n) for n = prod p_i^e_i.
        Returns (score, n).
        """
        n = 1
        divisors = [1]
        
        for i, e in enumerate(exps):
            if e == 0:
                continue
            p = primes[i]
            n *= (p ** e)
            if n > LIMIT:
                return -1, n
            
            # Update divisors
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divisors = new_divs
        
        if n < 2:
            return 0, n
            
        divisors.sort()
        
        score = 0
        prev = divisors[0]
        for i in range(1, len(divisors)):
            curr = divisors[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
            
        return score, n

    # Strategy: Systematic search over "dense" exponent configurations.
    # The optimal n is likely to have high exponents for small primes and 0 for large ones,
    # or be squarefree with the maximum number of small primes.
    
    best_score = -1
    best_n = 1
    best_exps = [0] * k
    
    # Generate candidates
    candidates = []
    
    # 1. Squarefree primorials with various subsets of the first 15 primes
    # We know 15 primes fit. Let's try all subsets of size 14, 15?
    # No, the "value" is in having *many* primes.
    # Let's try the full set of 15.
    candidates.append([1] * 15)
    
    # 2. Variations: Increase exponent of small primes, drop large primes.
    # If we drop prime 47, we have "space" for 47/2 = 23.5 -> can increase exp of 2 by 1?
    # n = 2^2 * 3 * ... * 43.
    # Let's generate candidates by starting with [1]*15 and applying "moves".
    
    base_exps = [1] * 15
    n_base = 1
    for p in primes:
        n_base *= p
        
    # Candidate 1: Base
    candidates.append(base_exps[:])
    
    # Candidate 2: Drop largest, bump smallest
    for drop_idx in range(14, 7, -1): # Drop primes 47, 43, 41, 37, 31, 29, 23
        for bump_idx in range(0, 3): # Bump 2, 3, 5
            exps = base_exps[:]
            if exps[drop_idx] == 0:
                continue
            # Check if we can bump
            # n_new = n_base / p[drop] * p[bump]
            # We need n_new <= LIMIT
            # n_base is ~6e17.
            # If we drop 47, n becomes ~1.3e16. We can multiply by 2^k easily.
            
            # Calculate max bump
            p_drop = primes[drop_idx]
            p_bump = primes[bump_idx]
            n_current = n_base // p_drop
            max_bump = 1
            while n_current * (p_bump ** max_bump) <= LIMIT:
                max_bump += 1
            max_bump -= 1
            
            # Try bumping by 1, 2, up to max_bump (but keep it reasonable, say up to 5)
            for b in range(1, min(max_bump, 5) + 1):
                exps = base_exps[:]
                exps[drop_idx] = 0
                exps[bump_idx] += b
                # Ensure we don't exceed simple bounds
                if exps[bump_idx] > 10: continue
                candidates.append(exps)
                
    # Candidate 3: Drop two large primes, bump two small ones
    for drop1 in range(14, 12, -1):
        for drop2 in range(drop1-1, 10, -1):
            for bump1 in range(0, 2):
                for bump2 in range(bump1, 2):
                    exps = base_exps[:]
                    if exps[drop1] == 0 or exps[drop2] == 0:
                        continue
                    exps[drop1] = 0
                    exps[drop2] = 0
                    n_current = n_base // (primes[drop1] * primes[drop2])
                    
                    p_b1 = primes[bump1]
                    p_b2 = primes[bump2]
                    
                    # Try bumping by 1 or 2
                    for b1 in [1, 2]:
                        for b2 in [0, 1, 2]:
                            if b1 == 0 and b2 == 0: continue
                            n_test = n_current * (p_b1 ** b1) * (p_b2 ** b2)
                            if n_test <= LIMIT:
                                exps_test = exps[:]
                                exps_test[bump1] += b1
                                exps_test[bump2] += b2
                                candidates.append(exps_test)

    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # Evaluate candidates
    for exps in unique_candidates:
        # Quick check: n <= LIMIT
        n_val = 1
        valid = True
        for i, e in enumerate(exps):
            if e > 0:
                n_val *= (primes[i] ** e)
                if n_val > LIMIT:
                    valid = False
                    break
        if not valid:
            continue
            
        score, n_final = calc_score_from_exponents(exps)
        if score > best_score:
            best_score = score
            best_n = n_final
            best_exps = exps[:]
            
        if time.time() > DEADLINE:
            break

    # If we have time left, do a small SA from the best found
    if time.time() < DEADLINE - 10:
        current_exps = best_exps[:]
        current_score = best_score
        current_n = best_n
        
        T = 5.0
        cooling_rate = 0.99
        
        # Precompute max exponents for safety
        max_exp = [0] * k
        for i, p in enumerate(primes):
            e = 1
            val = p
            while val * p <= LIMIT:
                val *= p
                e += 1
            max_exp[i] = e
            
        while time.time() < DEADLINE:
            i = random.randint(0, k - 1)
            delta = random.choice([-1, 1])
            new_exp = current_exps[i] + delta
            
            if new_exp < 0 or new_exp > max_exp[i]:
                continue
                
            p = primes[i]
            n_new = current_n
            if new_exp < current_exp:
                n_new //= (p ** (current_exp - new_exp))
            else:
                n_new *= (p ** (new_exp - current_exp))
                
            if n_new > LIMIT:
                continue
                
            old_exp = current_exps[i]
            current_exps[i] = new_exp
            score_new, _ = calc_score_from_exponents(current_exps)
            
            if score_new > current_score:
                current_score = score_new
                current_n = n_new
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_n
                    best_exps = current_exps[:]
            else:
                delta_score = score_new - current_score
                if delta_score == 0:
                    if random.random() < 0.5:
                        current_score = score_new
                        current_n = n_new
                else:
                    prob = math.exp(delta_score / T)
                    if random.random() < prob:
                        current_score = score_new
                        current_n = n_new
                    else:
                        current_exps[i] = old_exp
                        
            T *= cooling_rate
            if T < 0.01:
                T = 0.01

    return best_n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END

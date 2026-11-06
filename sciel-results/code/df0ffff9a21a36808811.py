import math
import time
from itertools import product

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin below 150s
    
    # Primes up to ~100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    max_val = 10**18
    
    def tau_perp(n):
        """Calculate the number of adjacent coprime pairs in sorted divisors of n."""
        if n < 2:
            return 0
        
        # Factorize n
        factors = {}
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors[p] = count
        if temp > 1:
            factors[temp] = 1
        
        if not factors:
            return 0
        
        # Generate all divisors
        divs = [1]
        for p, e in factors.items():
            current_size = len(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for i in range(current_size):
                    divs.append(divs[i] * factor)
        
        divs.sort()
        
        # Count adjacent coprime pairs
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    # Generate candidate exponent vectors
    candidates = []
    
    # Strategy 1: Numbers with many distinct small primes (exponent 1 each)
    # Product of first k primes <= 10^18
    prod = 1
    for i, p in enumerate(primes):
        if prod * p <= max_val:
            prod *= p
            candidates.append(prod)
        else:
            break
    
    # Strategy 2: Vary exponents on small primes
    # Focus on first 8-12 primes with small exponents
    def generate_candidates_dfs(idx, max_exp, current_n, current_tau, depth=0):
        if time.time() - start_time > TIME_LIMIT - 10:
            return
        
        if current_n > 1:
            candidates.append((current_tau, current_n))
        
        if idx >= len(primes) or depth > 15:
            return
        
        p = primes[idx]
        limit_n = max_val // current_n
        
        # Max exponent for this prime
        e_max_possible = 0
        temp = 1
        while temp * p <= limit_n and e_max_possible < max_exp:
            temp *= p
            e_max_possible += 1
        
        # Iterate exponents (start from 1 to keep numbers manageable)
        for e in range(1, min(e_max_possible, 4) + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            generate_candidates_dfs(idx + 1, e, next_n, next_tau, depth + 1)
    
    generate_candidates_dfs(0, 60, 1, 1)
    
    # Strategy 3: More aggressive - try specific patterns
    # Pattern: 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h * 23^i
    # with small exponents
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for item in candidates:
        if isinstance(item, tuple):
            n = item[1]
        else:
            n = item
        if n not in seen and 2 <= n <= max_val:
            seen.add(n)
            unique_candidates.append(n)
    
    # Sort by number of divisors (heuristic: more divisors may mean more coprime pairs)
    def num_divisors(n):
        count = 1
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                e = 0
                while temp % p == 0:
                    temp //= p
                    e += 1
                count *= (e + 1)
        if temp > 1:
            count *= 2
        return count
    
    unique_candidates.sort(key=num_divisors, reverse=True)
    
    # Evaluate top candidates
    best_n = unique_candidates[0] if unique_candidates else 2
    best_score = tau_perp(best_n)
    
    # Evaluate more candidates
    for n in unique_candidates[1:100]:
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
    
    # Local search: try adjusting exponents and replacing primes
    if time.time() - start_time < TIME_LIMIT - 20:
        # Get factorization of best_n
        def get_factors(n):
            factors = {}
            temp = n
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    count = 0
                    while temp % p == 0:
                        temp //= p
                        count += 1
                    factors[p] = count
            if temp > 1:
                factors[temp] = 1
            return factors
        
        best_factors = get_factors(best_n)
        
        # Multiple rounds of local search
        for round_num in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            improved = False
            local_candidates = []
            
            # Try increasing/decreasing exponents
            for p in list(best_factors.keys()):
                e = best_factors[p]
                # Try increasing
                new_n = best_n * p
                if new_n <= max_val:
                    local_candidates.append(new_n)
                # Try decreasing
                if e > 1:
                    new_n = best_n // p
                    if new_n >= 2:
                        local_candidates.append(new_n)
            
            # Try adding new primes
            for p in primes:
                if p not in best_factors:
                    new_n = best_n * p
                    if new_n <= max_val:
                        local_candidates.append(new_n)
            
            # Try replacing one prime with another
            for p_remove in list(best_factors.keys()):
                for p_add in primes:
                    if p_add != p_remove and p_add not in best_factors:
                        # Remove p_remove completely, add p_add
                        new_n = best_n // (p_remove ** best_factors[p_remove]) * p_add
                        if 2 <= new_n <= max_val:
                            local_candidates.append(new_n)
            
            # Evaluate local candidates
            for n in local_candidates:
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                score = tau_perp(n)
                if score > best_score:
                    best_score = score
                    best_n = n
                    best_factors = get_factors(best_n)
                    improved = True
                    break
            
            if not improved:
                break
    
    return best_n

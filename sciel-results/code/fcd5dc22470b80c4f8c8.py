import math
import time

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin below 150s
    
    def elapsed():
        return time.time() - start_time
    
    max_val = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def tau_perp(n):
        """Compute the score: number of adjacent coprime pairs in sorted divisors."""
        if n <= 1:
            return 0
        # Factorize n
        factors = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((p, count))
        if temp > 1:
            factors.append((temp, 1))
        
        # Generate divisors
        divs = [1]
        for p, e in factors:
            current_size = len(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for i in range(current_size):
                    divs.append(divs[i] * factor)
        
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    def heuristic_score(n):
        """Fast heuristic: approximate score based on factorization."""
        if n <= 1:
            return 0
        temp = n
        factor_count = 0
        total_exponents = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factor_count += 1
                total_exponents += count
        if temp > 1:
            factor_count += 1
            total_exponents += 1
        
        # Heuristic: more distinct primes and moderate exponents tend to give higher scores
        # The score is roughly proportional to the number of "transitions" between coprime divisors
        return factor_count * (total_exponents + 1)
    
    # Phase 1: Generate candidates via DFS
    candidates = []
    
    def dfs(idx, max_exp, current_n, current_tau, exponent_tuple):
        if elapsed() > TIME_LIMIT - 10:
            return
        
        if current_n > 1:
            candidates.append((current_tau, current_n, exponent_tuple))
        
        if idx >= len(primes):
            return
        
        p = primes[idx]
        limit_n = max_val // current_n
        
        # Calculate max exponent
        e_max_possible = 0
        temp_val = 1
        while temp_val * p <= limit_n and e_max_possible < max_exp:
            temp_val *= p
            e_max_possible += 1
        
        # Try e from 1 to e_max_possible (include e=0 by not adding this prime)
        for e in range(1, e_max_possible + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            next_exp = exponent_tuple + (e,)
            dfs(idx + 1, e, next_n, next_tau, next_exp)
            
            if elapsed() > TIME_LIMIT - 10:
                return
        
        # Also try skipping this prime (e=0) if we haven't used all primes
        if idx + 1 < len(primes):
            dfs(idx + 1, max_exp, current_n, current_tau, exponent_tuple + (0,))
    
    # Start DFS with reasonable bounds
    dfs(0, 60, 1, 1, ())
    
    # Phase 2: Also add some hand-crafted candidates
    # Square-free products of first k primes
    sf_n = 1
    for p in primes:
        if sf_n * p <= max_val:
            sf_n *= p
        else:
            break
    candidates.append((2 ** len([p for p in primes if sf_n % p == 0]), sf_n, (1,) * len([p for p in primes if sf_n % p == 0])))
    
    # High-exponent on small primes
    for base_exp in [10, 15, 20, 25, 30]:
        n = 2 ** base_exp
        if n <= max_val:
            candidates.append((base_exp + 1, n, (base_exp,)))
    
    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    
    # Phase 3: Evaluate candidates, prioritizing by heuristic
    # Sort by heuristic score descending, then by tau
    def sort_key(c):
        tau, n, exps = c
        h = heuristic_score(n)
        return (h, tau)
    
    candidates.sort(key=sort_key, reverse=True)
    
    # Evaluate top candidates
    best_n = candidates[0][1]
    best_score = -1
    eval_limit = min(len(candidates), 2000)
    
    for i in range(eval_limit):
        if elapsed() > TIME_LIMIT - 20:
            break
        tau, n, exps = candidates[i]
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
    
    # Phase 4: Local search - try modifications to best_n
    if elapsed() < TIME_LIMIT - 15:
        # Get factorization of best_n
        best_factors = {}
        temp = best_n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                best_factors[p] = count
        if temp > 1:
            best_factors[temp] = 1
        
        local_candidates = set()
        
        # Try adjusting exponents by +/- 1
        for p in list(best_factors.keys()):
            e = best_factors[p]
            # Increase
            new_n = best_n * p
            if new_n <= max_val:
                local_candidates.add(new_n)
            # Decrease
            if e > 1:
                new_n = best_n // p
                if new_n >= 2:
                    local_candidates.add(new_n)
        
        # Try adding new small primes
        for p in primes:
            if p not in best_factors:
                new_n = best_n * p
                if new_n <= max_val:
                    local_candidates.add(new_n)
        
        # Try prime swaps: replace one prime factor with another
        for p_remove in list(best_factors.keys()):
            e_remove = best_factors[p_remove]
            base = best_n // (p_remove ** e_remove)
            for p_add in primes:
                if p_add == p_remove:
                    continue
                if p_add in best_factors:
                    continue
                # Try adding p_add with exponent 1
                new_n = base * p_add
                if 2 <= new_n <= max_val:
                    local_candidates.add(new_n)
                # Try adding p_add with exponent 2
                new_n = base * p_add * p_add
                if 2 <= new_n <= max_val:
                    local_candidates.add(new_n)
        
        # Evaluate local candidates
        for n in local_candidates:
            if elapsed() > TIME_LIMIT - 5:
                break
            score = tau_perp(n)
            if score > best_score:
                best_score = score
                best_n = n
        
        # Second round of local search from new best
        if elapsed() < TIME_LIMIT - 8:
            best_factors2 = {}
            temp = best_n
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    count = 0
                    while temp % p == 0:
                        temp //= p
                        count += 1
                    best_factors2[p] = count
            if temp > 1:
                best_factors2[temp] = 1
            
            local_candidates2 = set()
            for p in list(best_factors2.keys()):
                e = best_factors2[p]
                new_n = best_n * p
                if new_n <= max_val:
                    local_candidates2.add(new_n)
                if e > 1:
                    new_n = best_n // p
                    if new_n >= 2:
                        local_candidates2.add(new_n)
            
            for p in primes:
                if p not in best_factors2:
                    new_n = best_n * p
                    if new_n <= max_val:
                        local_candidates2.add(new_n)
            
            for n in local_candidates2:
                if elapsed() > TIME_LIMIT - 3:
                    break
                score = tau_perp(n)
                if score > best_score:
                    best_score = score
                    best_n = n
    
    return best_n

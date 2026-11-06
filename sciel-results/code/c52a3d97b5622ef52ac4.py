import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    # Primes to consider - include more primes for better coverage
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_tau_perp = 0
    best_candidates = []
    LIMIT = 10**18
    
    # Helper to add candidate with factorization
    def add_candidate(n, tau_perp, factorization):
        nonlocal best_tau_perp, best_candidates
        if tau_perp > best_tau_perp:
            best_tau_perp = tau_perp
            best_candidates = [(n, tau_perp, factorization)]
        elif tau_perp == best_tau_perp:
            best_candidates.append((n, tau_perp, factorization))
        else:
            # Keep top 30 candidates by tau_perp
            if len(best_candidates) < 30:
                best_candidates.append((n, tau_perp, factorization))
                best_candidates.sort(key=lambda x: x[1], reverse=True)
                best_candidates = best_candidates[:30]
            else:
                if tau_perp > best_candidates[-1][1]:
                    best_candidates.append((n, tau_perp, factorization))
                    best_candidates.sort(key=lambda x: x[1], reverse=True)
                    best_candidates = best_candidates[:30]

    def calc_tau_perp_from_factors(factors):
        """Calculate tau_perp directly from factorization without re-factorizing."""
        # Generate divisors from factorization
        divisors = [1]
        for p, exp in factors:
            new_divs = []
            p_power = 1
            for i in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        
        divisors.sort()
        
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score

    def dfs(current_n, current_factors, max_exp, prime_idx):
        if time.time() > deadline:
            return
        
        # Option: stop adding primes - evaluate current candidate
        if current_n > 1:
            tau_perp = calc_tau_perp_from_factors(current_factors)
            add_candidate(current_n, tau_perp, list(current_factors))
            
        if prime_idx >= len(primes):
            return
            
        p = primes[prime_idx]
        
        # Calculate max possible exp for this prime
        limit_for_p = LIMIT // current_n
        if limit_for_p < p:
            return
            
        max_possible_exp = 0
        temp = 1
        while temp * p <= limit_for_p and max_possible_exp < max_exp:
            temp *= p
            max_possible_exp += 1
            
        if max_possible_exp == 0:
            return

        # Iterate exponents from max_possible_exp down to 1
        for exp in range(max_possible_exp, 0, -1):
            next_n = current_n * (p ** exp)
            next_factors = current_factors + [(p, exp)]
            
            dfs(next_n, next_factors, exp, prime_idx + 1)
            
            if time.time() > deadline:
                return

    # Start DFS
    dfs(1, [], 100, 0)
    
    if not best_candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19

    best_n = best_candidates[0][0]
    
    return best_n

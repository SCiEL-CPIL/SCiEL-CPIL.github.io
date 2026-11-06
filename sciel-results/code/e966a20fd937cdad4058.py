import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    best_tau = 0
    best_candidates = []
    LIMIT = 10**18
    
    # Helper to add candidate
    def add_candidate(n, tau):
        nonlocal best_tau, best_candidates
        if tau > best_tau:
            best_tau = tau
            best_candidates = [(n, tau)]
        elif tau == best_tau:
            best_candidates.append((n, tau))
        else:
            # Keep top 20 candidates by tau
            if len(best_candidates) < 20:
                best_candidates.append((n, tau))
                best_candidates.sort(key=lambda x: x[1], reverse=True)
                best_candidates = best_candidates[:20]
            else:
                # If new tau is better than the worst in the list
                if tau > best_candidates[-1][1]:
                    best_candidates.append((n, tau))
                    best_candidates.sort(key=lambda x: x[1], reverse=True)
                    best_candidates = best_candidates[:20]

    def dfs(current_n, current_tau, max_exp, prime_idx):
        if time.time() > deadline:
            return
        
        # Option: stop adding primes
        if current_n > 1:
            add_candidate(current_n, current_tau)
            
        if prime_idx >= len(primes):
            return
            
        p = primes[prime_idx]
        
        # Try exponents from max_exp down to 1
        # We must ensure current_n * p^exp <= LIMIT
        # Also, to keep exponents non-increasing, exp <= max_exp
        
        # Calculate max possible exp for this prime
        # p^exp <= LIMIT / current_n
        limit_for_p = LIMIT // current_n
        if limit_for_p < p:
            return
            
        max_possible_exp = 0
        temp = 1
        while temp * p <= limit_for_p and max_possible_exp < max_exp:
            temp *= p
            max_possible_exp += 1
            
        # If no exponent is possible, we can't add this prime with exp >= 1
        if max_possible_exp == 0:
             # But we might be able to skip this prime and try the next one?
             # The problem implies we use the first k primes.
             # If we skip a prime, the "first k primes" assumption for structure might break,
             # but generally, using smaller primes is better for maximizing divisors.
             # However, if we can't fit p^1, we definitely can't fit larger primes.
             # So we just return.
             return

        # Iterate exponents from max_possible_exp down to 1
        # Note: We want non-increasing exponents.
        # So next max_exp will be this exp.
        for exp in range(max_possible_exp, 0, -1):
            next_n = current_n * (p ** exp)
            next_tau = current_tau * (exp + 1)
            
            # Pruning: If next_n * p_next > LIMIT, we can't add more primes after this one?
            # Actually, we check in the recursive call.
            
            dfs(next_n, next_tau, exp, prime_idx + 1)
            
            if time.time() > deadline:
                return

    # Start DFS
    # Initial state: n=1, tau=1, max_exp=100 (large enough), prime_idx=0
    dfs(1, 1, 100, 0)
    
    if not best_candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19

    # Evaluate tau_perp for top candidates
    def calc_tau_perp(n):
        # Factorize n (we know the factors from construction, but let's do it generically or pass them)
        # For efficiency in this specific context, we can just generate divisors
        # But we don't have the factorization stored in the candidate tuple.
        # Let's re-factorize. It's fast.
        
        # Simple factorization
        factors = []
        temp_n = n
        p = 2
        while p * p <= temp_n:
            if temp_n % p == 0:
                exp = 0
                while temp_n % p == 0:
                    temp_n //= p
                    exp += 1
                factors.append((p, exp))
            p += 1 if p == 2 else 2 # Simple optimization, though primes are small
        
        if temp_n > 1:
            factors.append((temp_n, 1))
        
        # Generate divisors
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

    best_n = best_candidates[0][0]
    best_score = -1
    
    # Evaluate top candidates
    # Sort candidates by tau descending (already done)
    for n, tau in best_candidates:
        if time.time() > deadline:
            break
        try:
            score = calc_tau_perp(n)
            if score > best_score:
                best_score = score
                best_n = n
        except Exception:
            continue
            
    return best_n

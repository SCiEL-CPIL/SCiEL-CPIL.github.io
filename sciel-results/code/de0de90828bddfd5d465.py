import math
import time

def construct():
    start_time = time.time()
    deadline = 100.0 # 20s margin for 120s limit
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    num_primes = len(primes)
    max_m = 10**18
    
    candidates = []
    
    def dfs(idx, current_m, max_exp, exponents):
        if time.time() > deadline:
            return
        if idx == num_primes:
            divs = 1
            for e in exponents:
                divs *= (e + 1)
            candidates.append((divs, current_m, tuple(exponents)))
            return
        
        p = primes[idx]
        for e in range(max_exp, -1, -1):
            # Check if current_m * p^e <= max_m
            # We can break if p^e is too large, but we are going down, so we can't break early on large e.
            # We can calculate p^e and check.
            try:
                pe = p ** e
            except OverflowError:
                continue
                
            if pe > max_m // current_m:
                continue
                
            next_m = current_m * pe
            exponents.append(e)
            dfs(idx + 1, next_m, e, exponents)
            exponents.pop()
            
    dfs(0, 1, 60, [])
    
    # Sort by number of divisors descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_m = 1
    best_ks = []
    best_count = 0
    
    # Precompute powers and sigmas for primes
    p_pows = {}
    p_sigmas = {}
    for p in primes:
        p_pows[p] = [1] * 61
        p_sigmas[p] = [1] * 61
        for a in range(1, 61):
            p_pows[p][a] = p_pows[p][a-1] * p
            p_sigmas[p][a] = p_sigmas[p][a-1] + p_pows[p][a]
            
    # Evaluate top candidates
    for divs, m, exps in candidates:
        if time.time() > deadline:
            break
        if divs < best_count:
            continue
            
        # Generate divisors with exponent vectors
        divs_with_exps = [(1, (0,)*num_primes)]
        
        for i, p in enumerate(primes):
            e = exps[i]
            if e == 0:
                continue
            new_divs = []
            for d, exp_vec in divs_with_exps:
                for a in range(e + 1):
                    new_exp_vec = list(exp_vec)
                    new_exp_vec[i] = a
                    new_divs.append((d * p_pows[p][a], tuple(new_exp_vec)))
            divs_with_exps = new_divs
        
        count = 0
        ks = []
        for k, exp_vec in divs_with_exps:
            sigma_k = 1
            for i, p in enumerate(primes):
                a = exp_vec[i]
                if a > 0:
                    sigma_k *= p_sigmas[p][a]
            
            if k * sigma_k == m:
                count += 1
                ks.append(k)
                
        if count > best_count:
            best_count = count
            best_m = m
            best_ks = ks
            
    return {'M': best_m, 'k': best_ks}

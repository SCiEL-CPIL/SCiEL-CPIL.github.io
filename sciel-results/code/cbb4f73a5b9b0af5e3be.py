import math
import time

def construct():
    start_time = time.time()
    time_limit = 110  # Safety margin for 120s limit
    
    max_val = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    candidates = set()
    
    # Precompute primorials
    primorials = [1]
    for p in primes:
        primorials.append(primorials[-1] * p)
    
    # Small primes for the multiplier 'm'
    small_primes = primes[:8]  # 2, 3, 5, 7, 11, 13, 17, 19
    
    def gen_multipliers(idx, max_exp, current_m, limit_m):
        """
        Generate multipliers m using primes from small_primes[idx:]
        such that current_m * (product of future primes) <= limit_m
        and exponents are non-increasing (standard canonical form for smooth numbers)
        """
        if idx >= len(small_primes):
            candidates.add(current_m)
            return
            
        p = small_primes[idx]
        # Max exponent for p
        e_max = 0
        temp = 1
        while temp * p <= limit_m and e_max < max_exp:
            temp *= p
            e_max += 1
            
        for e in range(0, e_max + 1):
            next_m = current_m * (p ** e)
            next_limit = limit_m // (p ** e)
            gen_multipliers(idx + 1, e, next_m, next_limit)
            
    # For each k, generate P_k * m
    for k in range(1, len(primorials)):
        pk = primorials[k]
        if pk > max_val:
            break
        
        # The multiplier m can be up to max_val // pk
        limit_m = max_val // pk
        
        # We want m to be composed of small primes.
        # We allow exponents up to 10 (arbitrary, limited by limit_m)
        gen_multipliers(0, 10, 1, limit_m)
        
        # Add the base primorial itself (m=1 is handled by gen_multipliers with e=0)
        # But explicitly add P_k to be safe
        candidates.add(pk)

    # 2. Generate Highly Composite candidates (standard DFS)
    # This covers numbers with few distinct primes but high exponents.
    
    def dfs_hc(idx, max_exp, current_n, current_tau):
        if time.time() - start_time > 50:
            return
            
        if current_n > 1:
            candidates.add(current_n)
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        limit_n = max_val // current_n
        
        e_max = 0
        temp = 1
        while temp * p <= limit_n and e_max < max_exp:
            temp *= p
            e_max += 1
            
        for e in range(1, e_max + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            dfs_hc(idx + 1, e, next_n, next_tau)
            
            if time.time() - start_time > 50:
                return

    dfs_hc(0, 60, 1, 1)
    
    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47

    # Convert to list
    candidate_list = list(candidates)
    
    # We need to evaluate tau_perp for these.
    # To avoid evaluating too many, we can prioritize.
    # But the set might be large. Let's check size.
    # Primorial family:
    # k=15, limit_m ~ 1.6. m=1. (1 candidate)
    # k=14, limit_m ~ 31. m in {1..31} smooth.
    # k=10, limit_m ~ 10^9.
    # The number of smooth numbers can be large.
    # Let's limit the evaluation to a manageable number, prioritizing those with high tau.
    
    def get_tau(n):
        temp_n = n
        tau = 1
        for p in primes:
            if temp_n % p == 0:
                count = 0
                while temp_n % p == 0:
                    temp_n //= p
                    count += 1
                tau *= (count + 1)
            if temp_n == 1:
                break
        if temp_n > 1:
            tau *= 2
        return tau

    # Calculate tau for sorting
    # This might be slow for many candidates.
    # Let's just sort by n? No, higher n doesn't mean higher tau_perp.
    # Let's sort by tau descending.
    
    # Optimization: Only compute tau for candidates that look promising?
    # Or just compute tau for all if count < 10000.
    
    if len(candidate_list) > 5000:
        # Filter to keep only those with many prime factors or high value?
        # For now, let's just take a random sample or top by value?
        # Actually, let's just compute tau for all. 5000 * 15 primes is fast.
        pass

    tau_scores = []
    for n in candidate_list:
        tau_scores.append((get_tau(n), n))
        
    # Sort by tau descending
    tau_scores.sort(key=lambda x: x[0], reverse=True)
    
    def tau_perp(n):
        # Factorize n using small primes
        factors = []
        temp_n = n
        for p in primes:
            if temp_n % p == 0:
                count = 0
                while temp_n % p == 0:
                    temp_n //= p
                    count += 1
                factors.append((p, count))
            if temp_n == 1:
                break
        
        if temp_n > 1:
            factors.append((temp_n, 1))

        # Generate divisors
        divs = [1]
        for p, e in factors:
            current_divs = list(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for d in current_divs:
                    divs.append(d * factor)
                    
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    best_score = -1
    best_n = candidate_list[0]
    
    # Evaluate top candidates
    # We want to ensure we evaluate the Primorial ones.
    # Since they have high tau, they should be in the top.
    # But to be safe, let's evaluate top 2000.
    
    eval_limit = 2000
    
    for i, (tau, n) in enumerate(tau_scores):
        if i >= eval_limit:
            break
        if time.time() - start_time > time_limit - 10:
            break
            
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n

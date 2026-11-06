import math
import time

def construct():
    start_time = time.time()
    time_limit = 145 # Safety margin for 150s limit
    
    max_val = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    candidates = set()
    
    def dfs(idx, max_exp, current_n, current_tau):
        if time.time() - start_time > 50:
            return
            
        if current_n > 1:
            candidates.add((current_tau, current_n))
            
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
            dfs(idx + 1, e, next_n, next_tau)
            
            if time.time() - start_time > 50:
                return

    dfs(0, 60, 1, 1)
    
    # Add Primorials and richer variations
    pk = 1
    for i, p in enumerate(primes):
        pk *= p
        if pk > max_val:
            break
        candidates.add((2**(i+1), pk))
        
        # More variations: try different small powers on first few primes
        for a in range(1, 12):
            n = pk * (2 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))
                
        for a in range(1, 8):
            n = pk * (3 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))
                
        for a in range(1, 6):
            n = pk * (5 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))
                
        for a in range(1, 5):
            n = pk * (7 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))
        
        # Try removing the largest prime and adding extra power of a small one
        if i >= 2:
            pk_no_last = pk // primes[i]
            for a in range(2, 14):
                n = pk_no_last * (2 ** a)
                if n <= max_val and n > 1:
                    candidates.add((2**i * (a+1), n))
            for a in range(2, 10):
                n = pk_no_last * (3 ** a)
                if n <= max_val and n > 1:
                    candidates.add((2**i * (a+1), n))
            for a in range(2, 7):
                n = pk_no_last * (5 ** a)
                if n <= max_val and n > 1:
                    candidates.add((2**i * (a+1), n))
            
            # Try removing two largest primes and adding extra power
            if i >= 4:
                pk_no_last2 = pk_no_last // primes[i-1]
                for a in range(2, 16):
                    n = pk_no_last2 * (2 ** a)
                    if n <= max_val and n > 1:
                        candidates.add((2**(i-1) * (a+1), n))
                for a in range(2, 12):
                    n = pk_no_last2 * (3 ** a)
                    if n <= max_val and n > 1:
                        candidates.add((2**(i-1) * (a+1), n))

    # Generate additional candidates by shifting exponents within primorial bases
    # Focus on primorials P13-P15 as suggested by cell findings
    def generate_shifted_candidates(base_primes, max_shifts=3):
        """Generate candidates by shifting exponents within a primorial base"""
        base_n = 1
        for p in base_primes:
            base_n *= p
        
        if base_n > max_val:
            return
        
        # Start with all exponents = 1
        exponents = [1] * len(base_primes)
        
        # Generate all combinations of exponent shifts
        # Shift weight to smaller primes (increase their exponents)
        # while decreasing larger primes' exponents to stay within bounds
        
        def shift_search(idx, remaining_budget, current_exps, current_n):
            if time.time() - start_time > 60:
                return
            
            # Record this configuration
            n = 1
            for i, p in enumerate(base_primes):
                n *= p ** current_exps[i]
            if n <= max_val and n > 1:
                tau = 1
                for e in current_exps:
                    tau *= (e + 1)
                candidates.add((tau, n))
            
            if idx >= len(base_primes):
                return
            
            p = base_primes[idx]
            
            # Try increasing exponent of current prime (shifting weight to smaller primes)
            max_e = current_exps[idx] + remaining_budget + 2
            for e in range(current_exps[idx], max_e + 1):
                if current_n * (p ** (e - current_exps[idx])) > max_val:
                    break
                new_exps = current_exps[:]
                new_exps[idx] = e
                new_n = current_n * (p ** (e - current_exps[idx]))
                new_budget = remaining_budget - (e - current_exps[idx])
                shift_search(idx + 1, new_budget, new_exps, new_n)
            
            # Also try decreasing exponent (freeing up budget for smaller primes)
            if current_exps[idx] > 1:
                for e in range(1, current_exps[idx]):
                    new_exps = current_exps[:]
                    new_exps[idx] = e
                    new_n = current_n // (p ** (current_exps[idx] - e))
                    new_budget = remaining_budget + (current_exps[idx] - e)
                    shift_search(idx + 1, new_budget, new_exps, new_n)
        
        shift_search(0, 0, exponents, base_n)
    
    # Generate shifted candidates for primorials P13 through P15
    for num_primes in [13, 14, 15]:
        if num_primes <= len(primes):
            generate_shifted_candidates(primes[:num_primes])
    
    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47

    candidate_list = list(candidates)
    candidate_list.sort(key=lambda x: x[0], reverse=True)
    
    def tau_perp(n):
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
    best_n = candidate_list[0][1]
    
    evaluated = set()
    # Evaluate more candidates - tau_perp is fast for these sizes
    for i, (tau, n) in enumerate(candidate_list):
        if time.time() - start_time > time_limit - 3:
            break
        if n in evaluated:
            continue
        evaluated.add(n)
        
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    # Local search: try to improve best_n by small exponent adjustments
    if time.time() - start_time < time_limit - 10:
        def local_search(n, max_iterations=500):
            """Try to improve n by adjusting exponents of its prime factors"""
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
            
            best_local = n
            best_local_score = tau_perp(n)
            
            for _ in range(max_iterations):
                if time.time() - start_time > time_limit - 5:
                    break
                
                improved = False
                # Try increasing or decreasing each exponent by 1
                for i, (p, e) in enumerate(factors):
                    # Try increasing exponent
                    if e < 60:
                        new_factors = factors[:]
                        new_factors[i] = (p, e + 1)
                        new_n = 1
                        for pp, ee in new_factors:
                            new_n *= pp ** ee
                        if new_n <= max_val:
                            new_score = tau_perp(new_n)
                            if new_score > best_local_score:
                                best_local_score = new_score
                                best_local = new_n
                                factors = new_factors
                                improved = True
                                break
                    
                    # Try decreasing exponent (if > 1)
                    if e > 1:
                        new_factors = factors[:]
                        new_factors[i] = (p, e - 1)
                        new_n = 1
                        for pp, ee in new_factors:
                            new_n *= pp ** ee
                        if new_n > 1:
                            new_score = tau_perp(new_n)
                            if new_score > best_local_score:
                                best_local_score = new_score
                                best_local = new_n
                                factors = new_factors
                                improved = True
                                break
                
                if not improved:
                    break
            
            return best_local, best_local_score
        
        improved_n, improved_score = local_search(best_n)
        if improved_score > best_score:
            best_score = improved_score
            best_n = improved_n
    
    return best_n

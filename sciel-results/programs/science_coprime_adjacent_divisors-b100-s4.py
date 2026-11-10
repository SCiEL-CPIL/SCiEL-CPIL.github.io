import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Primes up to 2000 is plenty for n <= 10^18
    primes = sieve(2000)
    
    def get_divisors(factors):
        # factors: list of (p, e)
        divs = [1]
        for p, e in factors:
            new_divs = []
            powers = [1]
            for _ in range(e):
                powers.append(powers[-1] * p)
            for d in divs:
                for pow_val in powers:
                    new_divs.append(d * pow_val)
            divs = new_divs
        return divs

    def tau_perp(n, factors):
        # factors: list of (p, e)
        # n is the product
        if len(factors) == 0:
            return 0
        divs = get_divisors(factors)
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Initialize with a strong candidate
    # Start with 2^a * 3^b * ...
    # Heuristic: exponents decreasing
    best_n = 1
    best_score = 0
    best_factors = []
    
    # Try multiple initial configurations
    init_configs = [
        ([2,3,5,7,11,13,17,19,23,29,31,37], [4,2,1,1,1,1,1,1,1,1,1,1]),
        ([2,3,5,7,11,13,17,19,23,29,31,37], [3,3,1,1,1,1,1,1,1,1,1,1]),
        ([2,3,5,7,11,13,17,19,23,29,31,37,41,43], [3,2,1,1,1,1,1,1,1,1,1,1,1,1]),
        ([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47], [2,2,1,1,1,1,1,1,1,1,1,1,1,1,1]),
    ]
    
    for init_primes_list, init_exps_list in init_configs:
        if len(init_primes_list) != len(init_exps_list):
            continue
        init_factors = list(zip(init_primes_list, init_exps_list))
        prod = 1
        for p, e in init_factors:
            prod *= p ** e
        if 2 <= prod <= 10**18:
            s = tau_perp(prod, init_factors)
            if s > best_score:
                best_score = s
                best_n = prod
                best_factors = list(init_factors)
    
    # Fallback if no valid init
    if best_factors == []:
        for p in primes:
            if best_n * p > 10**18: break
            best_n *= p
            best_factors.append((p, 1))
        best_score = tau_perp(best_n, best_factors)

    current_factors = list(best_factors)
    current_prod = best_n
    current_score = best_score

    # Local Search with multiple move types
    improved = True
    no_improve_count = 0
    max_no_improve = 8
    
    while improved and time.time() < deadline:
        improved = False
        best_move_score = current_score
        best_new_factors = None
        best_new_prod = current_prod
        
        current_primes_set = {p for p, e in current_factors}
        
        # 1. Exponent changes (increase/decrease by 1 or 2)
        for i, (p, e) in enumerate(current_factors):
            # Decrease by 1 or 2
            for delta in [1, 2]:
                if e > delta:
                    new_e = e - delta
                    new_prod = current_prod // (p ** delta)
                    if new_prod >= 2:
                        new_factors = list(current_factors)
                        new_factors[i] = (p, new_e)
                        s = tau_perp(new_prod, new_factors)
                        if s > best_move_score:
                            best_move_score = s
                            best_new_factors = new_factors
                            best_new_prod = new_prod
                            
            # Increase by 1 or 2
            for delta in [1, 2]:
                if current_prod * (p ** delta) <= 10**18:
                    new_e = e + delta
                    new_prod = current_prod * (p ** delta)
                    new_factors = list(current_factors)
                    new_factors[i] = (p, new_e)
                    s = tau_perp(new_prod, new_factors)
                    if s > best_move_score:
                        best_move_score = s
                        best_new_factors = new_factors
                        best_new_prod = new_prod

        # 2. Swap primes (try nearby primes)
        for i, (p, e) in enumerate(current_factors):
            base = current_prod // (p ** e)
            # Try primes in a window around p
            p_idx = primes.index(p) if p in primes else 0
            for q in primes:
                if q in current_primes_set:
                    continue
                new_prod = base * (q ** e)
                if new_prod > 10**18 or new_prod < 2:
                    continue
                new_factors = list(current_factors)
                new_factors[i] = (q, e)
                s = tau_perp(new_prod, new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_new_factors = new_factors
                    best_new_prod = new_prod

        # 3. Add new prime (exponent 1)
        for q in primes:
            if q in current_primes_set:
                continue
            new_prod = current_prod * q
            if new_prod > 10**18:
                continue
            new_factors = list(current_factors)
            new_factors.append((q, 1))
            s = tau_perp(new_prod, new_factors)
            if s > best_move_score:
                best_move_score = s
                best_new_factors = new_factors
                best_new_prod = new_prod

        # 4. Remove a prime (if exponent > 1, reduce to 1; if exp==1, remove)
        for i, (p, e) in enumerate(current_factors):
            if e > 1:
                new_e = 1
                new_prod = current_prod // (p ** (e - 1))
                new_factors = list(current_factors)
                new_factors[i] = (p, new_e)
                s = tau_perp(new_prod, new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_new_factors = new_factors
                    best_new_prod = new_prod
            elif len(current_factors) > 1:
                # Remove prime entirely
                new_factors = list(current_factors)
                new_factors.pop(i)
                new_prod = current_prod // p
                if new_prod >= 2:
                    s = tau_perp(new_prod, new_factors)
                    if s > best_move_score:
                        best_move_score = s
                        best_new_factors = new_factors
                        best_new_prod = new_prod

        if best_new_factors is not None:
            current_factors = best_new_factors
            current_prod = best_new_prod
            current_score = best_move_score
            improved = True
            no_improve_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_n = current_prod
                best_factors = list(current_factors)
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try random perturbation to escape local optimum
                if time.time() < deadline - 5:
                    perturbed = False
                    for _ in range(20):
                        if time.time() > deadline - 5:
                            break
                        # Random perturbation: change a random exponent
                        if len(current_factors) == 0:
                            break
                        idx = random.randint(0, len(current_factors) - 1)
                        p, e = current_factors[idx]
                        # Try increasing or decreasing
                        if e < 5 and current_prod * p <= 10**18:
                            new_e = e + 1
                            new_prod = current_prod * p
                            new_factors = list(current_factors)
                            new_factors[idx] = (p, new_e)
                            s = tau_perp(new_prod, new_factors)
                            if s > current_score - 2:  # Allow small degradation
                                current_factors = new_factors
                                current_prod = new_prod
                                current_score = s
                                perturbed = True
                                break
                        elif e > 1:
                            new_e = e - 1
                            new_prod = current_prod // p
                            new_factors = list(current_factors)
                            new_factors[idx] = (p, new_e)
                            s = tau_perp(new_prod, new_factors)
                            if s > current_score - 2:
                                current_factors = new_factors
                                current_prod = new_prod
                                current_score = s
                                perturbed = True
                                break
                    if not perturbed:
                        break
                else:
                    break
            elif no_improve_count > 3:
                # Continue but with reduced perturbation tolerance
                continue

    return best_n

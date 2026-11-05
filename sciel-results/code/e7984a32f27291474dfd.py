import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    num_primes = len(primes)
    
    def get_divisors_from_exp(exp_list):
        divs = [1]
        for i, e in enumerate(exp_list):
            if e == 0:
                continue
            p = primes[i]
            new_divs = []
            power = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * power)
                power *= p
            divs = new_divs
        return sorted(divs)
    
    def calc_score(exp_list):
        divs = get_divisors_from_exp(exp_list)
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    def get_n_from_exp(exp_list):
        n = 1
        for i, e in enumerate(exp_list):
            if e > 0:
                n *= (primes[i] ** e)
        return n
    
    # Global best
    best_n = 2
    best_score = 0
    best_exp = [0] * num_primes
    
    def update_best(exp_list):
        nonlocal best_n, best_score, best_exp
        n = get_n_from_exp(exp_list)
        if n < 2 or n > 10**18:
            return
        score = calc_score(exp_list)
        if score > best_score:
            best_score = score
            best_n = n
            best_exp = exp_list[:]
    
    # Phase 1: DFS over monotone exponent sequences
    # e_1 >= e_2 >= ... >= e_k >= 1, then zeros
    # We explore by choosing how many primes to use and what exponents
    
    def dfs(idx, current_prod, max_exp_allowed, exp_list):
        nonlocal best_score
        if time.time() > deadline:
            return
        
        # Option: stop here (all remaining exponents are 0)
        if current_prod >= 2:
            # Compute tau for quick pruning
            tau = 1
            for i in range(idx):
                tau *= (exp_list[i] + 1)
            if tau - 1 > best_score:
                update_best(exp_list[:idx] + [0] * (num_primes - idx))
        
        # Option: continue with more primes
        if idx >= num_primes:
            return
        
        p = primes[idx]
        
        # Try exponents from min(max_exp_allowed, reasonable_max) down to 1
        # Reasonable max: p^e <= 10^18 / current_prod
        max_e = max_exp_allowed
        # Also limit by product constraint
        prod_limit = 10**18 // current_prod if current_prod > 0 else 10**18
        pe = 1
        while pe * p <= prod_limit:
            pe *= p
            max_e = min(max_e, int(math.log(pe, p)))
        
        for e in range(max_e, 0, -1):
            new_prod = current_prod * (p ** e)
            if new_prod > 10**18:
                continue
            exp_list[idx] = e
            dfs(idx + 1, new_prod, e, exp_list)
            exp_list[idx] = 0
    
    exp_list = [0] * num_primes
    dfs(0, 1, 60, exp_list)  # 2^60 > 10^18, so max exp for 2 is ~59
    
    # Phase 2: Local refinement (hill-climbing) on the best candidate and nearby
    # Try to improve by adjusting individual exponents
    
    def local_refine(exp_list, max_iterations=500):
        nonlocal best_score, best_n, best_exp
        current_exp = exp_list[:]
        current_n = get_n_from_exp(current_exp)
        current_score = calc_score(current_exp)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try each prime index
            for idx in range(num_primes):
                # Try +1
                if current_exp[idx] < 100:  # cap to avoid huge numbers
                    new_exp = current_exp[:]
                    new_exp[idx] += 1
                    new_n = get_n_from_exp(new_exp)
                    if new_n <= 10**18 and new_n >= 2:
                        new_score = calc_score(new_exp)
                        if new_score > current_score:
                            current_exp = new_exp
                            current_n = new_n
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n
                                best_exp = current_exp[:]
                
                # Try -1
                if current_exp[idx] > 0:
                    new_exp = current_exp[:]
                    new_exp[idx] -= 1
                    new_n = get_n_from_exp(new_exp)
                    if new_n <= 10**18 and new_n >= 2:
                        new_score = calc_score(new_exp)
                        if new_score > current_score:
                            current_exp = new_exp
                            current_n = new_n
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_n = current_n
                                best_exp = current_exp[:]
        
        return current_exp
    
    # Refine the best found so far
    if best_score > 0:
        refined = local_refine(best_exp, max_iterations=200)
    
    # Phase 3: Additional search - try different starting points and refine
    # Generate several diverse candidates and refine each
    
    # Candidate 1: Primorial with varying lengths
    for k in range(5, 16):
        if time.time() > deadline:
            break
        exp_test = [0] * num_primes
        prod = 1
        valid = True
        for i in range(k):
            exp_test[i] = 1
            prod *= primes[i]
        if prod <= 10**18:
            update_best(exp_test)
    
    # Candidate 2: Heavy on small primes
    for e2 in range(2, 15):
        if time.time() > deadline:
            break
        exp_test = [0] * num_primes
        exp_test[0] = e2
        prod = 2 ** e2
        for i in range(1, num_primes):
            if prod * primes[i] > 10**18:
                break
            exp_test[i] = 1
            prod *= primes[i]
        if prod >= 2:
            update_best(exp_test)
    
    # Candidate 3: Mix of exponents
    for e2 in range(1, 10):
        for e3 in range(1, min(8, e2 + 1)):
            if time.time() > deadline:
                break
            exp_test = [0] * num_primes
            exp_test[0] = e2
            exp_test[1] = e3
            prod = (2 ** e2) * (3 ** e3)
            for i in range(2, num_primes):
                if prod * primes[i] > 10**18:
                    break
                exp_test[i] = 1
                prod *= primes[i]
            if prod >= 2:
                update_best(exp_test)
    
    # Final refinement on the best
    if best_score > 0 and time.time() < deadline:
        refined = local_refine(best_exp, max_iterations=500)
    
    return best_n

if __name__ == "__main__":
    n = construct()
    print(f"Constructed: {n}")

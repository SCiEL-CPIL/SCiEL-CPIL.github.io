import random
from fractions import Fraction
import time

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True

def construct():
    primes = [p for p in range(2, 1001) if is_prime(p)]
    # Use primes >= 7 to maximize coverage potential while staying within budget
    active_primes = [p for p in primes if p >= 7]
    
    # Verify budget
    sum_recip = sum(Fraction(1, p) for p in active_primes)
    if sum_recip > Fraction(1, 1):
        # Fallback to >= 13 if somehow over (shouldn't happen)
        active_primes = [p for p in primes if p >= 13]

    def get_covered_prefix(res_map, P_list, max_L=1000):
        covered = [False] * (max_L + 1)
        for p in P_list:
            r = res_map[p]
            start = r if r > 0 else p
            for n in range(start, max_L + 1, p):
                covered[n] = True
        L = 0
        for n in range(1, max_L + 1):
            if covered[n]:
                L = n
            else:
                break
        return L

    # Initialize
    current_res = {p: random.randint(0, p-1) for p in active_primes}
    current_L = get_covered_prefix(current_res, active_primes, 1000)
    
    best_res = current_res.copy()
    best_L = current_L
    
    # Simulated Annealing
    start_temp = 50.0
    end_temp = 0.1
    cooling_rate = 0.995
    temp = start_temp
    max_iter = 3000
    
    start_time = time.time()
    timeout = 100.0
    
    while temp > end_temp and (time.time() - start_time) < timeout:
        p = random.choice(active_primes)
        old_r = current_res[p]
        new_r = random.randint(0, p-1)
        if new_r == old_r:
            temp *= cooling_rate
            continue
            
        current_res[p] = new_r
        new_L = get_covered_prefix(current_res, active_primes, 1000)
        
        delta = new_L - current_L
        
        if delta > 0:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_res = current_res.copy()
        else:
            # Accept worse with probability exp(delta/temp)
            # delta is negative
            prob = 2.718281828 ** (delta / temp)
            if random.random() < prob:
                current_L = new_L
            else:
                current_res[p] = old_r
        
        temp *= cooling_rate
        
    # Local Search Refinement on best_res
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        # Find first hole in best_res up to 1000
        covered = [False] * (1001)
        for p in active_primes:
            r = best_res[p]
            start = r if r > 0 else p
            for n in range(start, 1001, p):
                covered[n] = True
        
        n_uncovered = -1
        for n in range(1, 1001):
            if not covered[n]:
                n_uncovered = n
                break
                
        if n_uncovered == -1:
            break
            
        # Try to cover n_uncovered
        for p in active_primes:
            new_r = n_uncovered % p
            if best_res[p] == new_r:
                continue
            old_r = best_res[p]
            best_res[p] = new_r
            
            # Check 1..n_uncovered-1
            temp_covered = [False] * (n_uncovered + 1)
            for pp in active_primes:
                rr = best_res[pp]
                start = rr if rr > 0 else pp
                for nn in range(start, n_uncovered + 1, pp):
                    temp_covered[nn] = True
            
            ok = all(temp_covered[nn] for nn in range(1, n_uncovered))
            if ok:
                improved = True
                break
            else:
                best_res[p] = old_r
                
    return [[p, best_res[p]] for p in active_primes]

import time
import random
from fractions import Fraction

def get_primes(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = 0
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    primes = get_primes(1000)
    
    def sum_recip_frac(prime_list):
        return sum(Fraction(1, p) for p in prime_list)

    def compute_coverage(res_list, p_set, L_limit):
        """Compute which numbers 1..L_limit are covered. Returns (covered_array, first_uncovered)."""
        covered = bytearray(L_limit + 1)  # 0 = uncovered, 1 = covered
        for i, p in enumerate(p_set):
            r = res_list[i]
            # First x >= 1 with x % p == r
            if r == 0:
                first_x = p
            else:
                first_x = r
            if first_x > L_limit:
                continue
            for x in range(first_x, L_limit + 1, p):
                covered[x] = 1
        # Find first uncovered
        first_uncovered = 0
        for x in range(1, L_limit + 1):
            if not covered[x]:
                first_uncovered = x
                break
        return covered, first_uncovered

    def eval_L(res_list, p_set, L_limit=100000):
        """Find the largest L such that 1..L are all covered."""
        _, first_uncovered = compute_coverage(res_list, p_set, L_limit)
        return first_uncovered - 1 if first_uncovered > 0 else L_limit

    # Phase 1: Build a good initial covering using greedy + local search
    # Start with all primes >= 13 (sum of reciprocals < 1)
    initial_primes = [p for p in primes if p >= 13]
    
    # Initialize residues randomly
    res_list = [random.randint(0, p - 1) for p in initial_primes]
    
    # Evaluate initial L
    current_L = eval_L(res_list, initial_primes, 5000)
    
    # Local search to improve L
    best_L = current_L
    best_res = res_list[:]
    best_pset = initial_primes[:]
    
    def local_search(p_set, res_list, time_budget=30.0):
        """Hill climbing + simulated annealing on residues to maximize L."""
        t0 = time.time()
        current_L = eval_L(res_list, p_set, 5000)
        best_L_local = current_L
        best_res_local = res_list[:]
        
        k = len(p_set)
        temp = 10.0
        cooling = 0.995
        
        while time.time() - t0 < time_budget and temp > 0.01:
            # Pick a random prime index
            i = random.randint(0, k - 1)
            p = p_set[i]
            old_r = res_list[i]
            
            # Try a random new residue
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            res_list[i] = new_r
            new_L = eval_L(res_list, p_set, 5000)
            
            delta = new_L - current_L
            if delta > 0:
                current_L = new_L
                if new_L > best_L_local:
                    best_L_local = new_L
                    best_res_local = res_list[:]
            else:
                # Accept with probability exp(delta / temp)
                if random.random() < (2.718281828 ** (delta / temp)):
                    current_L = new_L
                else:
                    res_list[i] = old_r
            
            temp *= cooling
        
        return best_L_local, best_res_local
    
    # Run local search with time budget
    remaining = deadline - time.time()
    ls_budget = min(40.0, remaining * 0.4)
    best_L, best_res = local_search(best_pset, best_res[:], ls_budget)
    
    # Phase 2: Try adding smaller primes to extend coverage
    small_primes = [p for p in primes if p < 13]
    small_primes.sort()  # 2, 3, 5, 7, 11
    
    for p_add in small_primes:
        if time.time() > deadline - 20:
            break
        
        new_pset = best_pset + [p_add]
        if sum_recip_frac(new_pset) > 1:
            continue
        
        # Add this prime with random residue
        new_res = best_res[:] + [random.randint(0, p_add - 1)]
        
        # Local search on the expanded set
        ls_budget2 = min(25.0, deadline - time.time() - 15)
        if ls_budget2 <= 0:
            break
        
        new_L, new_res_best = local_search(new_pset, new_res, ls_budget2)
        
        if new_L > best_L:
            best_L = new_L
            best_res = new_res_best
            best_pset = new_pset[:]
    
    # Phase 3: Final polish - try swapping out large primes for combinations of small ones
    if time.time() < deadline - 10:
        # Try removing the largest prime and see if we can do better
        if len(best_pset) > 1:
            # Sort by prime value
            sorted_idx = sorted(range(len(best_pset)), key=lambda i: best_pset[i])
            largest_idx = sorted_idx[-1]
            
            p_remove = best_pset[largest_idx]
            trial_pset = best_pset[:largest_idx] + best_pset[largest_idx+1:]
            trial_res = best_res[:largest_idx] + best_res[largest_idx+1:]
            
            # Try adding back some small primes
            for p_add in [2, 3, 5, 7, 11]:
                if p_add in trial_pset:
                    continue
                new_pset = trial_pset + [p_add]
                if sum_recip_frac(new_pset) > 1:
                    continue
                new_res = trial_res[:] + [random.randint(0, p_add - 1)]
                
                trial_L = eval_L(new_res, new_pset, 5000)
                if trial_L > best_L:
                    best_L = trial_L
                    best_res = new_res
                    best_pset = new_pset[:]
                    break
    
    # Final evaluation with higher L limit
    final_L = eval_L(best_res, best_pset, 100000)
    if final_L > best_L:
        best_L = final_L
    
    return [[p, r] for p, r in zip(best_pset, best_res)]

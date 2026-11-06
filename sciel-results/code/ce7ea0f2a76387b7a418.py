import time
import random
from fractions import Fraction

# Precompute primes up to 1000
def get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = get_primes(1000)

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s out of 120s limit
    
    # 1. Select a candidate set of primes
    # Greedy: smallest primes first, as long as sum(1/p) <= 1
    # We use exact rationals for the sum check to be safe, but floats are fast enough for selection.
    # Let's use Fraction for exactness in the final check, but float for search speed.
    
    selected_primes = []
    current_sum = 0.0
    for p in PRIMES:
        if current_sum + 1.0/p <= 1.0:
            selected_primes.append(p)
            current_sum += 1.0/p
        else:
            # Try skipping this prime and seeing if a larger one fits?
            # For now, just break or continue checking.
            # Since 1/p decreases, if 1/p doesn't fit, larger primes might fit.
            # But we want small primes for coverage.
            # Let's just collect all primes that fit individually into the remaining budget?
            # No, the budget is global.
            # A better strategy: Start with small primes. If we can't add p, try next.
            # But maybe we should swap?
            # Let's stick to a simple greedy for now: take smallest primes until sum > 1.
            # Actually, the greedy "take if fits" might miss better combinations.
            # Example: 2, 3, 5 sum = 1.033. Can't take 5.
            # 2, 3, 7 sum = 0.975.
            # 2, 5, 7 sum = 0.842.
            # 3, 5, 7 sum = 0.675.
            # 2, 3, 7, 11? 0.975 + 0.09 = 1.065. No.
            # So {2,3,7} is a valid set.
            # {2,5,7,11}? 0.842 + 0.09 = 0.932. Yes.
            # {2,5,7,11,13}? 0.932 + 0.077 = 1.009. No.
            # {2,5,7,11,17}? 0.932 + 0.058 = 0.99. Yes.
            
            # Let's generate a few candidate sets.
            pass

    # Let's try a specific set known to be good or just the greedy one.
    # Greedy smallest:
    # 2 (0.5), 3 (0.833), 5 (1.033 - fail).
    # So greedy gives [2, 3]. Sum = 0.833.
    # Next prime 7 (0.142). 0.833 + 0.142 = 0.975 <= 1. Add 7.
    # Next prime 11 (0.09). 0.975 + 0.09 = 1.065 > 1. Skip.
    # Next prime 13 (0.077). 0.975 + 0.077 = 1.052 > 1. Skip.
    # Next prime 17 (0.058). 0.975 + 0.058 = 1.033 > 1. Skip.
    # Next prime 19 (0.052). 0.975 + 0.052 = 1.027 > 1. Skip.
    # Next prime 23 (0.043). 0.975 + 0.043 = 1.018 > 1. Skip.
    # Next prime 29 (0.034). 0.975 + 0.034 = 1.009 > 1. Skip.
    # Next prime 31 (0.032). 0.975 + 0.032 = 1.007 > 1. Skip.
    # Next prime 37 (0.027). 0.975 + 0.027 = 1.002 > 1. Skip.
    # Next prime 41 (0.024). 0.975 + 0.024 = 0.999 <= 1. Add 41.
    # Next prime 43 (0.023). 0.999 + 0.023 > 1. Skip.
    # So set: [2, 3, 7, 41].
    
    # This set is small. Maybe we can do better by not taking 3?
    # If we don't take 3:
    # 2 (0.5).
    # 5 (0.7).
    # 7 (0.842).
    # 11 (0.932).
    # 13 (1.009) - fail.
    # 17 (0.99) - ok.
    # 19 (1.042) - fail.
    # 23 (1.033) - fail.
    # 29 (1.024) - fail.
    # 31 (1.022) - fail.
    # 37 (1.017) - fail.
    # 41 (1.014) - fail.
    # 43 (1.012) - fail.
    # 47 (1.011) - fail.
    # 53 (1.010) - fail.
    # 59 (1.008) - fail.
    # 61 (1.007) - fail.
    # 67 (1.006) - fail.
    # 71 (1.005) - fail.
    # 73 (1.005) - fail.
    # 79 (1.004) - fail.
    # 83 (1.004) - fail.
    # 89 (1.003) - fail.
    # 97 (1.003) - fail.
    # 101 (1.002) - fail.
    # 103 (1.002) - fail.
    # 107 (1.002) - fail.
    # 109 (1.002) - fail.
    # 113 (1.001) - fail.
    # 127 (1.001) - fail.
    # 131 (1.001) - fail.
    # 137 (1.001) - fail.
    # 139 (1.001) - fail.
    # 149 (1.000) - ok? 0.99 + 0.0067 = 0.9967.
    # So set: [2, 5, 7, 11, 17, 149].
    
    # It seems we can have more primes if we skip 3.
    # Let's try to optimize a set of primes.
    # We will define a function to check if a set is valid.
    
    def is_valid_set(prime_list):
        s = Fraction(0, 1)
        for p in prime_list:
            s += Fraction(1, p)
        return s <= 1

    # Candidate sets
    # Set 1: [2, 3, 7, 41]
    # Set 2: [2, 5, 7, 11, 17, 149]
    # Set 3: [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
    # Sum of reciprocals of all primes > 3 is approx 2.2 - (0.5+0.333) = 1.36.
    # So we can take a lot of primes if we skip 2 and 3.
    # But small primes are more effective for covering a prefix.
    
    # Let's use a local search that can add/remove primes.
    # State: list of (p, r).
    # We'll start with a few candidate sets and optimize residues.
    
    candidate_sets = [
        [2, 3, 7, 41],
        [2, 5, 7, 11, 17, 149],
        [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
    ]
    
    # Filter valid sets
    valid_sets = []
    for s in candidate_sets:
        if is_valid_set(s):
            valid_sets.append(s)
    
    if not valid_sets:
        # Fallback to seed-like
        valid_sets.append([13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997])
        # Check if valid
        if not is_valid_set(valid_sets[0]):
             return []

    best_L = 0
    best_solution = []
    
    # Local search function
    def optimize(prime_list, time_limit):
        # Initialize residues
        residues = [random.randint(0, p-1) for p in prime_list]
        
        # Coverage array
        max_check = 2000
        covered = [False] * (max_check + 1)
        
        def update_coverage():
            nonlocal covered
            covered = [False] * (max_check + 1)
            for i, (p, r) in enumerate(zip(prime_list, residues)):
                for n in range(r % p, max_check + 1, p):
                    if n > 0:
                        covered[n] = True
            # Find L
            L = 0
            for i in range(1, max_check + 1):
                if covered[i]:
                    L = i
                else:
                    break
            return L

        L = update_coverage()
        best_local_L = L
        best_local_residues = residues[:]
        
        start = time.time()
        while time.time() - start < time_limit:
            # Pick a random prime to change
            idx = random.randint(0, len(prime_list) - 1)
            p = prime_list[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p-1)
            
            if new_r == old_r:
                continue
                
            # Update coverage efficiently?
            # For now, full update is O(N * K * max_check/p).
            # With K~100, max_check=2000, this is fast.
            residues[idx] = new_r
            new_L = update_coverage()
            
            if new_L > best_local_L:
                best_local_L = new_L
                best_local_residues = residues[:]
            else:
                # Revert
                residues[idx] = old_r
                
        return best_local_L, best_local_residues

    # Run optimization on each valid set
    for ps in valid_sets:
        if time.time() > deadline:
            break
        L, res = optimize(ps, 10)
        if L > best_L:
            best_L = L
            best_solution = [[p, r] for p, r in zip(ps, res)]

    # Final check and return
    if not best_solution:
        # Fallback
        return [[13, 1], [17, 2], [19, 3]]
        
    return best_solution

import time
from itertools import combinations
from math import isqrt
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    max_val = 10000
    max_sum = 19999
    
    # Sieve for primes up to max_sum
    sieve = [True] * (max_sum + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(max_sum) + 1):
        if sieve[i]:
            for j in range(i*i, max_sum + 1, i):
                sieve[j] = False
    all_primes = [i for i in range(2, max_sum + 1) if sieve[i]]
    
    def factorize(n, primes):
        factors = set()
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                factors.add(p)
                while n % p == 0:
                    n //= p
        if n > 1:
            factors.add(n)
        return factors
    
    def evaluate_count(nums):
        all_factors = set()
        for i in range(12):
            for j in range(i+1, 12):
                s = nums[i] + nums[j]
                all_factors |= factorize(s, all_primes)
        return len(all_factors)
    
    # Try to find a set where all pairwise sums are P-smooth for a small P.
    # Use backtracking with a restricted candidate pool.
    
    # For a given P, generate candidate numbers that have good compatibility.
    
    def find_set_for_prime_set(P, max_candidates=200):
        """Find 12 numbers such that all pairwise sums are P-smooth."""
        ps = sorted(P)
        
        def is_smooth(n):
            for p in ps:
                while n % p == 0:
                    n //= p
            return n == 1
        
        # Generate candidate numbers: those that can form smooth sums with many others
        # Start with all numbers 1..10000, but that's too many.
        # Restrict to numbers that are themselves "nice" (smooth or near-smooth)
        
        # Generate P-smooth numbers up to 10000
        smooth_nums = set()
        def gen_smooth(n):
            if n > 10000:
                return
            smooth_nums.add(n)
            for p in ps:
                gen_smooth(n * p)
        gen_smooth(1)
        
        # Candidates: P-smooth numbers plus some structured numbers
        candidates = list(smooth_nums)
        # Also add multiples of small numbers
        for base in ps:
            for k in range(1, 10000 // base + 1):
                candidates.append(k * base)
        
        candidates = list(set(candidates))
        candidates.sort()
        
        # Limit candidates for tractability
        if len(candidates) > max_candidates * 3:
            # Keep a diverse subset
            random.seed(42)
            random.shuffle(candidates)
            candidates = candidates[:max_candidates * 3]
            candidates.sort()
        
        # Backtracking
        selected = []
        
        def backtrack(start_idx):
            if len(selected) == 12:
                return True
            for i in range(start_idx, len(candidates)):
                x = candidates[i]
                # Check compatibility with all selected
                ok = True
                for y in selected:
                    if not is_smooth(x + y):
                        ok = False
                        break
                if ok:
                    selected.append(x)
                    if backtrack(i + 1):
                        return True
                    selected.pop()
            return False
        
        if backtrack(0):
            return list(selected)
        return None
    
    # Try prime sets of increasing size
    small_primes = all_primes[:30]
    
    best_set = None
    best_score = float('inf')
    
    # Try size 2, 3, 4 prime sets
    for n_p in range(2, 6):
        if time.time() > deadline - 10:
            break
        
        for combo in combinations(small_primes, n_p):
            if time.time() > deadline - 5:
                break
            
            P = set(combo)
            result = find_set_for_prime_set(P, max_candidates=100)
            if result is not None:
                score = evaluate_count(result)
                if score < best_score:
                    best_score = score
                    best_set = result
                if score <= n_p:  # Perfect: only primes in P
                    return result
    
    # If no perfect solution found, do local search
    if best_set is None:
        # Start with a structured initial solution
        best_set = list(range(2, 26, 2))  # 2,4,...,24
        best_score = evaluate_count(best_set)
    
    # Local search to improve
    random.seed(42)
    current = list(best_set)
    curr_score = best_score
    
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        if curr_score <= 3:
            break
        
        for step in range(3000):
            if time.time() > deadline - 3:
                break
            
            pos = random.randint(0, 11)
            old = current[pos]
            
            # Generate candidate replacement
            r = random.random()
            if r < 0.4:
                base = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15])
                new_val = random.randint(1, 10000 // base) * base
            elif r < 0.7:
                new_val = random.randint(1, 200)
            else:
                new_val = random.randint(1, 10000)
            
            if new_val == old or new_val in current:
                continue
            
            current[pos] = new_val
            new_score = evaluate_count(current)
            
            if new_score < curr_score:
                curr_score = new_score
                if curr_score < best_score:
                    best_score = curr_score
                    best_set = list(current)
            else:
                current[pos] = old
    
    return best_set

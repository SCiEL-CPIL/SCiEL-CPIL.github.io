import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s kill
    
    # Try small prime sets in order of size
    # For each prime set, try to find 12 numbers whose pairwise sums are all P-smooth
    
    def is_smooth(n, prime_set):
        """Check if n's prime factors are all in prime_set"""
        if n <= 1:
            return True
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1
    
    def generate_smooth_numbers(max_val, prime_set):
        """Generate all P-smooth numbers up to max_val"""
        smooth = set()
        # Use a priority queue approach or recursive generation
        # For small prime sets, we can generate all products
        from math import log
        # Generate all numbers of form p1^a1 * p2^a2 * ... <= max_val
        def gen(idx, current):
            if current > max_val:
                return
            if idx == len(prime_set):
                if current >= 2:  # sums are at least 1+2=3, but we want >= 3
                    smooth.add(current)
                return
            p = prime_set[idx]
            gen(idx + 1, current)
            while current * p <= max_val:
                current *= p
                gen(idx + 1, current)
        gen(0, 1)
        return sorted(smooth)
    
    def find_numbers_for_prime_set(prime_set, max_val=20000, n=12):
        """Find n distinct positive integers <= 10000 whose pairwise sums are all P-smooth"""
        smooth_sums = set(generate_smooth_numbers(max_val, prime_set))
        # We need 12 distinct numbers in [1, 10000] such that for all i != j, a[i]+a[j] is in smooth_sums
        
        # Backtracking approach
        chosen = []
        
        def backtrack(start, depth):
            if depth == n:
                return True
            for i in range(start, 10001):
                candidate = i
                # Check if adding candidate works with all previously chosen
                ok = True
                for prev in chosen:
                    if (prev + candidate) not in smooth_sums:
                        ok = False
                        break
                if ok:
                    chosen.append(candidate)
                    if backtrack(candidate + 1, depth + 1):
                        return True
                    chosen.pop()
            return False
        
        # This might be too slow for large search space. Use heuristic.
        # Try to find numbers greedily with backtracking
        # Better: use a different approach - pick numbers that are "compatible"
        
        # Alternative: random search with local improvement
        best_found = None
        
        # Try multiple random starts
        for attempt in range(200):
            if time.time() - start_time > DEADLINE:
                break
            # Random initial set
            random_set = set(random.sample(range(1, 10001), n))
            
            # Try to improve: replace elements to reduce number of "bad" pairs
            # Actually, we want ALL pairs to be smooth, so let's use a different strategy
            
            # Greedy construction: build set one by one
            chosen = []
            # Start with a random small number
            first = random.randint(1, 100)
            chosen.append(first)
            
            for _ in range(n - 1):
                # Find a number that is compatible with all chosen
                candidates = []
                for c in range(chosen[-1] + 1, 10001):
                    ok = True
                    for prev in chosen:
                        if (prev + c) not in smooth_sums:
                            ok = False
                            break
                    if ok:
                        candidates.append(c)
                    if len(candidates) > 50:
                        break
                
                if not candidates:
                    break
                
                # Pick the smallest (or random)
                chosen.append(candidates[0])
            
            if len(chosen) == n:
                # Verify all pairs
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        if (chosen[i] + chosen[j]) not in smooth_sums:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_found = chosen
                    break
        
        if best_found:
            return best_found
        return None
    
    # Try prime sets of increasing size
    prime_sets_to_try = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
    ]
    
    for ps in prime_sets_to_try:
        if time.time() - start_time > DEADLINE:
            break
        result = find_numbers_for_prime_set(ps)
        if result is not None:
            return result
    
    # Fallback: if we couldn't find a perfect solution, use a heuristic
    # to minimize the number of distinct primes
    # This shouldn't happen, but as a safety net
    if time.time() - start_time > DEADLINE - 5:
        return list(range(1, 13))
    
    # Try a more general local search: start with random numbers and try to reduce prime count
    # by replacing elements
    def count_distinct_primes(arr):
        primes = set()
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                s = arr[i] + arr[j]
                # Factor s
                temp = s
                d = 2
                while d * d <= temp:
                    while temp % d == 0:
                        primes.add(d)
                        temp //= d
                    d += 1
                if temp > 1:
                    primes.add(temp)
        return len(primes)
    
    # Local search
    best_arr = list(range(1, 13))
    best_score = count_distinct_primes(best_arr)
    
    if time.time() - start_time < DEADLINE:
        # Try to improve with local search
        arr = list(range(1, 13))
        for iteration in range(100000):
            if time.time() - start_time > DEADLINE - 2:
                break
            # Pick a random element to replace
            idx = random.randint(0, 11)
            old_val = arr[idx]
            new_val = random.randint(1, 10000)
            if new_val in arr and new_val != old_val:
                continue
            arr[idx] = new_val
            score = count_distinct_primes(arr)
            if score < best_score:
                best_score = score
                best_arr = arr[:]
                if best_score <= 3:
                    break
            else:
                arr[idx] = old_val
    
    return best_arr

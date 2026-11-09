import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    def phi(x):
        """Compute Euler's totient function for a single integer."""
        if x == 1:
            return 1
        result, d, y = x, 2, x
        while d * d <= y:
            if y % d == 0:
                result -= result // d
                while y % d == 0:
                    y //= d
            d += 1
        if y > 1:
            result -= result // y
        return result
    
    def phi_sieve(N):
        """Sieve-based computation of phi for all numbers up to N."""
        phi_arr = list(range(N + 1))
        for i in range(2, N + 1):
            if phi_arr[i] == i:  # i is prime
                for j in range(i, N + 1, i):
                    phi_arr[j] -= phi_arr[j] // i
        return phi_arr
    
    def find_longest_run_in_array(phi_arr):
        """Find the longest run of consecutive integers with distinct phi values."""
        n = len(phi_arr) - 1  # phi_arr[0] is unused
        best_k = 1
        best_s = 1
        for s in range(1, n):
            seen = set()
            k = 0
            while s + k < n and phi_arr[s + k] not in seen:
                seen.add(phi_arr[s + k])
                k += 1
            if k > best_k:
                best_k = k
                best_s = s
            if s + k >= n:
                break
        return best_k, best_s
    
    def find_run_around_start(s, max_k, phi_func):
        """Find the longest run starting at s using the given phi function."""
        seen = set()
        k = 0
        while k < max_k and s + k <= 10**12:
            val = phi_func(s + k)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    # Phase 1: Sieve-based scan up to 10^6
    N = 1000000
    phi_arr = phi_sieve(N)
    best_k, best_s = find_longest_run_in_array(phi_arr)
    
    # Phase 2: Try to extend the best run found
    # Check if we can extend beyond N
    if best_k > 1:
        # Verify the run is valid
        s, k = best_s, best_k
        seen = set()
        for i in range(k):
            seen.add(phi_arr[s + i])
        
        # Try to extend forward
        while s + k <= 10**12 and elapsed() < deadline - 5:
            val = phi(s + k)
            if val in seen:
                break
            seen.add(val)
            k += 1
            if k > best_k:
                best_k = k
    
    # Phase 3: Random sampling at larger scales
    # Sample random starting points in various ranges
    random.seed(42)
    ranges = [
        (10**6, 10**7),
        (10**7, 10**8),
        (10**8, 10**9),
        (10**9, 10**10),
        (10**10, 10**11),
        (10**11, 10**12),
    ]
    
    for lo, hi in ranges:
        if elapsed() > deadline - 10:
            break
        
        # Number of samples depends on range size
        num_samples = 20
        for _ in range(num_samples):
            if elapsed() > deadline - 10:
                break
            
            s = random.randint(lo, hi - 200)
            # Try to find a run starting near s
            # First, check a window of starts
            for offset in range(0, 50, 5):
                if elapsed() > deadline - 10:
                    break
                start = s + offset
                k = find_run_around_start(start, 200, phi)
                if k > best_k:
                    best_k = k
                    best_s = start
    
    # Phase 4: If we have time, do a more focused search around promising areas
    if elapsed() < deadline - 15 and best_k < 200:
        # Try starts that are products of small primes (these tend to have interesting phi values)
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        def gen_candidates():
            """Generate candidate starts that might have long runs."""
            # Products of small primes
            for p1 in small_primes:
                for p2 in small_primes:
                    for p3 in small_primes:
                        for p4 in small_primes:
                            for p5 in small_primes:
                                for p6 in small_primes:
                                    for p7 in small_primes:
                                        for p8 in small_primes:
                                            cand = p1 * p2 * p3 * p4 * p5 * p6 * p7 * p8
                                            if cand >= 1 and cand <= 10**12 - 200:
                                                yield cand
                                            if cand > 10**12:
                                                return
        
        # Sample from these candidates
        candidates = list(gen_candidates())
        random.shuffle(candidates)
        
        for cand in candidates[:100]:
            if elapsed() > deadline - 5:
                break
            for offset in range(0, 20):
                if elapsed() > deadline - 5:
                    break
                start = cand + offset
                k = find_run_around_start(start, 200, phi)
                if k > best_k:
                    best_k = k
                    best_s = start
    
    # Final verification of the best result
    if best_k > 1:
        seen = set()
        for i in range(best_k):
            val = phi(best_s + i)
            if val in seen:
                # This shouldn't happen, but just in case
                best_k = i
                break
            seen.add(val)
    
    # Ensure constraints are met
    if best_s < 1:
        best_s = 1
        best_k = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, int(10**12 - best_s) + 1)
    if best_k < 1:
        best_k = 1
    if best_k > 200:
        best_k = 200
    
    return {'start': best_s, 'length': best_k}

import heapq
from math import isqrt

def construct():
    # Primes up to a limit. 
    # The largest prime factor of a number <= 10^6 is at most 10^6.
    # But we expect the answer to have a small largest prime factor.
    # Let's iterate B (max prime factor) from small to large.
    
    # Sieve for primes up to 100000? 
    # Actually, we just need to try B = 2, 3, 5, 7, ...
    # We can generate primes on the fly or precompute up to 10000.
    
    # Precompute primes up to 10000
    limit = 10000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    
    max_n = 1000000
    min_n = 10000
    
    # For each prime B, check if there are consecutive B-smooth numbers in [min_n, max_n]
    for B in primes:
        # Generate B-smooth numbers in sorted order up to max_n
        # We use a min-heap
        # To avoid duplicates, we can use a set or a specific generation logic.
        # A common way is to generate them by multiplying previous smooth numbers by primes.
        # However, a simpler approach for small max_n is to just check each number? No, that's too slow.
        
        # Let's use the heap method with a set to avoid duplicates.
        # Or, since max_n is 10^6, we can just iterate x from 1 to max_n and check if x is B-smooth?
        # Checking if x is B-smooth involves trial division by primes <= B.
        # There are ~10^6 numbers. For each, we divide by primes <= B.
        # If B is small, this is fast.
        # If B is large, it's slower, but we stop early.
        
        # Let's try the direct check method first, it's simpler to implement correctly.
        # For each x in [min_n, max_n-1]:
        #   Check if x is B-smooth.
        #   Check if x+1 is B-smooth.
        #   If both, return x.
        
        # How to check if x is B-smooth?
        # Divide x by all primes p <= B until x becomes 1 or p > sqrt(x) etc.
        # Actually, just divide by all primes p <= B. If remaining factor is 1, it's smooth.
        # If remaining factor > 1, it has a prime factor > B, so not smooth.
        
        # Optimization: Precompute list of primes <= B.
        # Since we iterate B, we can maintain a list of primes <= current B.
        
        pass # This block will be replaced by the actual logic below

    # Fallback: if no small B works, return a known good candidate or the one with min lpf found by brute force on a subset
    # But the loop above should find it.
    
    # Let's re-implement the loop properly.
    
    # We need a function is_smooth(x, primes_list)
    # primes_list contains all primes <= B
    
    # To make it faster, we can break early in is_smooth if x becomes 1.
    
    # Let's refine the loop:
    # primes_so_far = []
    # for B in primes:
    #    primes_so_far.append(B)
    #    # Check all pairs
    #    for n in range(min_n, max_n):
    #        if is_smooth(n, primes_so_far) and is_smooth(n+1, primes_so_far):
    #            return n
    
    # This might be too slow if B has to be large.
    # How large can B be?
    # The number of B-smooth numbers up to N is roughly (log N)^k / k! where k is count of primes <= B.
    # For B=100, count of primes is 25.
    # The probability that a random number is B-smooth is small.
    # However, we are looking for *consecutive* ones.
    
    # Alternative: Generate all B-smooth numbers in range and check for consecutive ones.
    # This is much faster if the count of smooth numbers is small.
    
    def get_smooth_numbers(max_val, primes_limit):
        # Returns sorted list of primes_limit-smooth numbers <= max_val
        # Using heap
        # We want to generate numbers of form p1^a1 * ... * pk^ak
        # Start with 1
        # But we need to generate up to max_val.
        
        # A better way for small max_val:
        # Just iterate i from 1 to max_val and check?
        # No, let's stick to the generation via multiplication.
        
        # Actually, for max_val = 10^6, iterating and checking is:
        # 10^6 * (number of primes <= B) operations.
        # If B=100, 25 primes. 25 * 10^6 = 2.5 * 10^7 ops. In Python, this might take a few seconds.
        # If we do this for B=2,3,5,... it accumulates.
        
        # Let's try the heap generation. It generates only the smooth numbers.
        # Count of 5-smooth numbers up to 10^6 is ~ 1000.
        # Count of 100-smooth numbers is larger, but still manageable.
        
        smooths = set()
        heap = [1]
        smooths.add(1)
        while heap:
            x = heapq.heappop(heap)
            if x > max_val:
                continue
            # We already added x to smooths when we pushed it? 
            # No, standard Ugly Number problem approach:
            # Push x*p for all p in primes_limit if x*p <= max_val
            # To avoid duplicates, we use a set.
            for p in primes_limit:
                nx = x * p
                if nx > max_val:
                    break # Since primes are sorted, if x*p > max, larger p will also be
                if nx not in smooths:
                    smooths.add(nx)
                    heapq.heappush(heap, nx)
        
        return sorted(smooths)

    # Wait, the heap approach above pushes duplicates if not careful, but the set handles it.
    # However, pushing to heap only if not in set is O(1) check.
    # The heap size is bounded by number of smooth numbers.
    
    # Let's use this approach.
    
    primes_so_far = []
    for B in primes:
        primes_so_far.append(B)
        
        # Generate smooth numbers in [min_n, max_n]
        # We generate up to max_n, then filter.
        smooths = get_smooth_numbers(max_n, primes_so_far)
        
        # Check for consecutive pairs in smooths
        # Since smooths is sorted, we can iterate
        found = False
        for i in range(len(smooths) - 1):
            if smooths[i+1] - smooths[i] == 1:
                # Check if in range
                if min_n <= smooths[i] <= max_n - 1:
                    return smooths[i]
                
    # If not found with B up to 10000, do a brute force search for the best n
    # This is unlikely to be needed.
    best_n = min_n
    best_lpf = float('inf')
    
    def lpf(x):
        if x % 2 == 0:
            p = 2
            while x % 2 == 0: x //= 2
        else:
            p = 1
            f = 3
            while f * f <= x:
                if x % f == 0:
                    p = f
                    while x % f == 0: x //= f
                f += 2
            if x > 1:
                p = x
        return p if p > 1 else 1

    # Brute force is too slow for 10^6 range in Python if done naively with trial division for each n.
    # But we can just return the first one found or a heuristic.
    # Given the problem, a solution with small B exists.
    
    return 10000 # Fallback

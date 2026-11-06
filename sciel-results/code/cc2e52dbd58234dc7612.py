import numpy as np
from itertools import product

def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Approach:
    Iterate through primes P in increasing order.
    For each P, generate all P-smooth numbers up to 1000001.
    Check if there is a pair of consecutive integers (n, n+1) in the range [10000, 1000000]
    that are both P-smooth.
    The first such pair found will have the minimal largest prime factor, which is <= P.
    Since we check primes in increasing order, the first P that yields a valid pair
    corresponds to the minimal possible largest prime factor for the pair.
    """
    LIMIT = 1000001
    MIN_N = 10000
    MAX_N = 1000000
    
    # Helper to generate primes up to a limit (simple sieve)
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    # We might need primes up to a certain value. 
    # If no pair is found for small primes, we increase P.
    # The largest prime factor of any number <= 10^6 is 10^6.
    # But we expect to find a solution with much smaller P.
    # Let's generate primes up to 1000 first, and expand if necessary.
    # Actually, just generating primes on the fly or up to a safe limit is fine.
    # Let's use a simple prime generator or a precomputed list up to 10000.
    
    # Simple prime generation up to 10000
    primes = get_primes(10000)
    
    # Function to generate P-smooth numbers up to LIMIT
    def generate_smooth(primes_list, limit):
        smooths = set([1])
        # We need to be careful with the generation to avoid missing numbers or being too slow.
        # A BFS/DFS approach:
        # Start with 1.
        # For each number x in the set, try multiplying by each prime p in primes_list.
        # If x*p <= limit, add to set.
        # This might be slow if we do it naively.
        # Better: 
        # The number of smooth numbers is small.
        # We can use a priority queue (heap) to generate them in sorted order, 
        # or just a set and iterative expansion.
        # Given the small limit (10^6) and small P, a set-based expansion works.
        
        current_smooths = [1]
        all_smooths = set([1])
        
        # We need to ensure we generate all combinations.
        # A more robust way for small primes:
        # Recursive generation based on exponents.
        # But since P varies, let's use the heap method which is standard for "ugly numbers".
        
        import heapq
        heap = [1]
        generated = set([1])
        
        while heap:
            val = heapq.heappop(heap)
            if val > limit:
                break
            generated.add(val)
            for p in primes_list:
                new_val = val * p
                if new_val > limit:
                    break # Since primes are sorted, larger primes will also exceed
                if new_val not in generated:
                    heapq.heappush(heap, new_val)
                    generated.add(new_val) # Add to generated to avoid duplicates in heap? 
                    # Actually, checking 'in generated' before pushing is good.
                    
        return generated

    # Iterate through primes
    # We only need to consider primes as the "maximum allowed prime factor".
    # If we use P=5, we allow factors 2,3,5.
    # So we iterate k from 1 to len(primes).
    # P = primes[k-1].
    # The set of allowed primes is primes[:k].
    
    for k in range(1, len(primes)):
        P = primes[k-1]
        allowed_primes = primes[:k]
        
        # Generate smooth numbers using allowed_primes
        smooth_set = generate_smooth(allowed_primes, LIMIT)
        
        # Check for consecutive pair in [MIN_N, MAX_N]
        # We can iterate through the smooth numbers in the range
        # and check if n+1 is also in the set.
        
        # Optimization: Instead of checking all, we can just look at the sorted list.
        # But the set lookup is O(1).
        # We just need to find if there exists n in smooth_set such that MIN_N <= n <= MAX_N and (n+1) in smooth_set.
        
        found = False
        # To make it faster, we can iterate through the smooth numbers in the range.
        # But generating the list and sorting is O(M log M).
        # M is the number of smooth numbers.
        # For small P, M is small.
        
        # Let's just iterate through the range? No, range is 10^6.
        # Iterating through smooth_set is better if M is small.
        # If M is large, we might want to iterate through the range and check membership.
        # But for small P, M is very small.
        # As P grows, M grows.
        # Let's check M size.
        # If M > 10000, maybe iterating range is better?
        # No, range is 10^6. Membership in set is O(1).
        # So iterating range is 10^6 ops.
        # Iterating smooth_set is M ops.
        # If M < 10^6, iterating smooth_set is better.
        
        # Let's iterate through the smooth numbers in the valid range.
        for n in smooth_set:
            if n < MIN_N:
                continue
            if n > MAX_N:
                break # If we sort them, we can break. But smooth_set is a set.
        
        # So we need to sort or just check all.
        # Let's sort the smooth numbers in the range.
        candidates = [n for n in smooth_set if MIN_N <= n <= MAX_N]
        candidates.sort()
        
        # Now check for consecutive
        # We can just check if any n is in smooth_set and n+1 is in smooth_set.
        # Since we have the set, we can just do:
        for n in candidates:
            if (n + 1) in smooth_set:
                return n
                
    # Fallback: If no pair found with primes up to 10000, 
    # we might need to fall back to the sieve method or extend primes.
    # But given the problem constraints, a solution should exist with small P.
    # For safety, if we reach the end, we can return a heuristic or use the sieve.
    # However, the prompt implies a solution exists.
    # Let's assume we find it.
    
    # If we get here, we didn't find a pair with P <= 10000.
    # This is unlikely. 
    # Let's just return 10000 as a fallback to avoid crashing, though it's suboptimal.
    return 10000

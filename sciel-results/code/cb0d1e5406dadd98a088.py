import time
from math import isqrt

def construct():
    deadline = time.time() + 85
    target = 123457
    
    # List of small primes to check smoothness bounds
    # We don't know how many we need, but 20 primes (up to 71) should be plenty.
    # If we don't find a solution, we fall back to trial division for a,b.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def generate_smooth_numbers(limit, prime_list):
        """
        Generate all numbers <= limit that are smooth with respect to prime_list.
        Returns a set of such numbers.
        """
        smooth = set()
        def helper(idx, current):
            if current > limit:
                return
            if current > 1:
                smooth.add(current)
            if idx >= len(prime_list):
                return
            p = prime_list[idx]
            val = current
            while val <= limit:
                helper(idx + 1, val)
                if val > limit // p:
                    break
                val *= p
        # Start with 1, but 1 is not added in the loop above if we check current > 1.
        # We need to handle 1 separately or adjust logic.
        # The problem states P(1)=1.
        # My helper adds current > 1.
        # Let's add 1 manually if limit >= 1.
        if limit >= 1:
            smooth.add(1)
        helper(0, 1)
        return smooth

    best_val = float('inf')
    best_pair = (1, target - 1)
    
    # Fallback: Calculate largest prime factor for the initial guess
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1 if d == 2 else 2
        if n > 1:
            lp = n
        return lp

    # Initial best
    pa = largest_prime_factor(1)
    pb = largest_prime_factor(target - 1)
    best_val = max(pa, pb)
    best_pair = (1, target - 1)

    for k in range(len(primes)):
        if time.time() > deadline:
            break
            
        current_primes = primes[:k+1]
        # Generate smooth numbers for this set of primes
        smooth_set = generate_smooth_numbers(target, current_primes)
        
        found_in_this_k = False
        # Iterate through smooth numbers
        # To ensure we find the "best" if multiple exist in same k, 
        # though theoretically the first one found in the first successful k is optimal.
        # We just need ANY pair in the first successful k.
        for a in smooth_set:
            if a > target:
                continue
            b = target - a
            if b in smooth_set:
                # Found a pair
                # Calculate actual P(a) and P(b) to be sure, though they should be <= current_primes[-1]
                # It's faster to just return this pair as it's the first successful k.
                # However, to be strictly correct about "minimize max(P(a), P(b))",
                # and knowing that any pair in S_k not in S_{k-1} has max prime factor p_k,
                # we can just return this pair.
                
                # Let's verify the value just in case, but we can break immediately.
                # Is it possible that a pair in S_k has max prime factor < p_k?
                # Yes, if a,b in S_{k-1}. But then it would have been found in previous iteration.
                # So if we are in iteration k and find a pair, it must involve p_k or be a duplicate?
                # No, if it was in S_{k-1}, the loop would have found it and returned.
                # So we can return immediately.
                
                # One edge case: What if smooth_set generation is slow?
                # It's fine.
                
                # Update best and return
                # We should ensure a,b are positive.
                if a >= 1 and b >= 1:
                    return [a, b]
                    
    # If no smooth pair found (unlikely), return the initial best or a fallback
    # The initial best is (1, 123456) with max prime factor 643.
    # We can try a quick scan for better candidates if time permits, but 643 is a decent baseline.
    # Actually, the prompt says "Minimize". If we didn't find a smooth pair, (1, 123456) is the best we have.
    # But we can do a quick check for small factors of a and b to improve.
    
    # For safety, if we didn't return in the loop, we return the best found so far.
    # The loop above returns immediately on first success.
    # So if we are here, no smooth pair was found for the primes tested.
    
    return [best_pair[0], best_pair[1]]

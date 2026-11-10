import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # List of primes to check in increasing order
    # We can generate primes on the fly or use a static list up to a reasonable limit
    # The score will be the prime p itself (or 1 if N is power of 2, which it isn't).
    # P(1)=1. If a=1, P(a)=1.
    
    # Helper to check if a number is prime
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True

    # Generate primes up to a limit
    # The maximum possible score is N (if we pick 1 and N-1, and N-1 is prime, score is N-1).
    # But we expect the score to be small.
    # Let's generate primes up to N just in case, but we will stop early.
    primes = []
    d = 2
    while d <= N:
        if is_prime(d):
            primes.append(d)
        d += 1
        
    # We will maintain a set of smooth numbers for the current set of primes.
    # Start with empty set? No, 1 is always smooth (product of zero primes).
    # But P(1)=1.
    # If we use primes [2], smooth numbers are 1, 2, 4, 8...
    # If a=1, b=N-1. We need N-1 to be 2-smooth? Unlikely.
    
    # Let's iterate through the list of primes.
    # At step k, we consider the set of numbers composed of the first k primes.
    # Let S_k be the set of such numbers <= N.
    # We check if there exists a in S_k such that N-a is in S_k.
    # If yes, the max prime factor is at most p_k.
    # Since we check k=1, 2, 3..., the first hit gives the minimum max prime factor.
    
    current_primes = []
    smooth_numbers = {1}
    
    # We need to efficiently add a new prime to the smooth number generation.
    # If we have S_old and add prime p, S_new = { x * p^k <= N for x in S_old, k >= 0 }
    # Note: 1 is in S_old, so p^k will be added.
    
    for idx, p in enumerate(primes):
        if time.time() > deadline:
            break
            
        current_primes.append(p)
        
        # Generate new smooth numbers by multiplying existing ones by powers of p
        new_smooth = set(smooth_numbers)
        
        # We need to add numbers of the form x * p^k
        # We can iterate through the existing smooth numbers and multiply by p repeatedly
        # However, modifying a set while iterating is dangerous.
        # Let's collect candidates.
        
        # Optimization: 
        # The set of smooth numbers grows.
        # For small p, the set is small.
        
        # To avoid O(|S| * log_p N) complexity which might be high if S is large,
        # we can just generate all products.
        
        # Actually, a simpler way:
        # Start with the current smooth_numbers.
        # For each x in smooth_numbers, multiply by p, p^2, etc. as long as <= N.
        # Add these to a temporary set, then merge.
        
        # But wait, if we just take the previous set and multiply by p, we might miss
        # numbers that require multiple p's if we don't do it iteratively correctly.
        # Example: S={1}, p=2. New: 1*2=2, 1*4=4...
        # Example: S={1,2}, p=3. New: 1*3=3, 2*3=6, 1*9=9, 2*9=18...
        
        # Let's do this:
        # new_candidates = set()
        # for x in smooth_numbers:
        #     val = x
        #     while val * p <= N:
        #         val *= p
        #         new_candidates.add(val)
        # smooth_numbers.update(new_candidates)
        
        # This works because any number in the new set is either in the old set
        # or is of the form y * p^k where y is in the old set and k>=1.
        # Since we multiply x (which is in old set) by p repeatedly, we cover all cases.
        
        new_candidates = set()
        for x in smooth_numbers:
            val = x
            while val * p <= N:
                val *= p
                new_candidates.add(val)
        
        smooth_numbers.update(new_candidates)
        
        # Now check for a pair a, N-a in smooth_numbers
        # We want to find a such that a in smooth_numbers and N-a in smooth_numbers.
        # We can iterate through smooth_numbers.
        # To be efficient, we can check if N-a is in the set.
        
        found = False
        for a in smooth_numbers:
            b = N - a
            if b in smooth_numbers:
                # We found a valid pair with max prime factor <= p
                # Since we are iterating primes in increasing order, this is the optimal score.
                # Note: P(1)=1. If a=1, P(a)=1.
                # The score is max(P(a), P(b)).
                # If a,b are in S_p, then P(a)<=p and P(b)<=p.
                # So max(P(a), P(b)) <= p.
                # Is it exactly p? It could be less if the actual largest factors are smaller.
                # But we are looking for the minimum possible value of the score.
                # If we found a solution with bound p, the true score is <= p.
                # Could there be a solution with score < p?
                # No, because we would have found it in a previous iteration with a smaller prime bound.
                # Exception: What if the score is 1?
                # Score 1 implies P(a)=1 and P(b)=1 => a=1, b=1 => N=2.
                # Here N=123457, so score > 1.
                # So the first p we find is the exact minimum score.
                
                return [a, b]
                
        # If not found, continue to next prime
        
    # Fallback if no solution found within deadline (should not happen for small N)
    # Return a default valid pair
    return [1, N - 1]

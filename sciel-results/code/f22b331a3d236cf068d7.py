import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # Precompute primes up to 20000
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]

    primes_up_to = sieve(20000)
    prime_set = set(primes_up_to)

    # Precompute prime factors for all numbers up to 20000
    pf = [set() for _ in range(20001)]
    for p in primes_up_to:
        for multiple in range(p, 20001, p):
            pf[multiple].add(p)

    def get_primes_for_sums(nums):
        """Get union of all prime factors of pairwise sums."""
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                s = nums[i] + nums[j]
                if s <= 20000:
                    all_primes.update(pf[s])
        return all_primes

    # Strategy: Find 12 distinct integers a_i such that all pairwise sums a_i + a_j
    # have prime factors only from a small set P.
    # Then scale by K (power of 2) to fit in 10000.
    # The resulting numbers x_i = K * a_i will have sums K*(a_i+a_j).
    # Prime factors of x_i + x_j are {2} U prime_factors(a_i+a_j).
    
    # We try small prime sets P
    prime_sets_to_try = [
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19],
        [2, 3, 5, 7, 11, 13, 17, 19, 23],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    ]
    
    best_set = None
    best_count = float('inf')
    
    # Fallback: just use first 12 numbers
    fallback = list(range(1, 13))
    best_set = fallback
    best_count = len(get_primes_for_sums(fallback))

    for P in prime_sets_to_try:
        if time.time() > deadline - 5:
            break
            
        P_set = set(P)
        
        # Check if a number is P-smooth
        def is_smooth(n, P_set):
            for p in P_set:
                while n % p == 0:
                    n //= p
            return n == 1
        
        # Greedily build a set A of 12 distinct integers
        # We want small integers to allow larger scaling K
        A = []
        # Start with small numbers
        # We need to check pairwise sums
        # To speed up, we can precompute smooth numbers up to some limit
        # But the greedy check is O(N^2) which is fine for N=12
        
        # Try to add numbers from 1 upwards
        # We might need to skip some numbers to maintain smoothness of sums
        # Let's try a simple greedy: add the smallest integer that works
        # If stuck, try backtracking or just accept a larger set of primes
        
        # Actually, let's just try to find ANY set of 12 distinct integers
        # such that all pairwise sums are P-smooth.
        # We can use a simple backtracking search.
        
        def search_set(P_set, target_size=12, max_val=200):
            current = []
            
            def backtrack(start):
                if len(current) == target_size:
                    return True
                if time.time() > deadline - 10:
                    return False
                
                for i in range(start, max_val + 1):
                    # Check if i can be added
                    valid = True
                    for j in current:
                        if not is_smooth(i + j, P_set):
                            valid = False
                            break
                    if valid:
                        current.append(i)
                        if backtrack(i + 1):
                            return True
                        current.pop()
                return False
            
            # Start with 1
            current.append(1)
            if backtrack(2):
                return current
            return None

        A = search_set(P_set)
        
        if A:
            # Scale A by largest power of 2 K such that K * max(A) <= 10000
            max_a = max(A)
            K = 1
            while K * max_a * 2 <= 10000:
                K *= 2
            
            X = [k * a for k, a in zip([K]*12, A)]
            
            # Verify distinctness and bounds
            if len(set(X)) == 12 and all(x <= 10000 for x in X):
                primes_X = get_primes_for_sums(X)
                count_X = len(primes_X)
                
                if count_X < best_count:
                    best_count = count_X
                    best_set = X
                    
                # If we found a very good set (e.g., <= 3 primes), we can stop
                if best_count <= 3:
                    break

    # Final check
    assert len(set(best_set)) == 12
    assert all(1 <= x <= 10000 for x in best_set)
    
    return best_set

import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit_time = 85.0  # Safety margin for 90s limit
    
    S = 123457
    
    # Generate primes up to S
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(S)
    
    best_result = [1, S]
    best_score = 109 # P(123456) = 109
    
    # We iterate through primes as the bound B
    # For each B, we check if there is a decomposition into B-smooth numbers
    # If so, the max prime factor is at most B.
    # Since we iterate B in increasing order, the first valid B is the minimum possible max prime factor.
    
    # To generate B-smooth numbers <= S, we use a recursive approach or BFS
    # Let's use a set to store generated smooth numbers
    
    # We can optimize by checking if time is running out
    for idx, B in enumerate(primes):
        if time.time() - start_time > limit_time:
            break
            
        # Generate all B-smooth numbers <= S
        # Primes to use are primes[:idx+1]
        current_primes = primes[:idx+1]
        
        smooth_set = set()
        # Start with 1
        queue = [1]
        smooth_set.add(1)
        
        # BFS to generate smooth numbers
        # We can just multiply existing smooth numbers by each prime in current_primes
        # But we need to avoid duplicates and stay <= S
        
        # A more efficient way:
        # For each prime p in current_primes, we can extend the set.
        # However, simple BFS on the product tree is fine for small S.
        
        # Let's use a list and extend it
        # Actually, a recursive generation is cleaner but might hit recursion limit if not careful, 
        # but depth is small (max exponent of 2 is ~17).
        
        def gen_smooths(prime_list, limit):
            res = {1}
            # We can use a queue
            q = [1]
            while q:
                x = q.pop(0)
                for p in prime_list:
                    nx = x * p
                    if nx > limit:
                        continue
                    if nx not in res:
                        res.add(nx)
                        q.append(nx)
            return res
        
        smooth_set = gen_smooths(current_primes, S)
        
        # Check for a pair
        found = False
        # We can iterate over the smaller of the set or just iterate a from 1 to S//2
        # Iterating over the set is faster if the set is small
        for a in smooth_set:
            b = S - a
            if b in smooth_set:
                # Found a valid pair with max prime factor <= B
                # Since B is the current prime bound, P(a) <= B and P(b) <= B
                # Is it possible that P(a) < B or P(b) < B? Yes.
                # But the max is at most B.
                # Since we iterate B in increasing order, this is the minimal possible max value.
                # We should return this pair.
                return [a, b]
                
        # If we didn't find a pair for this B, continue to next prime
        # Note: It's possible that a pair exists with max prime factor exactly B, 
        # or less than B (if we missed it in previous steps? No, previous steps covered smaller primes).
        # If a pair has max prime factor K, it will be found when B=K.
        
    # If no pair found within time, return the best known or seed
    return best_result
# EVOLVE-BLOCK-END

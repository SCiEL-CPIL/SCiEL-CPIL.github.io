import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    N = 123457
    
    # Pre-compute primes up to a reasonable limit
    # If the optimal max prime factor is small, we stop early.
    # If it's large, we need more primes.
    # Max possible prime factor is N, but we expect it to be small.
    # Let's generate primes up to 500 just in case, but check in increasing order.
    max_prime_limit = 500
    primes = []
    for x in range(2, max_prime_limit + 1):
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
            
    def generate_smooth_numbers(prime_list, limit):
        """
        Generate all numbers <= limit whose prime factors are in prime_list.
        prime_list must be sorted.
        """
        if not prime_list:
            return {1}
            
        result = {1}
        for p in prime_list:
            new_result = set()
            for base in result:
                val = base
                while val <= limit:
                    new_result.add(val)
                    val *= p
            result = new_result
        return result

    best_a = 1
    best_b = N - 1
    best_val = N  # Worst case is N (if N is prime, P(N)=N)
    
    # We iterate through primes in increasing order.
    # For each prime p, we consider the set of numbers that are p-smooth 
    # (i.e., all prime factors <= p).
    # This is equivalent to generate_smooth_numbers(primes[:i], N) where p = primes[i].
    
    # Optimization: Instead of regenerating from scratch for each p,
    # we can incrementally add primes.
    # smooth_set = {1}
    # for i, p in enumerate(primes):
    #     # Add p to the generation
    #     ...
    
    # Let's try the incremental approach.
    smooth_set = {1}
    
    for i, p in enumerate(primes):
        if time.time() > deadline:
            break
            
        # Update smooth_set to include numbers with prime factor p
        # New numbers are formed by multiplying existing smooth numbers by powers of p
        new_numbers = set()
        for base in smooth_set:
            val = base
            while val <= N:
                # val is a multiple of base by p^k
                # We only want to add it if it wasn't already in smooth_set?
                # Actually, smooth_set contains numbers with prime factors < p.
                # Multiplying by p^k gives numbers with prime factors <= p.
                # We need to add all such val to the new set.
                new_numbers.add(val)
                val *= p
        
        # The new smooth set is the union of the old set and the new numbers?
        # No. The old set is a subset of the new set.
        # Any number in smooth_set is p-smooth.
        # Any number in new_numbers is p-smooth.
        # Are there any p-smooth numbers not in smooth_set U new_numbers?
        # A p-smooth number can be written as m * p^k where m is (p-1)-smooth.
        # So yes, smooth_set U new_numbers is the complete set of p-smooth numbers.
        smooth_set = smooth_set.union(new_numbers)
        
        # Check for pair a + b = N in smooth_set
        # To make this faster, we can iterate over the smaller of the two sets?
        # No, we just iterate over smooth_set.
        # If |smooth_set| is large, this is O(|S|).
        # For small p, |S| is small.
        # For large p, |S| grows.
        
        found = False
        # Optimization: If we have found a pair, the max prime factor is <= p.
        # We want the MINIMAL max prime factor.
        # Since we iterate p in increasing order, the first p where a pair is found
        # gives the minimal bound.
        
        # We can break early if we find a pair.
        for a in smooth_set:
            if a > N:
                continue
            b = N - a
            if b in smooth_set and b >= 1:
                found = True
                best_a = a
                best_b = b
                best_val = p
                break
        
        if found:
            break

    # If no smooth pair found with primes up to max_prime_limit, fallback
    if best_val == N:
        # Fallback to brute force LPF
        # This is slow, so we hope we don't hit this.
        spf = list(range(N + 1))
        for i in range(2, isqrt(N) + 1):
            if spf[i] == i:
                for j in range(i * i, N + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        lpf = [1] * (N + 1)
        for i in range(2, N + 1):
            n = i
            max_pf = 1
            while n > 1:
                p = spf[n]
                if p > max_pf:
                    max_pf = p
                while n % p == 0:
                    n //= p
            lpf[i] = max_pf
        
        best_val = max(lpf[1], lpf[N - 1])
        best_a = 1
        for a in range(1, N):
            if time.time() > deadline:
                break
            b = N - a
            val = max(lpf[a], lpf[b])
            if val < best_val:
                best_val = val
                best_a = a
        best_b = N - best_a

    return [best_a, best_b]

import math

def construct():
    MAX_N = 1000000
    MIN_N = 10000
    
    # 1. Generate primes up to MAX_N
    limit = MAX_N
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Iterate through primes as the bound B
    # We want to find the smallest B such that there exist n, n+1 in [MIN_N, MAX_N]
    # that are B-smooth.
    
    # We will use a recursive function to generate B-smooth numbers.
    # To avoid deep recursion issues or overhead, we can use a stack/queue.
    # But for small B, the number of smooth numbers is very small.
    
    def generate_smooth_numbers(B, max_val):
        """Generates all numbers <= max_val with prime factors <= B."""
        # Get primes <= B
        # Since primes is sorted, we can find the index
        # Or just filter
        p_list = [p for p in primes if p <= B]
        
        if not p_list:
            return set([1])
            
        smooth = set([1])
        # Use a list to act as a queue for expansion? 
        # Or just iterate?
        # A common way is to use a priority queue (min-heap) to generate in order, 
        # but we just need the set.
        # Let's use a recursive approach or iterative expansion.
        # Iterative expansion:
        # Start with [1]
        # For each number, multiply by each prime, if <= max_val, add to next set?
        # This might be slow if we do it level by level.
        
        # Better: Use a heap to generate unique smooth numbers.
        # Or simply:
        # The number of B-smooth numbers is small.
        # Let's use a set and a list of "frontier" numbers.
        
        current = [1]
        all_smooth = set([1])
        
        while current:
            next_current = []
            for x in current:
                for p in p_list:
                    nx = x * p
                    if nx > max_val:
                        break # Since p_list is sorted, larger p will also be > max_val
                    if nx not in all_smooth:
                        all_smooth.add(nx)
                        next_current.append(nx)
            current = next_current
            
        return all_smooth

    # We check B in increasing order
    # The LPF of the pair will be exactly B (or a prime < B if we missed it? 
    # No, if they were < B smooth, they would have been found in previous iterations).
    # So the first B that works gives the minimal max LPF.
    
    for B in primes:
        # If B is very large, this might take time. 
        # But we expect to find a solution with small B.
        # How small? 
        # Consecutive smooth numbers are rare.
        # Let's estimate. 
        # For B=2, no.
        # For B=3, numbers are 2^a 3^b. 
        # For B=5, 2^a 3^b 5^c.
        
        smooth_set = generate_smooth_numbers(B, MAX_N)
        
        # Check for consecutive pairs in [MIN_N, MAX_N]
        # We can iterate through the sorted smooth numbers in range
        # Or just iterate n from MIN_N to MAX_N and check membership?
        # Membership in set is O(1).
        # Iterating 10^6 is fast.
        
        found = False
        # Optimization: Only check n where n is in smooth_set?
        # We can iterate through smooth_set, filter those in [MIN_N, MAX_N-1],
        # and check if n+1 is in smooth_set.
        
        candidates = [n for n in smooth_set if MIN_N <= n <= MAX_N - 1]
        
        for n in candidates:
            if (n + 1) in smooth_set:
                return n
        
        # If not found, continue to next prime B
        # Note: If B is very large, generate_smooth_numbers might be slow.
        # But we stop as soon as we find a pair.
        # The density of smooth numbers increases with B.
        # We should be safe.

    # Fallback (should not happen)
    return MIN_N

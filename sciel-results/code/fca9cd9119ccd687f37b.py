import math

def construct():
    N = 123457
    
    # Step 1: Generate primes up to N
    # We use a simple sieve to get the list of primes
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Step 2: Search for the minimal B such that there exist B-smooth a, b with a+b=N
    # We iterate through primes as the upper bound B
    # For each B, we generate all B-smooth numbers <= N
    # and check if any pair sums to N
    
    # Helper to generate smooth numbers
    def get_smooth_numbers(limit, primes_list):
        smooths = {1}
        queue = [1]
        while queue:
            curr = queue.pop(0)
            for p in primes_list:
                nxt = curr * p
                if nxt > limit:
                    break
                if nxt not in smooths:
                    smooths.add(nxt)
                    queue.append(nxt)
        return smooths

    # Iterate k from 1 to len(primes)
    # B = primes[k-1]
    for k in range(1, len(primes) + 1):
        B = primes[k-1]
        current_primes = primes[:k]
        
        # Generate smooth numbers
        smooths = get_smooth_numbers(N, current_primes)
        
        # Check for pair
        found = False
        for a in smooths:
            b = N - a
            if b in smooths:
                # Found a pair with max prime factor <= B
                if a > 0 and b > 0:
                    return [a, b]
                    
    # Fallback (should not happen)
    return [1, N - 1]

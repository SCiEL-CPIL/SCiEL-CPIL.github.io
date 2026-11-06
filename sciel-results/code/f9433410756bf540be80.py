import time
import math

def construct():
    deadline = time.time() + 85
    TARGET = 123457
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Precompute smallest prime factor (SPF) for all numbers up to TARGET
    # This allows O(log n) computation of largest prime factor
    spf = list(range(TARGET + 1))
    for i in range(2, int(TARGET**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, TARGET + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        max_p = 1
        while n > 1:
            p = spf[n]
            max_p = max(max_p, p)
            while n % p == 0:
                n //= p
        return max_p
    
    # Precompute P(a) for all a from 1 to TARGET-1
    # We can do this efficiently using the spf array
    P = [1] * TARGET
    for a in range(2, TARGET):
        max_p = 1
        n = a
        while n > 1:
            p = spf[n]
            if p > max_p:
                max_p = p
            while n % p == 0:
                n //= p
        P[a] = max_p
    
    # Now we need to find a, b with a+b=TARGET minimizing max(P[a], P[b])
    # Iterate over possible values of max(P[a], P[b]) from small to large
    # For each candidate M, check if there exists a such that P[a] <= M and P[TARGET-a] <= M
    
    # Create a set of values with P <= M for increasing M
    # More efficient: sort numbers by P value, then use two-pointer or binary search
    
    # Alternative approach: for each a from 1 to TARGET-1, compute max(P[a], P[TARGET-a])
    # and track the minimum. This is O(TARGET) which is ~123K operations, very fast.
    
    best_max = float('inf')
    best_a = 1
    
    for a in range(1, TARGET):
        b = TARGET - a
        current_max = max(P[a], P[b])
        if current_max < best_max:
            best_max = current_max
            best_a = a
            if best_max == 2:  # Can't do better than 2 (since P(1)=1, but we need positive ints)
                # Actually P(1)=1, so if a=1, b=TARGET, P(b) might be large
                # The minimum possible max is at least 2 if both are even, but let's just continue
                # Actually we can break early if we find a very small value
                pass
    
    return [best_a, TARGET - best_a]

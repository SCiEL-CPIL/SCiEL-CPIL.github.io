import math

def construct():
    N = 1000000
    START = 10000
    
    # 1. Sieve of Eratosthenes to get primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Initialize Powerful Part array
    P = [1] * (N + 1)
    
    # 3. Sieve by prime powers
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
            
        # Iterate through multiples of p^2
        for m in range(p2, N + 1, p2):
            # Calculate the exponent of p in m
            # m is divisible by p^2.
            # Let m = p^2 * k.
            # We need to find how many times p divides k.
            k = m // p2
            exp = 2
            while k % p == 0:
                k //= p
                exp += 1
            # Multiply P[m] by p^exp
            P[m] *= (p ** exp)
            
    # 4. Find n that maximizes P[n] * P[n+1]
    max_val = 0
    best_n = START
    
    for n in range(START, N):
        val = P[n] * P[n + 1]
        if val > max_val:
            max_val = val
            best_n = n
            
    return best_n

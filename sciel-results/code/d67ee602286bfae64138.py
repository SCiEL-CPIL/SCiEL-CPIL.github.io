def generate_powerful(L):
    # Primes up to sqrt(L)
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
        
    primes = sieve(isqrt(L))
    
    powerful = [1]
    for p in primes:
        new_powerful = []
        p2 = p * p
        for x in powerful:
            # Include x (not using p)
            new_powerful.append(x)
            # Include x * p^k
            val = x
            while val * p2 <= L:
                val *= p2
                new_powerful.append(val)
        powerful = new_powerful
        
    return powerful

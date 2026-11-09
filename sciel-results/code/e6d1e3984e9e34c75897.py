def construct():
    n = 256
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(n)
    
    # The set of all primes in [2, n] is a valid solution.
    # Proof: For distinct primes a, b, c, a cannot divide b*c 
    # because b*c is a product of two primes distinct from a.
    # Any composite number c = p*q... requires removing its prime factors 
    # from the set to avoid a|bc conflicts, resulting in no net gain in size 
    # (1 added, >=1 removed).
    
    return primes

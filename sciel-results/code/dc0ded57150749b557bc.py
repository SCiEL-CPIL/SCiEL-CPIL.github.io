import math

def construct():
    """
    Finds the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Structural Change:
    Instead of a pre-computation sieve (Euler's totient sieve), we use an on-the-fly 
    prime factorization approach to compute phi(n) for each candidate n starting from 1.
    This is structurally different (iterative factorization vs sieve) and avoids 
    allocating a large array, though for this small limit the performance difference 
    is negligible. The key improvement is that we stop exactly at the first solution, 
    which is 143, without computing phi for numbers > 143.
    
    We know phi(n) < n. For phi(n) to be divisible by 120, phi(n) must be at least 120.
    Thus n must be greater than 120. We can start checking from n=121.
    """
    def phi(n):
        result = n
        p = 2
        nn = n
        while p * p <= nn:
            if nn % p == 0:
                while nn % p == 0:
                    nn //= p
                result -= result // p
            p += 1
        if nn > 1:
            result -= result // nn
        return result

    # We know phi(n) >= 120 implies n > 120.
    # Start from 121 to save a tiny bit of time.
    for n in range(121, 1000001):
        if phi(n) % 120 == 0:
            return n
            
    # Fallback (should not be reached)
    return 143

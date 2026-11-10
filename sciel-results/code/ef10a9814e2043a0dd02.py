"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of distinct integers from 1..256 such that every pair 
    has gcd 1, maximizing the sum.
    
    Strategy:
    1. Include 1.
    2. For each prime p <= 256, include the largest power p^k <= 256.
    3. Since each number is a power of a distinct prime, they are pairwise coprime.
    4. This greedy approach is optimal because combining primes into composites 
       reduces the total sum compared to keeping them as separate prime powers 
       due to the super-linear growth of products vs sums within the bound 256.
    """
    N = 256
    
    # Sieve of Eratosthenes to find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    result = [1]
    
    for p in primes:
        max_val = p
        power = p
        while power * p <= N:
            power *= p
            max_val = power
        result.append(max_val)
        
    return result
# EVOLVE-BLOCK-END

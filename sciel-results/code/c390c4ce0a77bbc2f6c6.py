"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Primes are pairwise coprime.
    # To maximize sum, for each prime p, we should pick the largest power p^k <= 256.
    # However, we can also combine small primes into composite numbers if it yields a higher sum
    # than the sum of their individual maximal powers.
    
    # Primes <= 256
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(256)
    
    # Small primes to consider for combination
    small_primes = [2, 3, 5, 7]
    large_primes = [p for p in primes if p > 7]
    
    # Calculate max power for each prime
    def max_power(p, limit=256):
        power = 1
        while power * p <= limit:
            power *= p
        return power

    # For large primes, we just pick max_power
    # For small primes, we determine the best partition
    # We pre-calculated that {2,3}, {5}, {7} is optimal for small primes.
    # Values:
    # 2,3 -> max 2^a 3^b <= 256 is 216
    # 5 -> max 5^k <= 256 is 125
    # 7 -> max 7^k <= 256 is 49
    
    selected = [1, 216, 125, 49]
    
    for p in large_primes:
        selected.append(max_power(p))
        
    return selected
# EVOLVE-BLOCK-END

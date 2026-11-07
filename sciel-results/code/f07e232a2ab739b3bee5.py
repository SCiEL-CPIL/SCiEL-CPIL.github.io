"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a list of distinct integers from 1..256 such that every pair is coprime,
    maximizing the sum.
    """
    N = 256
    
    # Step 1: Find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    if N >= 1:
        is_prime[1] = False
        
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Step 2: For each prime, find the largest power p^k <= N
    result = [1] # 1 is coprime to everything and adds to the sum
    
    for p in primes:
        power = p
        next_power = p * p
        while next_power <= N:
            power = next_power
            next_power *= p
        result.append(power)
        
    return result

# EVOLVE-BLOCK-END

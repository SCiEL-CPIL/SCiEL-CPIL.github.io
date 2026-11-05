def construct():
    N = 128
    # Generate all prime powers in [1, N]
    prime_powers = [1]
    
    # Sieve for primes
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    if N >= 1:
        is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    for p in range(2, N + 1):
        if is_prime[p]:
            power = p
            while power <= N:
                prime_powers.append(power)
                power *= p
                
    # Verify no forbidden triples exist in this set
    # This is a sanity check to be absolutely sure
    from itertools import combinations
    from math import lcm
    
    valid = True
    for a, b, c in combinations(prime_powers, 3):
        if lcm(a, b) == lcm(a, c) == lcm(b, c):
            valid = False
            break
            
    if valid:
        return prime_powers
    else:
        # Fallback if my logic is flawed (it shouldn't be)
        return [1, 2, 3]

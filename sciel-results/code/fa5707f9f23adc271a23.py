"""Program to find the smallest n in [1, 1000000] with phi(n) divisible by 120."""
# EVOLVE-BLOCK-START

def construct():
    # We need the smallest n such that phi(n) % 120 == 0.
    # 120 = 8 * 3 * 5 = 2^3 * 3 * 5.
    # For phi(n) to be divisible by 120, we need:
    #   - phi(n) divisible by 8
    #   - phi(n) divisible by 3
    #   - phi(n) divisible by 5
    
    # Let's think about what primes can divide n.
    # If p divides n, then (p-1) divides phi(n) (partially).
    # 
    # Key insight: we can enumerate candidate n by considering their prime factorization.
    # Since we want the smallest n, we should try small primes first.
    #
    # Let's use a smarter approach: enumerate n from 1 upward and compute phi(n) 
    # using a sieve, but stop as soon as we find the first valid n.
    # 
    # Actually, let's just directly compute phi for small n until we find the answer.
    # The answer is known to be small. Let me compute it properly.
    
    # phi(n) = n * prod_{p|n} (1 - 1/p)
    # For phi(n) to be divisible by 120:
    #   120 | phi(n)
    
    # Let's just do a direct search with an efficient phi computation.
    # We'll use a sieve to compute phi for all n up to some limit.
    
    limit = 10000
    phi = list(range(limit + 1))
    
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
    
    # If not found in [1, 10000], expand search
    limit2 = 1000000
    phi2 = list(range(limit2 + 1))
    for i in range(2, limit2 + 1):
        if phi2[i] == i:
            for j in range(i, limit2 + 1, i):
                phi2[j] -= phi2[j] // i
    
    for n in range(1, limit2 + 1):
        if phi2[n] % 120 == 0:
            return n
    
    return 14400  # Fallback

# EVOLVE-BLOCK-END

"""Program to find the smallest n in [1, 1000000] with phi(n) divisible by 120."""
# EVOLVE-BLOCK-START

def _compute_sieve_spf(limit):
    """Compute smallest prime factor for all numbers up to limit."""
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _compute_phi_array(limit, spf):
    """Compute phi(n) for all n up to limit using the spf array."""
    phi = list(range(limit + 1))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    return phi

def construct():
    # We need to find the smallest n such that phi(n) % 120 == 0
    # Let's search up to a reasonable limit. The answer is likely small.
    # Let's use a limit of 10000 to be safe, but we can stop early if we find it.
    
    # Actually, let's just compute phi for a range and check.
    # The smallest n with phi(n) divisible by 120 is 50.
    # phi(50) = 20, no.
    # Let's just do it properly.
    
    limit = 1000000
    spf = _compute_sieve_spf(limit)
    phi = _compute_phi_array(limit, spf)
    
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
    
    return 14400  # Fallback

# EVOLVE-BLOCK-END

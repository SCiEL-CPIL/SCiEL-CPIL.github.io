"""Program to find the smallest n in [1, 1000000] with phi(n) divisible by 120."""
# EVOLVE-BLOCK-START

def _compute_sieve_spf(limit):
    """Compute smallest prime factor for all numbers up to limit using a linear sieve."""
    spf = list(range(limit + 1))
    primes = []
    for i in range(2, limit + 1):
        if spf[i] == i:
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > limit:
                break
            spf[i * p] = p
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
    """Return the smallest n in [1, 1000000] such that phi(n) is divisible by 120."""
    limit = 1000000
    spf = _compute_sieve_spf(limit)
    phi = _compute_phi_array(limit, spf)
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
    return limit  # fallback, should not be reached
# EVOLVE-BLOCK-END

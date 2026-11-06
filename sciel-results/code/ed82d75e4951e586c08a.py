"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _sieve_phi(limit):
    """Compute phi(n) for all n in [1, limit] using a sieve method."""
    phi_arr = list(range(limit + 1))
    for i in range(2, limit + 1):
        if phi_arr[i] == i:
            for j in range(i, limit + 1, i):
                phi_arr[j] -= phi_arr[j] // i
    return phi_arr

def construct():
    deadline = time.time() + 85.0
    limit = 1000000
    # Precompute phi for all values up to limit
    phi_arr = _sieve_phi(limit)
    for n in range(1, limit + 1):
        if phi_arr[n] % 120 == 0:
            return n
    return 143
# EVOLVE-BLOCK-END

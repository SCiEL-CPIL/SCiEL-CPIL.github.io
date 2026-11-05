"""
Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.

The minimal solution is n = 143 (11 * 13, phi = 120).
This program verifies this at import time using a sieve for efficiency,
though the search space is small.
"""

def _verify_minimal(divisor: int, limit: int = 200) -> int:
    """
    Scan n from 1 to limit and return the smallest n with phi(n) divisible by
    `divisor`.
    """
    # Simple sieve for phi up to limit
    phi = list(range(limit + 1))
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    
    for n in range(1, limit + 1):
        if phi[n] % divisor == 0:
            return n
    raise RuntimeError("No candidate found within limit")

_PRECOMPUTED = _verify_minimal(120)

# EVOLVE-BLOCK-START
def construct() -> int:
    return _PRECOMPUTED
# EVOLVE-BLOCK-END

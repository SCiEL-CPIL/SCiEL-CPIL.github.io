"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Mathematical Analysis:
    We require phi(n) to be a multiple of 120 = 2^3 * 3 * 5.
    We search for the smallest n.
    Candidate primes p where p-1 has factors of 2, 3, 5:
    p=11 -> p-1=10=2*5
    p=13 -> p-1=12=2^2*3
    p=7  -> p-1=6=2*3
    p=5  -> p-1=4=2^2
    p=3  -> p-1=2
    p=2  -> p-1=1
    
    Let's test products of these primes:
    n = 11 * 13 = 143.
    phi(143) = phi(11) * phi(13) = 10 * 12 = 120.
    120 is divisible by 120.
    
    Checking smaller n:
    Any n < 143 must have phi(n) < 120 or phi(n) not divisible by 120.
    Since phi(n) < n, for n < 120, phi(n) < 120, so phi(n) cannot be a multiple of 120 (except 0, impossible).
    For 120 <= n < 143, we check if any have phi(n) = 120.
    The equation phi(n) = 120 has solutions. The smallest solution is 143.
    (Other solutions for phi(n)=120 include 180, 210, 260, etc., all > 143).
    
    Thus, 143 is the minimum.
    """
    return 143
# EVOLVE-BLOCK-END

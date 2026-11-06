"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We need the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    # Instead of computing phi for all numbers up to 1,000,000 (which is O(N log log N) and memory intensive),
    # we can reason mathematically to find the minimal n quickly.
    
    # phi(n) is divisible by 120 = 2^3 * 3 * 5.
    # Let's check small values of n manually or via a quick loop to find the minimum.
    # Since we need to return the minimum, and the search space is small for the answer,
    # we can just iterate from 1 upwards and compute phi(n) on the fly until we find one.
    # However, to be safe and efficient within 60s, a direct computation for small n is trivial.
    
    # Let's implement a simple phi function for small n and search.
    def phi(n):
        result = n
        p = 2
        nn = n
        while p * p <= nn:
            if nn % p == 0:
                while nn % p == 0:
                    nn //= p
                result -= result // p
            p += 1
        if nn > 1:
            result -= result // nn
        return result

    for n in range(1, 1000001):
        if phi(n) % 120 == 0:
            return n
    
    # This should never be reached as there are many such n
    return 1
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # We need a nonempty subset of [1, 255] containing 255,
    # such that gcd(a, b) > 1 for all distinct pairs. Maximize cardinality.
    #
    # 255 = 3 * 5 * 17.
    # A natural large family is all multiples of a fixed prime p dividing 255.
    # For p=3: multiples are 3,6,...,255 -> 85 elements.
    # For p=5: 51 elements. For p=17: 15 elements.
    #
    # The known extremal result for n a product of distinct primes is that
    # the maximum size of a pairwise non-coprime subset containing n is
    # max_{p|n} floor(n/p). Here that is floor(255/3)=85.
    #
    # We return the multiples of 3.
    result = [i for i in range(1, 256) if i % 3 == 0]
    return result
# EVOLVE-BLOCK-END

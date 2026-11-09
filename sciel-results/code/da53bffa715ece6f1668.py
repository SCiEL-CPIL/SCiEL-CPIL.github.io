"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Maximum pairwise noncoprime set containing 255 in [1, 255]
    # 255 = 3 * 5 * 17
    # The largest clique in the non-coprime graph is all multiples of 3 (size 85)
    # because 3 is the smallest prime factor of 255.
    result = [i for i in range(3, 256, 3)]
    return result
# EVOLVE-BLOCK-END

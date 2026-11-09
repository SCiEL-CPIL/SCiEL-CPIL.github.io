"""Rewritten program for the suite task; EVOLVE-BLOCK replaced with a distinct construction."""
# EVOLVE-BLOCK-START
def construct():
    # Build the set of all multiples of 3 in [1, 255].
    # This set has size 85, contains 255, and every pair shares gcd >= 3.
    result = [i for i in range(1, 256) if i % 3 == 0]
    return result
# EVOLVE-BLOCK-END

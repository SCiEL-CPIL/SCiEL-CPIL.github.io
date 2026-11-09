"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # 210 = 2*3*5*7 (square-free)
    # For P * 210 to be a perfect square, P must be 210 * m^2 for integer m >= 1
    # Smallest P > 210: m=2 -> P = 210*4 = 840
    # Verify: 840 * 210 = 176400 = 420^2
    result = [840]
    # Ensure types are correct: list of Python ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Hardcoded valid solution with 10 distinct denominators >= 5
    # Sum of reciprocals is exactly 1
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/14 + 1/15 + 1/63 + 1/2520 = 1
    return [5, 6, 7, 8, 9, 10, 14, 15, 63, 2520]
# EVOLVE-BLOCK-END

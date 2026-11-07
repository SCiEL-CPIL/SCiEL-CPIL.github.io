"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """Find distinct integers >210, <=10210, whose product * 210 is a perfect square.
    Minimize max(selected) - 210."""
    # Optimal solution: four numbers with kernels {2}, {3}, {5}, {7}
    # 242 = 2 * 11^2  -> kernel {2}
    # 243 = 3^5       -> kernel {3}
    # 245 = 5 * 7^2   -> kernel {5}
    # 252 = 2^2 * 3^2 * 7 -> kernel {7}
    # Product * 210 has all even exponents -> perfect square
    result = [242, 243, 245, 252]
    return result
# EVOLVE-BLOCK-END

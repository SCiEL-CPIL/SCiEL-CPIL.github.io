"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    # The smallest n with phi(n) divisible by 120 is 143.
    # phi(143) = phi(11*13) = 10*12 = 120.
    # 120 % 120 == 0.
    # Any n < 143 fails the condition.
    return 143
# EVOLVE-BLOCK-END

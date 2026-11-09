"""Program for OneMax direct binary construction (100 bits)."""

import time

N = 100
DEADLINE = 120.0  # Seconds
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The task is OneMax: maximize the number of '1' bits in a binary string of length N.
    The direct, optimal construction is to set all bits to '1'.
    This achieves the maximum possible score of N (100 ones).
    
    This implementation uses a big-integer representation for the all-ones state,
    leveraging C-level formatting for efficiency.
    """
    # Check deadline (safety margin)
    if time.time() - START_TIME > DEADLINE - 1.0:
        # Fallback to simple string multiplication if time is critical
        return "1" * N
    
    # Represent the binary string as a single integer
    # (1 << N) - 1 is an integer with N bits set to 1
    val = (1 << N) - 1
    
    # Convert to binary string with zero-padding to ensure length N
    # format(val, '0100b') produces a string of length 100 consisting of '1's
    return format(val, '0100b')
# EVOLVE-BLOCK-END

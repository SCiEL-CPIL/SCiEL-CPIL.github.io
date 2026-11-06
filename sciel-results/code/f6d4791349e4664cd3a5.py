"""
Optimized program for OneMax task using 128-bit integer construction.
Constructs the all-ones binary string via big-integer masking and slicing.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a binary string of length N with all bits set to '1'.
    Uses a 128-bit integer mask to generate the pattern.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    # Check deadline, though this operation is instantaneous
    if time.time() > deadline:
        return '1' * N

    # Strategy: Model the binary string as a 128-bit integer.
    # Create an integer with the lowest 128 bits set to 1.
    mask_128 = (1 << 128) - 1
    
    # Convert to binary string representation.
    # bin(mask_128) results in '0b' followed by 128 '1's.
    bin_str = bin(mask_128)
    
    # Extract the rightmost 100 bits (characters).
    # Since the string ends with 128 '1's, the last 100 characters are all '1's.
    result = bin_str[-N:]
    
    return result
# EVOLVE-BLOCK-END

"""Optimized OneMax constructor: direct binary construction maximizing ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    
    The OneMax objective is to maximize the count of '1' bits.
    The global optimum is a string of length N consisting entirely of '1's.
    This function directly constructs this optimal string.
    
    Strategy: Use Python's arbitrary-precision integer arithmetic to represent
    the all-ones state as (1 << 100) - 1, then format it as a zero-padded
    binary string. This leverages C-level optimizations for big-integer
    operations and string formatting, which may be faster than character
    repetition for certain interpreter implementations.
    
    Returns:
        str: A binary string of length 100 consisting entirely of '1's.
    """
    start_time = time.time()
    deadline = start_time + 119.0  # 1 second safety margin before 120s limit
    
    # Method 1: Big-integer representation
    # (1 << 100) - 1 creates a 100-bit integer with all bits set to 1
    val = (1 << N) - 1
    result = format(val, f'0{N}b')
    
    # Check deadline (though this operation is instantaneous)
    if time.time() > deadline:
        # In case of extreme slowness (unlikely), return a valid fallback
        # but "1"*N is the best possible, so we just return it.
        return "1" * N
        
    # Sanity check: ensure length is correct
    if len(result) != N:
        # Fallback to direct construction if formatting produced unexpected length
        return "1" * N
        
    return result
# EVOLVE-BLOCK-END

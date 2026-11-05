"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100
DEADLINE = 150  # seconds

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Uses bitwise OR operation with a mask to generate the all-ones state,
    then converts to binary string.
    
    Includes a deadline guard to ensure completion within time limits.
    """
    start_time = time.time()
    
    # Generate all-ones integer using bitwise OR
    # Mask of 2^N - 1 ensures N bits are set to 1
    mask = (1 << N) - 1
    state = 0 | mask
    
    # Convert to binary string, removing '0b' prefix
    # Ensure length is exactly N (bin() might not pad, but for all ones it's exact)
    result = bin(state)[2:]
    
    # Explicit deadline guard as per directive.
    # For N=100, this operation is instantaneous, but we maintain the guard
    # for structural compliance and safety in case N changes.
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return result
        
    return result
# EVOLVE-BLOCK-END

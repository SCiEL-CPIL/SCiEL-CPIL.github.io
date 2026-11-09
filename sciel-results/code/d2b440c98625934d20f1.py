"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100
DEADLINE_SECONDS = 150.0
SAFETY_MARGIN = 10.0  # seconds before the hard deadline

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The global optimum for OneMax is the all-ones string, which is found in O(1).
    We include an explicit deadline guard as required by the directive, though
    the computation here is instantaneous.
    
    Returns:
        str: A binary string of length N (100) consisting of all '1's.
    """
    start_time = time.time()
    
    # The OneMax objective is maximized by setting all bits to 1.
    # This is the unique global optimum.
    result = "1" * N
    
    # Explicit deadline guard: check if we've exceeded the safe time limit.
    # For this O(1) construction, this will never trigger, but the guard is
    # present per the directive.
    elapsed = time.time() - start_time
    if elapsed > (DEADLINE_SECONDS - SAFETY_MARGIN):
        # In a more complex search, we would return the best result found so far.
        # Here, the result is already optimal and computed instantly.
        pass
    
    return result
# EVOLVE-BLOCK-END

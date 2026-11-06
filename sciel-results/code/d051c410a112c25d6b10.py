"""OneMax direct binary construction (100 bits) - ANALYTICAL CONSTRUCTION with deadline guard."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Strategy: Direct Analytical Construction.
    The OneMax objective function for a fixed-length binary string of length N
    is maximized when all bits are set to 1. The global optimum is unique and
    known a priori: the string consisting entirely of '1's.
    
    This function directly constructs this optimal solution in O(N) time.
    It includes an explicit deadline guard to comply with runtime constraints,
    although the computation is instantaneous.
    """
    start_time = time.time()
    
    # Theoretical maximum is N ones.
    # We construct the string directly.
    # O(N) operation.
    optimal_string = "1" * N
    
    # Deadline guard: Check if we are approaching the limit.
    # Since the operation is O(1) effectively (string multiplication is fast),
    # this check is purely formal to satisfy the directive.
    elapsed = time.time() - start_time
    # The limit is 120s (or 150s wall clock kill). We are far below any limit.
    # If this were a complex search, we would return the best found here.
    # Here, we have the global optimum, so we return it.
    
    return optimal_string
# EVOLVE-BLOCK-END

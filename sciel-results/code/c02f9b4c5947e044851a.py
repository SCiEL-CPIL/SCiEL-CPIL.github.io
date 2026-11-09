"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100
DEADLINE_MARGIN = 5.0  # seconds
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The task is OneMax: maximize the number of '1' bits in a binary string of length N.
    The direct, optimal construction is to set all bits to '1'.
    This achieves the maximum possible score of N (100 ones).
    
    The solution is deterministic and O(1). An explicit deadline guard is included
    per the compute budget directive, though no search loop requires it.
    """
    # Explicit deadline guard
    # Check if we are close to the timeout (150s limit, margin 5s)
    if time.time() - START_TIME > 145.0:
        # Return best result found (which is the optimal string)
        return "1" * N
        
    # The optimal solution is deterministic and O(1)
    return "1" * N
# EVOLVE-BLOCK-END

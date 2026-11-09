"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100
DEADLINE_MARGIN = 5.0  # seconds
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    # Construct the global optimum: a 100-bit binary string of all ones
    # This maximizes the number of ones (the LMX OneMax objective)
    
    # Explicit deadline guard
    # Check if we are close to the timeout (150s limit, margin 5s)
    if time.time() - START_TIME > 145.0:
        # Return best result found (which is the optimal string)
        return "1" * N
        
    # The optimal solution is deterministic and O(1)
    return "1" * N
# EVOLVE-BLOCK-END

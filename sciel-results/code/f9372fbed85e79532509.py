"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""

import time

N = 100
START_TIME = time.time()
TIME_LIMIT = 150.0

# EVOLVE-BLOCK-START
def construct():
    """
    Directly constructs the optimal 100-bit binary string for OneMax.
    The optimal solution is a string of length N consisting entirely of '1's.
    
    Strategy: 
    1. Check deadline guard.
    2. Return a string of '1's of length N.
    This is the global optimum. No search or evolution is required.
    """
    # Deadline guard: If we are close to the time limit, return the known optimal solution immediately.
    # Although this operation is instantaneous, the guard satisfies the contract requirement.
    if time.time() - START_TIME > TIME_LIMIT - 1.0:
        return "1" * N

    # The global optimum for OneMax with N bits is a string of N ones.
    # Direct construction is the most efficient and reliable method.
    return "1" * N
# EVOLVE-BLOCK-END

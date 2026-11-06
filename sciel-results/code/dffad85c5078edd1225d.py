import time

N = 100
START_TIME = time.time()
DEADLINE = 120.0  # Safety margin well below 150s limit

# EVOLVE-BLOCK-START
def construct():
    """
    OneMax: maximize the number of 1s in a binary string of length N.
    
    Direct construction of the known optimal solution: all bits set to 1.
    """
    if time.time() - START_TIME > DEADLINE:
        return "0" * N
    
    return "1" * N
# EVOLVE-BLOCK-END

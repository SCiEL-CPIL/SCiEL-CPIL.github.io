"""OneMax direct binary construction achieving global optimum."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string that maximizes the number of ones.
    
    Directly constructs the global optimum string of N '1's.
    Includes a deadline guard for safety, though construction is instantaneous.
    
    Returns:
        str: A binary string of length 100 consisting entirely of '1's.
    """
    t0 = time.time()
    deadline = t0 + 118.0  # 2-second safety margin before 120s wall-clock limit

    # Direct construction of the optimal string
    # This is the global optimum for OneMax
    result = '1' * N

    # Deadline guard: if we somehow exceed time (unlikely for this operation),
    # return the result which is already optimal.
    if time.time() - t0 > deadline:
        return result

    return result
# EVOLVE-BLOCK-END

import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Direct construction of the optimal 100-bit binary string.
    
    The objective is to maximize the number of '1's in a binary string of length N.
    The maximum possible number of '1's is N, achieved by the string consisting of
    all '1's. This is the unique global optimum for the OneMax problem.
    
    This function returns this optimal string directly, ensuring maximum score
    with minimal computational overhead.
    """
    # The optimal solution is a string of N '1's
    return "1" * N
# EVOLVE-BLOCK-END

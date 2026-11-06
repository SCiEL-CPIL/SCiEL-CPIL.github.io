"""
OneMax (100 bits) direct binary construction.
The objective is to maximize the count of '1' characters in a 100-bit string.
The global optimum is the all-ones string, which yields a score of 100.
"""
import time

N = 100
DEADLINE_SECONDS = 150.0

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the optimal 100-bit binary string for the OneMax problem.
    
    The OneMax objective is f(x) = sum(x_i) for x_i in {0, 1}.
    The unique global maximizer is x_i = 1 for all i.
    
    This function directly constructs this optimal string.
    It includes a deadline guard to ensure compliance with evaluation timeouts,
    although the construction is O(N) and instantaneous.
    """
    start_time = time.time()
    
    # Direct construction of the global optimum
    # This is the highest possible score (100 ones)
    result = "1" * N
    
    # Deadline guard: 
    # The operation is O(N) with N=100, which is negligible (< 1 microsecond).
    # This check is a formality to satisfy the requirement for an explicit deadline guard.
    # If this were a more complex search, we would return the best result found 
    # if time.time() - start_time > DEADLINE_SECONDS - 0.1
    if time.time() - start_time > DEADLINE_SECONDS - 0.1:
         # This branch is effectively unreachable for N=100 direct construction
         return result

    # "Polish" pass: Verify the result is valid and optimal.
    # In a stochastic search, this would be a local improvement step.
    # Here, we simply assert the property that makes it optimal.
    # len(result) must be N and all characters must be '1'.
    # Since we constructed it as "1" * N, this is guaranteed.
    
    return result
# EVOLVE-BLOCK-END

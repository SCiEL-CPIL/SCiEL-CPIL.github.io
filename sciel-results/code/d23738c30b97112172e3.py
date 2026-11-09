"""
Program for OneMax direct binary construction (100 bits).

This program implements the direct construction of the global optimum for the OneMax problem.
It follows a 'model-and-optimize' framework where the model is the linear objective function
maximizing the count of ones, and the optimization step identifies the unique global maximum.
"""
import time

N = 100
DEADLINE = 150.0
SAFETY_MARGIN = 1.0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The task is OneMax: maximize the number of '1' bits in a binary string of length N.
    The objective function is f(x) = sum(x_i) for x_i in {0, 1}.
    The gradient of the objective with respect to each bit is +1, implying the unique
    global maximum is the vector of all 1s.
    
    Implementation:
    1. Directly construct the candidate string of N '1's.
    2. Perform a trivial verification of the objective value.
    3. Check against the wall-clock deadline to ensure compliance with execution constraints.
    
    Returns:
        str: A binary string of length N consisting entirely of '1's.
    """
    start_time = time.time()
    
    # Theoretical optimum for OneMax is all bits set to 1.
    # We construct this directly using efficient string multiplication.
    candidate = '1' * N
    
    # Verification step: Calculate the objective value (number of ones).
    # This is O(N) but N is small (100), so it's negligible.
    ones_count = candidate.count('1')
    
    # Deadline guard: Ensure we do not exceed the allowed wall-clock time.
    # Although the construction is instantaneous, this guard satisfies the
    # requirement to track elapsed time and return the best result found
    # with a safety margin before the limit.
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # In a more complex search, we would return the best candidate found so far.
        # Here, the candidate is already optimal, so we just proceed to return.
        pass

    return candidate
# EVOLVE-BLOCK-END

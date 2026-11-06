import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    The global optimum for the OneMax problem is a string of all '1's.
    """
    start_time = time.time()
    best_result = "1" * N
    # Since the solution is found immediately, we return it.
    # In a search context, we would check time here.
    if time.time() - start_time > 119:
        return best_result
    return best_result
# EVOLVE-BLOCK-END

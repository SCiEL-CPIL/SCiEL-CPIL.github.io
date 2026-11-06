"""OneMax direct binary construction (100 bits).

The objective is to maximize the number of ones in a 100-bit binary string.
The optimal solution is a string of all ones. Since the task is deterministic
and the optimum is trivially known, we return the optimal string directly.
"""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """Return a 100-bit binary string with all ones (optimal for OneMax)."""
    return "1" * N
# EVOLVE-BLOCK-END

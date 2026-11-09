"""Program for OneMax direct binary construction (100 bits)."""

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    The task is OneMax: maximize the number of '1' bits in a binary string of length N.
    The theoretical maximum is N (100) ones.
    
    This function directly constructs the global optimum, which is a string of 
    '1's of length N. This approach is optimal as it requires no search, 
    no iteration, and guarantees the maximum possible score instantly.
    """
    return '1' * N
# EVOLVE-BLOCK-END

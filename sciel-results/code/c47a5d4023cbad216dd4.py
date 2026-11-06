"""Program for OneMax direct binary construction (100 bits)."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Initialize a NumPy boolean array of length N with all True values.
    # This leverages vectorized memory representation.
    bits = np.ones(N, dtype=bool)
    
    # Convert the boolean array to a string of '1's using NumPy's character module
    # for a more vectorized approach compared to Python loops/generators.
    # First, cast to a small integer type for efficiency.
    int_bits = bits.astype(np.int8)
    
    # Use np.char.mod to format each integer as a string in parallel.
    # This returns a NumPy array of strings.
    str_bits = np.char.mod('%d', int_bits)
    
    # Join the array of single-character strings into the final binary string.
    return ''.join(str_bits)
# EVOLVE-BLOCK-END

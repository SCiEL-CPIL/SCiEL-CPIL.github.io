"""Program for OneMax direct binary construction (100 bits)."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Initialize a NumPy boolean array of length N with all True values.
    # Each True represents a '1' in the binary string.
    bits = np.ones(N, dtype=bool)
    # Convert boolean array to integer array (0/1)
    int_bits = bits.astype(int)
    # Convert integer array to Unicode string array of length 1 ('0' or '1')
    str_bits = int_bits.astype('U1')
    # Join the array elements into a single string
    return ''.join(str_bits)
# EVOLVE-BLOCK-END

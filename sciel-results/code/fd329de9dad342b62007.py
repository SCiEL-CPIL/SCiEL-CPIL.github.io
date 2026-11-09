"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def construct():
    """Construct a 100-bit binary string maximizing ones using NumPy vectorization."""
    # Initialize a NumPy array of uint8 with all ones
    bits = np.full(N, 1, dtype=np.uint8)
    # Cast the binary array to a sequence of string characters using vectorized formatting
    char_sequence = np.char.mod('1', bits)
    # Concatenate the character array into the final binary string
    return ''.join(char_sequence)
# EVOLVE-BLOCK-END

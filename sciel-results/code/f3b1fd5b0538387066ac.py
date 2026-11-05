"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Models the construction as a deterministic bitwise OR operation between 
    a zero-initialized integer and a mask of 2^100 - 1.
    """
    # Zero-initialized integer
    state = 0
    
    # Mask with all 100 bits set to 1
    mask = (1 << N) - 1
    
    # Deterministic bitwise OR to set all bits to 1
    state = state | mask
    
    # Convert to binary string, stripping '0b' prefix
    return bin(state)[2:]
# EVOLVE-BLOCK-END

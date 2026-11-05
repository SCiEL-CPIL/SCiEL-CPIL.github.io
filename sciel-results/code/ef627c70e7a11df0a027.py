"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Models the construction as a deterministic bitwise OR operation 
    between a zero-initialized integer and a mask of 2^100 - 1.
    """
    # Initialize with zero
    val = 0
    
    # Create the mask of 2^N - 1 (all N bits set to 1)
    mask = (1 << N) - 1
    
    # Perform bitwise OR to set all bits
    val = val | mask
    
    # Convert to binary string and strip the '0b' prefix
    binary_string = bin(val)[2:]
    
    # Ensure strict length N
    if len(binary_string) < N:
        binary_string = binary_string.zfill(N)
    elif len(binary_string) > N:
        binary_string = binary_string[-N:]
        
    return binary_string
# EVOLVE-BLOCK-END

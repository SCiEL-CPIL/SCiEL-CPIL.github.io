"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    
    Since the objective is OneMax (maximize number of 1s) and the only constraint
    is that the output is a binary string of length N=100, the optimal solution
    is simply a string of all 1s.
    
    Uses Python's arbitrary-precision integer arithmetic:
    - Initialize a zero integer
    - Create a mask of 2^N - 1 (all N bits set to 1)
    - Perform bitwise OR to set all bits
    - Convert to binary string using bin()
    - Strip the '0b' prefix
    
    Includes an explicit deadline guard to ensure compliance with time limits,
    although the operation is constant time.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin of 5 seconds before 150s limit
    
    # Model construction as bitwise OR between zero and mask
    val = 0
    mask = (1 << N) - 1
    val = val | mask
    
    # Check deadline (will almost always pass, but required by directive)
    if time.time() > deadline:
        # Fallback: return a valid string if somehow we hit the deadline
        # In this constant-time case, this branch is effectively unreachable
        # but ensures we return *something* valid if the environment is extremely slow
        return "1" * N
        
    # Convert to binary string and strip the '0b' prefix
    binary_string = bin(val)[2:]
    
    # Ensure the string is exactly N characters long
    if len(binary_string) < N:
        binary_string = binary_string.zfill(N)
    elif len(binary_string) > N:
        binary_string = binary_string[-N:]
    
    return binary_string
# EVOLVE-BLOCK-END

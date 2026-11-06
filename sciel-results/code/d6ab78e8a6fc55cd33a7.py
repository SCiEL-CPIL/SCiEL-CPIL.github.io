"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a binary error-correcting code of length 18 with minimum distance 6.
    Uses a systematic search with bit-parallel conflict masks to find a large code.
    Improved with Gray-code ordering and optimized mask application.
    """
    N = 18
    MIN_DIST = 6
    
    size = 1 << N
    forbidden = np.zeros(size, dtype=bool)
    
    # Generate masks of weight < MIN_DIST (0 to 5)
    # This is done once.
    masks_list = [i for i in range(size) if i.bit_count() < MIN_DIST]
    masks = np.array(masks_list, dtype=np.uint32)
    
    # Function to mark forbidden neighbors
    def mark_forbidden(c):
        # c is an integer in [0, size)
        # masks are in [0, size)
        # XOR result is in [0, size)
        neighbors = c ^ masks
        forbidden[neighbors] = True
        
    # Start with 0
    code = [0]
    mark_forbidden(0)
    
    # Iterate through candidates in Gray code order
    # Gray code ensures that consecutive candidates differ by only 1 bit.
    # This can sometimes lead to better packing in greedy heuristics by exploring
    # the space more uniformly than linear counting.
    for i in range(1, size):
        c = i ^ (i >> 1)
        if not forbidden[c]:
            code.append(c)
            mark_forbidden(c)
            
    return code

# EVOLVE-BLOCK-END

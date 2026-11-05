"""
OneMax direct binary construction (100 bits)
Constructs a 100-bit binary string maximizing ones.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a binary string of length N with all bits set to 1.
    This is the global optimum for the OneMax objective.
    Uses a set-theoretic union approach to derive the indices for the ones.
    Includes a time deadline guard to ensure safe termination.
    """
    start_time = time.time()
    deadline = 120.0  # Safety margin below the 150s hard limit
    
    # Strategy: Model the task as a set-theoretic union operation.
    # We create a set of indices for the 100 bits.
    # The 'union with itself' is a no-op logically, but establishes the domain.
    # Then we map each index to '1'.
    
    # Create the set of valid indices for the N bits
    indices_set = set(range(N))
    
    # Take the union with itself (redundant, but follows the directive's structural hint)
    # This ensures we are working with a set of unique indices covering the full range.
    combined_indices = indices_set.union(indices_set)
    
    # Construct the string by mapping each index to '1'.
    # Since we want to maximize ones, every position in the combined set becomes '1'.
    # We sort the indices to ensure a deterministic order (0 to N-1).
    result = ''.join('1' for _ in sorted(combined_indices))
    
    # Check time to satisfy the guard requirement
    if time.time() - start_time > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END

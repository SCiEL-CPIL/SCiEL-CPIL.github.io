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
    
    Strategy: Model the task as a set-theoretic union operation.
    We take the union of a set of N unique integer indices with itself.
    The cardinality of this union is N. We then map each element in the sorted
    union to the character '1' to construct the string.
    
    This derives the all-ones solution from set cardinality rather than
    direct string repetition or bitwise masks, while maintaining the exact
    same output and performance characteristics (O(N) time, O(N) space).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin of 5 seconds before 150s limit
    
    # Structural change: Use set-theoretic union to generate the indices.
    # Create a set of unique indices 0 to N-1.
    indices_set = set(range(N))
    
    # Perform the union operation of the set with itself.
    # This results in the same set, confirming the cardinality is N.
    union_set = indices_set.union(indices_set)
    
    # Verify cardinality (optional safety check, low cost)
    # if len(union_set) != N:
    #     # Fallback to simple construction if something unexpected happens
    #     return "1" * N
    
    # Map each index in the sorted union to the character '1'.
    # Sorting ensures a deterministic order, though the content is identical for all.
    # Using a list comprehension for efficiency.
    result = ''.join('1' for _ in sorted(union_set))
    
    # Ensure the length is exactly N (pad with zeros if necessary, though not needed here)
    if len(result) < N:
        result = result.zfill(N)
    elif len(result) > N:
        result = result[-N:]
        
    # Explicit deadline guard as per directive
    if time.time() > deadline:
        return result
        
    return result
# EVOLVE-BLOCK-END

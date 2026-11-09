"""OneMax 100-bit binary construction: maximize number of ones."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing the number of ones.
    The optimal solution is the string of all ones.
    Returns:
        str: A string of length N consisting entirely of '1's.
    """
    # Deadline guard (not strictly necessary for O(N) operation but included per directive)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # The global optimum for OneMax is all ones.
    # No search or evolution is required.
    result = "1" * N
    
    # Check time (will always pass for this simple operation)
    if time.time() > deadline:
        # Fallback should not be needed, but returns best known (which is the same)
        return result
        
    return result
# EVOLVE-BLOCK-END

"""
OneMax 100-bit direct construction.
Returns the global optimum: a 100-character string of '1's.
"""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return the optimal 100-bit string for OneMax: all bits set to 1.
    """
    deadline_start = time.monotonic()
    
    # The global optimum for OneMax is trivially all ones.
    best_candidate = "1" * 100
    
    # Explicit deadline guard: check elapsed time against a safety margin.
    # The task limit is 150s. We use a 5s margin, so 145s.
    # This is an O(1) operation, so this guard is purely defensive per directive.
    if time.monotonic() - deadline_start > 145:
        # If we were to somehow be near the deadline (impossible for O(1)),
        # we still return the best candidate, which is already optimal.
        return best_candidate
    
    return best_candidate
# EVOLVE-BLOCK-END

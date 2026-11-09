"""OneMax direct binary construction (100 bits)."""
import time
import math

N = 100
# Explicit deadline guard: keep a safety margin before the 150 s wall-clock kill.
_DEADLINE_S = 140.0

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The optimal solution for OneMax is a string of all '1's.
    We directly return the global optimum with a deadline guard.
    """
    t0 = time.monotonic()
    
    # The objective is deterministic and known: maximize count of '1's.
    # The global optimum is a string of N '1's.
    # This is an O(N) operation with N=100, which is essentially instantaneous.
    
    # Construct the optimal string directly.
    # Using a loop allows for a theoretical check against the deadline,
    # though it will never trigger for N=100.
    bits = []
    for i in range(N):
        # If we were to do more complex work, we'd check time here.
        # For this specific task, we just append '1'.
        bits.append('1')
        
        # Deadline guard (never expected to trigger)
        if time.monotonic() - t0 > _DEADLINE_S:
            # Return best result found so far (partial ones)
            return ''.join(bits)
            
    return ''.join(bits)
# EVOLVE-BLOCK-END

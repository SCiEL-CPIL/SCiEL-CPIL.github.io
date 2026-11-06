"""OneMax direct binary construction (100 bits) - global optimum via big-int representation."""
import time

N = 100
DEADLINE_SECONDS = 115  # Safety margin before 120s limit

# Pre-compute the optimal solution using big-integer arithmetic
# (1 << 100) - 1 gives a 100-bit integer with all bits set to 1
_OPTIMAL_INT = (1 << N) - 1
_OPTIMAL_STR = bin(_OPTIMAL_INT)[2:]  # Convert to binary string without '0b' prefix

# Fallback in case of any conversion issue
if len(_OPTIMAL_STR) != N:
    _OPTIMAL_STR = "1" * N

# EVOLVE-BLOCK-START
def construct():
    """Return a 100-bit binary string with all ones (global optimum)."""
    t0 = time.time()
    
    # Deadline guard: if we somehow exceed our safety margin, return immediately
    # In practice, this function returns in constant time
    if time.time() - t0 > DEADLINE_SECONDS:
        return _OPTIMAL_STR
    
    # Return the pre-computed optimal string
    return _OPTIMAL_STR
# EVOLVE-BLOCK-END

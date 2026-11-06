import time
import numpy as np

N = 100
DEADLINE_SECONDS = 150.0

# Pre-allocate the NumPy array of ones at module load time
_OPT_ARRAY = np.ones(N, dtype=np.uint8)
# Pre-compute the string representation if possible?
# The directive says "converting it to a string only upon invocation".
# So we should NOT pre-compute the string, but convert it in construct().

# EVOLVE-BLOCK-START
def construct():
    t0 = time.monotonic()
    
    # Convert the pre-allocated NumPy array to a binary string
    # Using astype to 'S1' and then decoding is a vectorized operation
    opt_str = _OPT_ARRAY.astype('S1').tobytes().decode('ascii')
    
    # Safety margin: return fallback if we somehow exceed the deadline
    if (time.monotonic() - t0) > (DEADLINE_SECONDS - 0.5):
        return "0" * N
    
    return opt_str
# EVOLVE-BLOCK-END

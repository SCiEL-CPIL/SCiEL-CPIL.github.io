"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    The global optimum for OneMax is all '1's.
    Uses bytearray with ASCII '1' (0x31) for efficient construction.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    # Create a mutable buffer of 100 bytes, each set to ASCII '1' (0x31)
    buf = bytearray(b'\x31' * N)
    # Convert to UTF-8 string
    return buf.decode('utf-8')
# EVOLVE-BLOCK-END

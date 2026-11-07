"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

# Strategy: Use exact counts for all 400 items using uint16 (2 bytes each).
# Total size: 400 * 2 = 800 bytes. Fits in 1024 bytes.
# This eliminates all collision noise and overflow issues (max count 1024 < 65535).
# Error will be 0 for all streams.

def update(state, item):
    b = bytearray(state)
    offset = item * 2
    # Read current uint16 value (little-endian)
    val = b[offset] + (b[offset + 1] << 8)
    # Increment
    val += 1
    # Write back
    b[offset] = val & 0xFF
    b[offset + 1] = (val >> 8) & 0xFF
    return bytes(b)

def query(state, item):
    offset = item * 2
    # Read uint16 value (little-endian)
    val = state[offset] + (state[offset + 1] << 8)
    return val

# EVOLVE-BLOCK-END

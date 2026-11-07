"""Rewritten program: direct-addressing 16-bit exact counters for 400 items."""

# EVOLVE-BLOCK-START

def update(state, item):
    """Increment the 16-bit little-endian counter for 'item' in-place.
    
    State layout: first 800 bytes hold 400 uint16 counters (little-endian).
    Remaining 224 bytes are unused (kept zero).
    """
    # Convert to bytearray for in-place modification
    ba = bytearray(state)
    offset = item * 2
    # Read current 16-bit LE value
    lo = ba[offset]
    hi = ba[offset + 1]
    val = lo | (hi << 8)
    # Increment
    val += 1
    # Clamp at 65535 (shouldn't be needed for 1024 items, but safe)
    if val > 65535:
        val = 65535
    # Write back
    ba[offset] = val & 0xFF
    ba[offset + 1] = (val >> 8) & 0xFF
    return bytes(ba)

def query(state, item):
    """Return the exact count for 'item' from the 16-bit LE counter."""
    offset = item * 2
    lo = state[offset]
    hi = state[offset + 1]
    return lo | (hi << 8)

# EVOLVE-BLOCK-END

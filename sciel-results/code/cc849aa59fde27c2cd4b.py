"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def update(state, item):
    """
    Update the state by incrementing the count for the given item.
    The state is a 1024-byte buffer where each item's count is stored as a single byte
    at a deterministic index derived from the item ID.
    """
    # Ensure state is a mutable bytearray
    b = bytearray(state)
    
    # Deterministic index derived from item ID
    # We have 400 possible items (0..399), so we can store each count in 1 byte
    # at position item * 2 (using 2 bytes per item to be safe, but since max count
    # is 1024, we need 2 bytes. However, the problem says state is 1024 bytes.
    # Let's reconsider: 400 items, each count up to 1024. 1024 fits in 2 bytes.
    # 400 * 2 = 800 bytes, which is under 1024.
    # But the task says "initial state is 1024 zero bytes". So we can use 2 bytes per item.
    # Let's use 2 bytes per item to handle counts up to 65535, which is safe.
    idx = item * 2
    
    # Read current count (little-endian uint16)
    val = b[idx] | (b[idx + 1] << 8)
    
    # Increment
    val += 1
    
    # Write back
    b[idx] = val & 0xFF
    b[idx + 1] = (val >> 8) & 0xFF
    
    return bytes(b)

def query(state, item):
    """
    Query the state for the frequency estimate of the given item.
    Returns the nonnegative frequency estimate.
    """
    # Deterministic index derived from item ID
    idx = item * 2
    
    # Read current count (little-endian uint16)
    val = state[idx] | (state[idx + 1] << 8)
    
    return val
# EVOLVE-BLOCK-END

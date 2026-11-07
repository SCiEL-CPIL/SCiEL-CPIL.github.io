def update(state, item):
    """
    Updates the state by incrementing the count for the given item.
    State layout:
    - Bytes 0..799: 400 little-endian uint16 counters for items 0..399.
    - Bytes 800..1023: Unused (zero).
    """
    # Convert to bytearray for mutable access
    b = bytearray(state)
    
    # Calculate offset for the item's counter
    # Items are 0..399, each counter is 2 bytes (uint16)
    offset = item * 2
    
    # Bounds check to prevent crash if item is out of expected range (though contract says 0..399)
    if offset < 0 or offset + 2 > 1024:
        return state

    # Read current uint16 value (little-endian)
    current_val = b[offset] | (b[offset + 1] << 8)
    
    # Increment
    new_val = current_val + 1
    
    # Handle potential overflow of uint16 (max 65535). 
    # Max count is 1024, so this won't happen, but safe to mask.
    new_val &= 0xFFFF
    
    # Write back uint16 value (little-endian)
    b[offset] = new_val & 0xFF
    b[offset + 1] = (new_val >> 8) & 0xFF
    
    return bytes(b)

def query(state, item):
    """
    Queries the state to get the frequency estimate for the given item.
    Returns the exact count stored in the direct counter.
    """
    offset = item * 2
    
    # Bounds check
    if offset < 0 or offset + 2 > 1024:
        return 0

    # Read uint16 value (little-endian)
    val = state[offset] | (state[offset + 1] << 8)
    
    return val

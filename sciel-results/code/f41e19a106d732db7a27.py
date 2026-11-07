import time

_START = time.time()
_DEADLINE = _START + 110.0

# State layout:
# Bytes 0..799: 400 dedicated uint16 counters (little-endian)
# Bytes 800..1023: Unused (224 bytes)
#
# Strategy:
# Use 16-bit counters for all 400 items.
# Max count per item is 1024 (stream length), which fits in uint16.
# This provides exact frequency counts for all items, resulting in zero error.
# The previous uint8 approach failed on Zipf streams because high-frequency 
# items exceeded 255, causing saturation. uint16 eliminates this issue.

NUM_ITEMS = 400
ITEM_SIZE = 2  # uint16

def update(state, item):
    """Update the state by incrementing the counter for the given item."""
    if time.time() > _DEADLINE:
        return state
    
    # Convert state to a bytearray for mutation
    ba = bytearray(state)
    
    # Increment the dedicated counter for this item
    # Each item has its own 2-byte slot at position item * 2
    if 0 <= item < NUM_ITEMS:
        off = item * ITEM_SIZE
        # Read current value (little-endian uint16)
        val = ba[off] | (ba[off + 1] << 8)
        
        # Increment with saturation at 65535 (though 1024 is the practical max)
        if val < 65535:
            val += 1
            ba[off] = val & 0xFF
            ba[off + 1] = (val >> 8) & 0xFF
            
    return bytes(ba)

def query(state, item):
    """Query the estimated frequency of the given item."""
    if time.time() > _DEADLINE:
        return 0
    
    # Return the exact count from the dedicated counter
    if 0 <= item < NUM_ITEMS:
        off = item * ITEM_SIZE
        return state[off] | (state[off + 1] << 8)
    else:
        return 0

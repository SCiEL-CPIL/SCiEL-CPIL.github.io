"""
Improved program: Exact Frequency Counting.

Since the item space is small (400 items) and the memory limit is 1024 bytes,
we can store exact counts for all items using 2 bytes per item (16-bit unsigned int).
400 items * 2 bytes = 800 bytes, which fits within the 1024 byte limit.
This eliminates the estimation error inherent in Count-Min Sketches, resulting in
near-zero error and a significantly higher score (ratio of seed error to this error).
"""

# EVOLVE-BLOCK-START

def update(state, item):
    """
    Updates the state by incrementing the exact count for the given item.
    State layout:
    - Bytes 0 to 799: 400 16-bit little-endian unsigned integers, representing counts for items 0-399.
    - Bytes 800 to 1023: Padding (zeros).
    """
    # Item is in range 0..399
    if item < 0 or item >= 400:
        # If item is out of expected range, return state unchanged to avoid crash
        # though the contract says items are 0..399
        return state
    
    offset = item * 2
    
    # Extract current count
    # state is bytes, slicing returns bytes
    current_bytes = state[offset:offset+2]
    if len(current_bytes) < 2:
        # Safety check, shouldn't happen if state is 1024 bytes
        current_val = 0
    else:
        current_val = int.from_bytes(current_bytes, 'little')
        
    # Increment
    new_val = current_val + 1
    
    # Cap at max uint16 to prevent overflow issues, though 1024 items max means max count is 1024
    if new_val > 65535:
        new_val = 65535
        
    # Convert back to bytes
    new_bytes = new_val.to_bytes(2, 'little')
    
    # Construct new state
    # Slicing bytes creates a new bytes object, which is fine for 1024 bytes
    new_state = state[:offset] + new_bytes + state[offset+2:]
    
    return new_state

def query(state, item):
    """
    Returns the exact frequency estimate for the given item.
    """
    if item < 0 or item >= 400:
        return 0.0
        
    offset = item * 2
    count_bytes = state[offset:offset+2]
    
    if len(count_bytes) < 2:
        return 0.0
        
    val = int.from_bytes(count_bytes, 'little')
    return float(val)

# EVOLVE-BLOCK-END

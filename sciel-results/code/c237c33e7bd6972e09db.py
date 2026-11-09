import struct
import time

# Constants
W = 128          # width of each row
D = 4            # depth (number of rows)
PRIME = 2147483647
# Reuse the same hash parameters as the seed for consistency, but we could tune them.
# The seed uses: p in (1035493289, 775374203, 1495862053, 2015145641)
# and index = ((item * p + 31 * row) % PRIME) % W
# We will use the same structure but with W=128.

# Pre-compute hash parameters
_PS = (1035493289, 775374203, 1495862053, 2015145641)

def update(state, item):
    """
    Update the Count-Min sketch state.
    state: bytes of length 1024 (4 rows * 128 counters * 2 bytes per counter)
    item: int in 0..399
    returns: new state as bytes
    """
    # Convert state to a mutable buffer
    # We use uint16 for counters. Max count is 1024, which fits in uint16 (max 65535).
    # 4 rows * 128 counters * 2 bytes = 1024 bytes.
    
    # Unpack the state into a list of counters or modify in place
    # For speed, we can use array module or just list of ints
    # Since state is bytes, we can use struct or array
    
    # Let's use a list of 512 integers for simplicity and speed in Python
    # Actually, converting bytes to a list of 512 uint16s might be slow.
    # Let's try to operate on the bytes directly or use a bytearray.
    
    # bytearray is mutable and fast enough for this small size.
    buf = bytearray(state)
    
    # For each row, compute the index and increment the counter
    for r in range(D):
        p = _PS[r]
        # Compute index: ((item * p + 31 * r) % PRIME) % W
        idx = ((item * p + 31 * r) % PRIME) % W
        
        # Each counter is 2 bytes (uint16)
        offset = r * W * 2 + idx * 2
        
        # Read current value
        current = buf[offset] | (buf[offset + 1] << 8)
        
        # Increment (saturating at 65535, though max is 1024 so no overflow)
        new_val = current + 1
        if new_val > 65535:
            new_val = 65535
            
        # Write back
        buf[offset] = new_val & 0xFF
        buf[offset + 1] = (new_val >> 8) & 0xFF
        
    return bytes(buf)

def query(state, item):
    """
    Query the estimated frequency of an item.
    state: bytes of length 1024
    item: int in 0..399
    returns: nonnegative frequency estimate (int)
    """
    # Use the minimum of the 4 row estimates
    min_est = float('inf')
    
    for r in range(D):
        p = _PS[r]
        idx = ((item * p + 31 * r) % PRIME) % W
        
        offset = r * W * 2 + idx * 2
        
        # Read the counter value
        val = state[offset] | (state[offset + 1] << 8)
        
        if val < min_est:
            min_est = val
            
    return int(min_est)

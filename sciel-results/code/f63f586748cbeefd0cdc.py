"""
Program using exact frequency counts stored in 2 bytes per item.
State layout: 400 items * 2 bytes/item = 800 bytes used.
Remaining 224 bytes are zero padding.
Item i count is stored at bytes [2*i : 2*i + 2] as little-endian uint16.
"""

def update(state, item):
    # Convert state to mutable bytearray
    buf = bytearray(state)
    
    # Calculate offset for this item
    offset = item * 2
    
    # Read current count
    # int.from_bytes requires a bytes-like object
    current_count = int.from_bytes(buf[offset:offset+2], byteorder='little')
    
    # Increment
    new_count = current_count + 1
    
    # Write back
    buf[offset:offset+2] = new_count.to_bytes(2, byteorder='little')
    
    # Return as bytes
    return bytes(buf)

def query(state, item):
    # Calculate offset for this item
    offset = item * 2
    
    # Read count
    count = int.from_bytes(state[offset:offset+2], byteorder='little')
    
    return count

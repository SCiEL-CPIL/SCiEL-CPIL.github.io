import struct

def update(state, item):
    # State is 1024 bytes.
    # We use the first 800 bytes for exact counts of items 0..399.
    # Each count is stored as a little-endian uint16.
    # 400 items * 2 bytes/item = 800 bytes.
    
    # Ensure item is within valid range
    if item < 0 or item > 399:
        # If out of range, return state unchanged to avoid crash, 
        # though contract says items are 0..399.
        return state

    offset = item * 2
    # Unpack current count
    count = struct.unpack_from('<H', state, offset)[0]
    
    # Increment
    count += 1
    
    # Pack back
    new_state = bytearray(state)
    struct.pack_into('<H', new_state, offset, count)
    
    return bytes(new_state)

def query(state, item):
    if item < 0 or item > 399:
        return 0
        
    offset = item * 2
    count = struct.unpack_from('<H', state, offset)[0]
    return count

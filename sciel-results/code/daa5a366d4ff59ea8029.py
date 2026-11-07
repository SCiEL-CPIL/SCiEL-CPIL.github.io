import struct

# Layout:
# 400 items, each stored as a little-endian uint16 (2 bytes).
# Total bytes used: 400 * 2 = 800 bytes.
# Remaining 224 bytes are padding (zeros).
# This allows exact counting for frequencies up to 65535, which is sufficient
# for a stream of 1024 items.

ITEM_COUNT = 400
BYTES_PER_ITEM = 2
TOTAL_BYTES = 1024

def update(state, item):
    # Ensure item is within bounds, though task guarantees 0..399
    if item < 0 or item >= ITEM_COUNT:
        return state
    
    buf = bytearray(state)
    offset = item * BYTES_PER_ITEM
    
    # Unpack uint16
    val = struct.unpack_from('<H', buf, offset)[0]
    
    # Increment
    val += 1
    
    # Pack uint16
    struct.pack_into('<H', buf, offset, val)
    
    return bytes(buf)

def query(state, item):
    if item < 0 or item >= ITEM_COUNT:
        return 0
        
    offset = item * BYTES_PER_ITEM
    val = struct.unpack_from('<H', state, offset)[0]
    return val

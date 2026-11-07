import struct

def update(state, item):
    # state is bytes, item is int 0..399
    # We use 2 bytes (16 bits) per item.
    # Total items: 400. Total bytes used: 800.
    # Remaining 224 bytes are unused (zeros).
    
    b = bytearray(state)
    offset = item * 2
    
    # Unpack 16-bit little-endian unsigned int
    count = struct.unpack_from('<H', b, offset)[0]
    
    # Increment
    count += 1
    
    # Pack back
    struct.pack_into('<H', b, offset, count)
    
    return bytes(b)

def query(state, item):
    offset = item * 2
    # Unpack 16-bit little-endian unsigned int
    count = struct.unpack_from('<H', state, offset)[0]
    return count

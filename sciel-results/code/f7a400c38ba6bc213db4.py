import struct

def update(state, item):
    # state is bytes of length 1024
    # item is int in 0..399
    # We use 2 bytes per item, so 400 items take 800 bytes.
    # The rest is padding.
    
    # Convert to bytearray for mutation
    buf = bytearray(state)
    
    # Calculate offset
    offset = item * 2
    
    # Unpack the current count (unsigned short, little-endian)
    count = struct.unpack_from('<H', buf, offset)[0]
    
    # Increment
    count += 1
    
    # Pack back
    struct.pack_into('<H', buf, offset, count)
    
    return bytes(buf)

def query(state, item):
    # state is bytes of length 1024
    # item is int in 0..399
    
    offset = item * 2
    
    # Unpack the count
    return struct.unpack_from('<H', state, offset)[0]

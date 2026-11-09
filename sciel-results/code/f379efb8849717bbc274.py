import struct

def update(state, item):
    if not isinstance(state, bytes) or len(state) != 1024:
        return state
    
    buf = bytearray(state)
    offset = item * 2
    
    # Unpack current value
    current = struct.unpack_from('<H', buf, offset)[0]
    
    # Increment
    new_val = current + 1
    
    # Max count is 1024, max uint16 is 65535, so no overflow check strictly needed 
    # but good practice.
    if new_val > 65535:
        new_val = 65535
        
    struct.pack_into('<H', buf, offset, new_val)
    
    return bytes(buf)

def query(state, item):
    if not isinstance(state, bytes) or len(state) != 1024:
        return 0.0
        
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return float(val)

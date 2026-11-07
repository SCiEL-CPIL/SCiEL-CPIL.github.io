import struct

def update(state, item):
    """
    Updates the state by incrementing the count for the given item.
    State is a byte string of length 1024.
    We store exact counts for items 0..399 in the first 800 bytes (2 bytes per item).
    The remaining 224 bytes are preserved as-is.
    """
    # Convert to mutable buffer
    buf = bytearray(state)
    
    # Calculate offset for the item's count (2 bytes per item)
    offset = item * 2
    
    # Unpack the current count (little-endian unsigned short)
    # If the state was initialized to zeros, this will be 0 initially.
    current_count = struct.unpack_from('<H', buf, offset)[0]
    
    # Increment the count
    new_count = current_count + 1
    
    # Pack the new count back into the buffer
    struct.pack_into('<H', buf, offset, new_count)
    
    # Return the state as bytes
    return bytes(buf)

def query(state, item):
    """
    Queries the state for the estimated frequency of the given item.
    Returns the exact count.
    """
    # Calculate offset for the item's count
    offset = item * 2
    
    # Unpack the count (little-endian unsigned short)
    count = struct.unpack_from('<H', state, offset)[0]
    
    return count

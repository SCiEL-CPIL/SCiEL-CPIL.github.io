import struct

def update(state, item):
    """
    Updates the state by incrementing the exact count for the given item.
    State layout: 400 uint16 counters, little-endian, stored in bytes 0-799.
    Bytes 800-1023 are padding.
    """
    buf = bytearray(state)
    offset = item * 2
    # Unpack uint16
    val = struct.unpack_from('<H', buf, offset)[0]
    # Increment
    val += 1
    # Pack back
    struct.pack_into('<H', buf, offset, val)
    return bytes(buf)

def query(state, item):
    """
    Returns the exact count for the given item.
    """
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return val

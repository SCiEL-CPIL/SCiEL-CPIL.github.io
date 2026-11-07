import struct

def update(state, item):
    """
    Updates the exact frequency counter for 'item' in the state.
    State layout: 400 little-endian uint16 counters (one per item 0..399) + padding.
    Total: 800 bytes used, 1024 bytes total.
    """
    ba = bytearray(state)
    offset = item * 2
    # Read current count
    count = struct.unpack_from('<H', ba, offset)[0]
    # Increment
    count += 1
    # Write back
    struct.pack_into('<H', ba, offset, count)
    return bytes(ba)

def query(state, item):
    """
    Queries the exact frequency of 'item'.
    """
    offset = item * 2
    return struct.unpack_from('<H', state, offset)[0]

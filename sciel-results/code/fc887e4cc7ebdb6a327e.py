import struct

def update(state, item):
    """
    Updates the state by incrementing the count for the given item.
    Uses a fixed 400-entry lookup table where each entry is a uint16.
    Total size used: 400 * 2 = 800 bytes.
    Remaining 224 bytes are unused (zeros).
    """
    offset = item * 2
    # Unpack the current 16-bit unsigned integer at the offset
    current_val = struct.unpack_from('<H', state, offset)[0]
    # Increment the count
    new_val = current_val + 1
    # Pack the new value
    packed_val = struct.pack('<H', new_val)
    # Construct the new state by replacing the 2 bytes at the offset
    return state[:offset] + packed_val + state[offset + 2:]

def query(state, item):
    """
    Queries the state for the frequency estimate of the given item.
    Returns the exact count stored in the uint16 entry.
    """
    offset = item * 2
    # Unpack the 16-bit unsigned integer
    val = struct.unpack_from('<H', state, offset)[0]
    return val

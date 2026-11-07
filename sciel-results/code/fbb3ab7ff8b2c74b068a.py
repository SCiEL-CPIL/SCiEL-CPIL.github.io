"""Exact frequency sketch using 1024-byte state for 400 items.
Stores uint16 counters at byte offsets 2*i for item i (0..399).
Max count 1024 fits comfortably in uint16 (max 65535).
"""

import struct

def update(state, item):
    """
    Increment the count for 'item' in the state.
    state: bytes of length 1024
    item: int in 0..399
    Returns new state as bytes of length 1024.
    """
    buf = bytearray(state)
    idx = item * 2
    val = struct.unpack_from('<H', buf, idx)[0]
    struct.pack_into('<H', buf, idx, val + 1)
    return bytes(buf)

def query(state, item):
    """
    Return the exact frequency estimate for 'item'.
    state: bytes of length 1024
    item: int in 0..399
    Returns nonnegative int.
    """
    idx = item * 2
    return struct.unpack_from('<H', state, idx)[0]

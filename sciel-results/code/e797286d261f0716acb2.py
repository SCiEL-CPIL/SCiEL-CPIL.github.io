"""Hybrid sketch: exact uint8 counters for items 0-127, 4-row uint8 sketch for 128-399."""
import struct

# Hash parameters for the sketch (same as seed)
_P = (1035493289, 775374203, 1495862053, 2015145641)
_M = 2147483647
_W = 224  # width of each sketch row for the tail (items 128-399)

def _sketch_index(item, row):
    """Compute index into the 224-wide sketch for item >= 128."""
    return ((item * _P[row] + 31 * row) % _M) % _W

def update(state, item):
    """Update the hybrid sketch state with one item."""
    state = bytearray(state)
    if item < 128:
        # Exact counter in first 128 bytes (uint8)
        if state[item] < 255:
            state[item] += 1
    else:
        # Sketch for items 128-399: 4 rows of 224 uint8 counters
        # Offset in state: 128 + row * 224
        for row in range(4):
            idx = _sketch_index(item, row)
            pos = 128 + row * _W + idx
            if state[pos] < 255:
                state[pos] += 1
    return bytes(state)

def query(state, item):
    """Query the hybrid sketch state for an item's frequency estimate."""
    if item < 128:
        # Exact count
        return int(state[item])
    else:
        # Minimum over 4 sketch rows
        min_count = 255
        for row in range(4):
            idx = _sketch_index(item, row)
            pos = 128 + row * _W + idx
            val = int(state[pos])
            if val < min_count:
                min_count = val
        return min_count

"""Memory-bounded streaming frequency sketches: 16-bit CM sketch."""
import struct

_ROWS = 4
_SLOTS = 128
_TOTAL_SIZE = 1024

# Hash primes for the CM sketch
_PRIMES = (1035493289, 775374203, 1495862053, 2015145641)


def _sketch_index(item, row):
    """Compute the column index for a given item and row in the CM sketch."""
    p = _PRIMES[row]
    return ((item * p + 31 * row) % 2147483647) % _SLOTS


def update(state, item):
    """Update the sketch state with a new item."""
    if not isinstance(state, bytes) or len(state) != _TOTAL_SIZE:
        return state

    state = bytearray(state)

    for row in range(_ROWS):
        col = _sketch_index(item, row)
        byte_offset = row * _SLOTS * 2 + col * 2
        # Read uint16 little-endian
        val = state[byte_offset] | (state[byte_offset + 1] << 8)
        if val < 65535:
            val += 1
        state[byte_offset] = val & 0xFF
        state[byte_offset + 1] = (val >> 8) & 0xFF

    return bytes(state)


def query(state, item):
    """Query the estimated frequency of an item."""
    if not isinstance(state, bytes) or len(state) != _TOTAL_SIZE:
        return 0

    min_val = 65535
    for row in range(_ROWS):
        col = _sketch_index(item, row)
        byte_offset = row * _SLOTS * 2 + col * 2
        val = state[byte_offset] | (state[byte_offset + 1] << 8)
        if val < min_val:
            min_val = val
    return float(min_val)

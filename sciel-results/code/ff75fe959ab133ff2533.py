"""Memory-bounded streaming frequency sketches: hybrid exact + CM sketch."""
import struct
import time

# State layout (1024 bytes total):
#   Bytes 0-127:   exact uint8 counters for items 0..127
#   Bytes 128-1023: 4 rows x 224 uint8 counters for items 128..399
#
# For items 0..127: exact counting (zero error, capped at 255)
# For items 128..399: 4-row Count-Min sketch with 224-wide uint8 counters
#   Hash: ((item * p + 31 * row) % 2147483647) % 224
#   Query: min of 4 rows (capped at 255)

_ROWS = 4
_SKETCH_WIDTH = 224  # 4 * 224 = 896 bytes
_EXACT_ITEMS = 128   # items 0..127 get exact counters
_TOTAL_SIZE = 1024
_SKETCH_OFFSET = _EXACT_ITEMS  # byte offset where sketch starts (128)

_PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
_M = 2147483647


def _sketch_index(item, row):
    """Compute the column index for a given item and row in the CM sketch."""
    p = _PRIMES[row]
    return ((item * p + 31 * row) % _M) % _SKETCH_WIDTH


def update(state, item):
    """Update the sketch state with a new item."""
    if not isinstance(state, bytes) or len(state) != _TOTAL_SIZE:
        return state

    buf = bytearray(state)

    if item < _EXACT_ITEMS:
        # Exact counting for items 0..127
        idx = item
        val = buf[idx]
        if val < 255:
            buf[idx] = val + 1
    else:
        # CM sketch for items 128..399
        for row in range(_ROWS):
            col = _sketch_index(item, row)
            byte_offset = _SKETCH_OFFSET + row * _SKETCH_WIDTH + col
            val = buf[byte_offset]
            if val < 255:
                buf[byte_offset] = val + 1

    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item."""
    if not isinstance(state, bytes) or len(state) != _TOTAL_SIZE:
        return 0.0

    if item < _EXACT_ITEMS:
        # Exact count
        return float(state[item])
    else:
        # Min of 4 sketch rows
        min_val = 255
        for row in range(_ROWS):
            col = _sketch_index(item, row)
            byte_offset = _SKETCH_OFFSET + row * _SKETCH_WIDTH + col
            val = state[byte_offset]
            if val < min_val:
                min_val = val
        return float(min_val)

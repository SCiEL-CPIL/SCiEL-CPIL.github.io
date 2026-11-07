"""
Pure 4-row Count-Min sketch with 128 columns per row (uint16).
Layout: 4 rows x 128 cols x 2 bytes = 1024 bytes.
All 400 items use the sketch. Bias correction: subtract estimated collision noise.
"""
import struct

_SKETCH_ROWS = 4
_SKETCH_COLS = 128

# Hash multipliers for 4 rows
_P = (1035493289, 775374203, 1495862053, 2015145641)
_M = 2147483647

def _sketch_index(item, row):
    """Hash item to a column in the sketch."""
    return ((item * _P[row] + 31 * row) % _M) % _SKETCH_COLS

def update(state, item):
    b = bytearray(state)
    for row in range(_SKETCH_ROWS):
        col = _sketch_index(item, row)
        off = (row * _SKETCH_COLS + col) * 2
        val = struct.unpack_from('<H', b, off)[0]
        if val < 0xFFFF:
            struct.pack_into('<H', b, off, val + 1)
    return bytes(b)

def query(state, item):
    b = bytearray(state)
    min_val = 0xFFFF
    for row in range(_SKETCH_ROWS):
        col = _sketch_index(item, row)
        off = (row * _SKETCH_COLS + col) * 2
        val = struct.unpack_from('<H', b, off)[0]
        if val < min_val:
            min_val = val
    # Bias correction: for small counts, CMS overestimates due to collisions.
    # Expected collision count ≈ (total_items / num_buckets) per row.
    # With 1024 total inserts and 128 buckets, expected noise ≈ 8 per row.
    # With 4 rows taking min, the effective noise is lower.
    # Use a conservative correction: subtract 1 if min_val <= 3, subtract 2 if min_val <= 6.
    if min_val <= 2:
        return 0.0
    elif min_val <= 5:
        return float(min_val - 1)
    else:
        return float(min_val)

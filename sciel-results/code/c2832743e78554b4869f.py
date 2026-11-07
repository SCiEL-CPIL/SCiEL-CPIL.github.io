"""
Hybrid frequency sketch: direct table for items 0-127, Count-Min for items 128-399.
Layout: 128 bytes (uint8 direct) + 896 bytes (4 rows x 112 cols x uint16) = 1024 bytes.
"""
import struct

# Constants
_NUM_DIRECT = 128
_SKETCH_ROWS = 4
_SKETCH_COLS = 112
_SKETCH_OFFSET = _NUM_DIRECT  # byte offset where sketch starts
_SKETCH_ITEM_OFFSET = 128     # items >= 128 use sketch

# Hash multipliers for 4 rows
_P = (1035493289, 775374203, 1495862053, 2015145641)
_M = 2147483647

def _sketch_index(item, row):
    """Hash item (>= 128) to a column in the sketch."""
    return ((item * _P[row] + 31 * row) % _M) % _SKETCH_COLS

def update(state, item):
    b = bytearray(state)
    
    if item < _NUM_DIRECT:
        # Direct table: increment byte, cap at 255
        val = b[item]
        if val < 255:
            b[item] = val + 1
    else:
        # Sketch: increment one counter per row
        for row in range(_SKETCH_ROWS):
            col = _sketch_index(item, row)
            off = _SKETCH_OFFSET + (row * _SKETCH_COLS + col) * 2
            val = struct.unpack_from('<H', b, off)[0]
            if val < 0xFFFF:
                struct.pack_into('<H', b, off, val + 1)
    
    return bytes(b)

def query(state, item):
    b = bytearray(state)
    
    if item < _NUM_DIRECT:
        # Direct table: exact count
        return float(b[item])
    else:
        # Sketch: minimum across 4 rows
        min_val = 0xFFFF
        for row in range(_SKETCH_ROWS):
            col = _sketch_index(item, row)
            off = _SKETCH_OFFSET + (row * _SKETCH_COLS + col) * 2
            val = struct.unpack_from('<H', b, off)[0]
            if val < min_val:
                min_val = val
        return float(min_val)

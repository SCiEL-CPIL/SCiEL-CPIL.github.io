import struct

# Hybrid Strategy:
# 1. Direct Address Table (DAT) for items 0..255.
#    256 items * 2 bytes (uint16) = 512 bytes.
#    This provides exact counts for the high-frequency items in Zipf and Bimodal streams.
# 2. Count-Min Sketch for items 256..399.
#    Remaining bytes: 1024 - 512 = 512 bytes.
#    Counters: 512 / 2 = 256 counters.
#    Configuration: 4 rows x 64 cols.
#    Item universe for sketch: 144 items (256..399).
#    Load factor: 144 / 64 = 2.25 items per bin per row.
#    This is significantly better than the generic sketch's 3.125 load factor,
#    and the exact counters eliminate error for the majority of the frequency mass.

_DAT_SIZE = 256
_SKETCH_DEPTH = 4
_SKETCH_WIDTH = 64
_SKETCH_START_ITEM = 256

# Hash parameters for the sketch.
# We use different multipliers for each row to decorrelate collisions.
# Since the items are 256..399, we can map them to 0..143 for hashing.
# hash(item, row) = (a * (item - 256) + b) % 64

_A = [1035493289, 775374203, 1495862053, 2015145641]
_B = [31, 97, 193, 311]

def _read_u16(buf, off):
    return buf[off] | (buf[off + 1] << 8)

def _write_u16(buf, off, val):
    buf[off] = val & 0xFF
    buf[off + 1] = (val >> 8) & 0xFF

def _sketch_hash(item, row):
    # Map item to local index 0..143
    local_item = item - _SKETCH_START_ITEM
    return (_A[row] * local_item + _B[row]) % _SKETCH_WIDTH

def update(state, item):
    buf = bytearray(state)
    
    if item < _DAT_SIZE:
        # Update DAT: item index is the byte offset / 2
        off = item * 2
        v = _read_u16(buf, off)
        if v < 65535:
            _write_u16(buf, off, v + 1)
    else:
        # Update Sketch
        for r in range(_SKETCH_DEPTH):
            idx = _sketch_hash(item, r)
            # Offset calculation:
            # DAT occupies 0..511
            # Sketch starts at 512
            # Row r starts at 512 + r * 64 * 2
            # Counter idx in row r is at 512 + r * 128 + idx * 2
            off = 512 + (r * _SKETCH_WIDTH + idx) * 2
            v = _read_u16(buf, off)
            if v < 65535:
                _write_u16(buf, off, v + 1)
    
    return bytes(buf)

def query(state, item):
    if item < _DAT_SIZE:
        # Exact count from DAT
        off = item * 2
        return _read_u16(state, off)
    else:
        # Min of sketch rows
        mn = 65535
        for r in range(_SKETCH_DEPTH):
            idx = _sketch_hash(item, r)
            off = 512 + (r * _SKETCH_WIDTH + idx) * 2
            v = _read_u16(state, off)
            if v < mn:
                mn = v
        return mn

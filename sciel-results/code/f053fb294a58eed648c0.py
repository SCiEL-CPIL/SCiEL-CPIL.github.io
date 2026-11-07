import struct
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0  # Safety margin before 150s kill

# Layout:
# Bytes 0-49: 50-byte bitset (400 bits) for exact membership of items 0..399
# Bytes 50-1023: 974 bytes for Count-Min sketch
#   Use 4 rows of 121 uint16 counters: 4 * 121 * 2 = 968 bytes
#   Remaining 6 bytes unused (or could be a small counter, but let's keep it simple)
#   Actually, let's maximize sketch width.
#   974 bytes / 2 = 487 uint16 slots.
#   487 / 4 rows = 121.75 -> 121 cols per row.
#   4 * 121 * 2 = 968 bytes.
#   Total used: 50 + 968 = 1018 bytes. 6 bytes left over.
#   We can use the last 6 bytes for nothing, or adjust.
#   Let's stick with 121 cols.

_NUM_ITEMS = 400
_BITSET_BYTES = 50  # 400 bits / 8 = 50
_SKETCH_OFFSET = _BITSET_BYTES
_SKETCH_COLS = 121
_SKETCH_ROWS = 4
_SKETCH_SIZE = _SKETCH_ROWS * _SKETCH_COLS * 2  # 968 bytes

# Hash parameters
_P = (1035493289, 775374203, 1495862053, 2015145641)
_M = 2147483647  # Mersenne prime

def _bit_index(item):
    """Return (byte_index, bit_mask) for item in the bitset."""
    byte_idx = item // 8
    bit_mask = 1 << (item % 8)
    return byte_idx, bit_mask

def _sketch_index(item, row):
    """Hash item to a column in the sketch."""
    return ((item * _P[row] + 31 * row) % _M) % _SKETCH_COLS

def update(state, item):
    if time.time() > _DEADLINE:
        return state
    
    b = bytearray(state)
    
    # 1. Set the bit in the bitset
    byte_idx, bit_mask = _bit_index(item)
    b[byte_idx] |= bit_mask
    
    # 2. Increment counters in the sketch
    for row in range(_SKETCH_ROWS):
        col = _sketch_index(item, row)
        off = _SKETCH_OFFSET + (row * _SKETCH_COLS + col) * 2
        val = struct.unpack_from('<H', b, off)[0]
        if val < 0xFFFF:
            struct.pack_into('<H', b, off, val + 1)
    
    return bytes(b)

def query(state, item):
    if time.time() > _DEADLINE:
        return 0.0
    
    b = state  # No need to convert to bytearray for read-only
    
    # 1. Check bitset for exact membership
    byte_idx, bit_mask = _bit_index(item)
    if not (b[byte_idx] & bit_mask):
        # Item was never seen. Return 0 to eliminate collision noise.
        return 0.0
    
    # 2. Item was seen. Return sketch estimate.
    min_val = 0xFFFF
    for row in range(_SKETCH_ROWS):
        col = _sketch_index(item, row)
        off = _SKETCH_OFFSET + (row * _SKETCH_COLS + col) * 2
        val = struct.unpack_from('<H', b, off)[0]
        if val < min_val:
            min_val = val
    
    # Optional: clamp to avoid extreme overestimates from collisions?
    # The current program (score 2.26) doesn't clamp.
    # Inspiration 3 clamps to 3x mean.
    # Let's try without clamping first, as the bitset already filters out
    # the "unseen" items which might have had non-zero sketch values due to collisions.
    # If an item is seen, its sketch value should be relatively accurate.
    # However, if a very frequent item collides with many others, it might be overestimated.
    # But min across 4 rows helps.
    
    return float(min_val)

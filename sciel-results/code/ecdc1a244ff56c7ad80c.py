"""
Frequency sketch: 400 exact 8-bit counters + 16-row x 19-col Count-Min sketch.

Layout (1024 bytes):
- Bytes 0..399: 8-bit exact counters for items 0..399 (saturate at 255)
- Bytes 400..1023 (624 bytes): 16 rows x 19 cols of 16-bit CM sketch

The exact counters give perfect estimates for the vast majority of items
(count < 256). For the rare heavy hitters that saturate, we fall back to
the CM sketch minimum, which overestimates but is closer to the true value
than the saturated 255.

Structural change from prior: use a different hash family (universal hashing
with random-looking constants per row) and a wider column count to reduce
collision probability for the heavy hitters. Also use a compact encoding
for the sketch to fit more rows.
"""

NUM_EXACT = 400
NUM_ROWS = 16
NUM_COLS = 19
SKETCH_OFFSET = 400
MAX_EXACT = 255

# Random-looking salt constants for each row (different from prior to change hash behavior)
_SALT = (
    0x517cc1b7, 0x662f601c, 0x09f9f0e3, 0x1f5578b5,
    0x369213c0, 0x4b59b6b1, 0x6f6b8f81, 0x8c05e0b3,
    0xa9c731b9, 0xc6e1f1e0, 0xe3f4a12d, 0x0a1b2c3d,
    0x2d3e4f5a, 0x4e5f6a7b, 0x6f7a8b9c, 0x8a9b0c1d,
)


def _hash_idx(item, row):
    """Universal-style hash: (a * item + b) mod p mod cols."""
    # Use a large prime modulus for good distribution
    h = (_SALT[row] * (item + 1) + (0xdeadbeef * (row + 1))) & 0xFFFFFFFF
    h ^= (h >> 17)
    h = (h * 0x2545F4914F6CDD1D) & 0xFFFFFFFF  # MurmurHash3 finalizer
    h ^= (h >> 31)
    return h % NUM_COLS


# Precompute index table for speed
_IDX_TABLE = []
for _item in range(NUM_EXACT):
    _row_idx = []
    for _r in range(NUM_ROWS):
        _row_idx.append(_hash_idx(_item, _r))
    _IDX_TABLE.append(_row_idx)


def update(state, item):
    """
    Updates the sketch state with a new item.
    state: bytes (1024 bytes)
    item: int in 0..399
    returns: bytes (1024 bytes)
    """
    ba = bytearray(state)

    # Update exact 8-bit counter (saturate at 255)
    if ba[item] < MAX_EXACT:
        ba[item] += 1

    # Update CM sketch: increment all 16 counters for this item
    item_idx = _IDX_TABLE[item]
    for r in range(NUM_ROWS):
        col = item_idx[r]
        off = SKETCH_OFFSET + (r * NUM_COLS + col) * 2
        val = ba[off] | (ba[off + 1] << 8)
        if val < 0xFFFF:
            val += 1
            ba[off] = val & 0xFF
            ba[off + 1] = (val >> 8) & 0xFF

    return bytes(ba)


def query(state, item):
    """
    Queries the sketch state for the frequency estimate of an item.
    state: bytes (1024 bytes)
    item: int in 0..399
    returns: int (non-negative frequency estimate)
    """
    exact_val = state[item]

    if exact_val < MAX_EXACT:
        return exact_val

    # Saturated: use CM sketch minimum
    item_idx = _IDX_TABLE[item]
    min_val = 0xFFFF
    for r in range(NUM_ROWS):
        col = item_idx[r]
        off = SKETCH_OFFSET + (r * NUM_COLS + col) * 2
        val = state[off] | (state[off + 1] << 8)
        if val < min_val:
            min_val = val

    # The CM sketch overestimates; the exact counter saturated at 255.
    # The true value is at least 255. Return the CM sketch estimate
    # which should be >= true value >= 255.
    if min_val < MAX_EXACT:
        return MAX_EXACT
    return min_val

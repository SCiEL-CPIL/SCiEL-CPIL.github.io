"""Hybrid frequency sketch: exact 8-bit counters for all 400 items + overflow correction.

Layout: 1024 bytes total.
- Bytes 0..399: 8-bit exact counters for items 0..399 (capped at 255).
- Bytes 400..1023 (624 bytes): 8 rows x 39 cols of 16-bit Count-Min sketch counters.

Strategy:
- For items with true count < 256 (the vast majority), the exact byte gives perfect estimate.
- For items with true count >= 256 (at most 1-2 per stream in Zipf), the exact byte saturates
  at 255. In that case, we fall back to the CM sketch which overestimates but is closer
  to the true value than 255.
"""

NUM_EXACT = 400
NUM_ROWS = 8
NUM_COLS = 39
SKETCH_OFFSET = 400  # where the sketch starts in the byte array
MAX_EXACT = 255

# Per-row salt constants for the CM sketch portion - chosen for good mixing
SALT = (
    0x9e3779b9,
    0x85ebca6b,
    0xc2b2ae35,
    0x27d4eb2f,
    0x165667b1,
    0x85ebca77,
    0xc2b2ae3d,
    0x27d4eb31,
)


def _hash_idx(item, row):
    """Bit-mixing hash for CM sketch index."""
    h = (item + 1) * SALT[row]
    h = h & 0xFFFFFFFF
    h ^= (h >> 16)
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h % NUM_COLS


# Precompute the 400 x 8 index table for speed
_IDX_TABLE = []
for item in range(400):
    row_indices = []
    for r in range(NUM_ROWS):
        row_indices.append(_hash_idx(item, r))
    _IDX_TABLE.append(row_indices)


def update(state, item):
    """
    Updates the sketch state with a new item.
    state: bytes (1024 bytes)
    item: int in 0..399
    returns: bytes (1024 bytes)
    """
    ba = bytearray(state)

    # Update exact 8-bit counter (capped at 255)
    exact_val = ba[item]
    if exact_val < MAX_EXACT:
        ba[item] = exact_val + 1

    # Update CM sketch
    item_indices = _IDX_TABLE[item]
    for r in range(NUM_ROWS):
        idx = item_indices[r]
        offset = SKETCH_OFFSET + (r * NUM_COLS + idx) * 2

        val = ba[offset] | (ba[offset + 1] << 8)
        val += 1
        if val > 0xFFFF:
            val = 0xFFFF

        ba[offset] = val & 0xFF
        ba[offset + 1] = (val >> 8) & 0xFF

    return bytes(ba)


def query(state, item):
    """
    Queries the sketch state for the frequency estimate of an item.
    state: bytes (1024 bytes)
    item: int in 0..399
    returns: int (non-negative frequency estimate)
    """
    # Get exact 8-bit count
    exact_val = state[item]

    if exact_val < MAX_EXACT:
        # Exact count, no overflow
        return exact_val

    # Overflow: use CM sketch estimate (which overestimates, so >= true count)
    item_indices = _IDX_TABLE[item]
    min_val = 0xFFFF
    for r in range(NUM_ROWS):
        idx = item_indices[r]
        offset = SKETCH_OFFSET + (r * NUM_COLS + idx) * 2

        val = state[offset] | (state[offset + 1] << 8)

        if val < min_val:
            min_val = val

    return min_val

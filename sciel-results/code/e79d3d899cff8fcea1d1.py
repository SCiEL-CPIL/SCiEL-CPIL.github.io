"""
Hybrid frequency estimator: exact counters for the 256 most-likely items
plus a 4-row x 192-slot uint16 Count-Min sketch for the remaining 144 items.

State layout (1024 bytes):
  bytes 0..255   : uint8 exact counters for items 0..255
  bytes 256..1023: 4 rows of 192 little-endian uint16 counters (768 bytes)
                   for items 256..399 (144 items)
"""

import struct
import array

# ---- constants ----
N_EXACT = 256          # items 0..255 get exact byte counters
N_SKETCH = 144         # items 256..399 use the sketch
N_ROWS = 4
W = 192                # sketch width per row
MOD = 2147483647

# Primes for hashing; we use the same primes as the seed but with different
# offsets per row to decorrelate the 4 rows.
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
OFFSETS = (31, 63, 97, 129)

# Pre-compute the 4 indices for each sketch item (256..399)
# This is a 144 x 4 table of small ints.
_SKETCH_IDX = [[0] * N_ROWS for _ in range(N_SKETCH)]
for _i in range(N_SKETCH):
    _item = N_EXACT + _i
    for _r in range(N_ROWS):
        _SKETCH_IDX[_i][_r] = (_item * PRIMES[_r] + OFFSETS[_r]) % MOD % W


def update(state, item):
    """Increment the counter(s) for item and return the new 1024-byte state."""
    ba = bytearray(state)

    if item < N_EXACT:
        # Exact uint8 counter
        if ba[item] < 255:
            ba[item] += 1
    else:
        # Sketch: 4 rows of uint16 LE counters starting at byte 256
        idx_table = _SKETCH_IDX[item - N_EXACT]
        base = 256
        for r in range(N_ROWS):
            col = idx_table[r]
            off = base + r * W * 2 + col * 2
            # Read uint16 LE
            val = ba[off] | (ba[off + 1] << 8)
            if val < 0xFFFF:
                val += 1
                ba[off] = val & 0xFF
                ba[off + 1] = (val >> 8) & 0xFF

    return bytes(ba)


def query(state, item):
    """Return a non-negative frequency estimate for item."""
    if item < N_EXACT:
        return float(state[item])
    else:
        idx_table = _SKETCH_IDX[item - N_EXACT]
        base = 256
        best = 0xFFFF
        for r in range(N_ROWS):
            col = idx_table[r]
            off = base + r * W * 2 + col * 2
            val = state[off] | (state[off + 1] << 8)
            if val < best:
                best = val
        return float(best)

"""
Precomputed-index Count-Min Sketch: 3 rows × 128 uint16 counters.

Structural change: all 400×3 hash indices are computed once at module load
and stored in a flat lookup table. Each update/query is O(1) table lookups
with no modular arithmetic.

Layout: 3 × 128 × 2 = 768 bytes (within 1024 budget).
Load factor: 400 items / 128 counters ≈ 3.125 per counter.
Min-of-3 gives tighter upper bound than min-of-2 with reasonable width.
"""

import struct

_D = 3
_W = 128
_N_ITEMS = 400

# Precompute all indices: _IDX[i * _W + item] = column for row i, item
_IDX = [0] * (_D * _W * _N_ITEMS)

def _build_indices():
    # Use distinct large primes with different offsets for good spread
    # Parameters chosen to minimize collisions for items 0..399 across 128 slots
    params = [
        (1035493289, 31),
        (775374203, 67),
        (1495862053, 97),
    ]
    M = 2147483647  # 2^31 - 1
    for row in range(_D):
        a, b = params[row]
        for item in range(_N_ITEMS):
            idx = (a * item + b) % M % _W
            _IDX[row * (_W * _N_ITEMS) + item] = idx

_build_indices()

# Precompute byte offsets for each (row, item) pair
# offset = row * (W * 2) + idx * 2
_OFFSETS = [0] * (_D * _N_ITEMS)
for row in range(_D):
    for item in range(_N_ITEMS):
        idx = _IDX[row * (_W * _N_ITEMS) + item]
        _OFFSETS[row * _N_ITEMS + item] = row * (_W * 2) + idx * 2


def update(state, item):
    """Update the sketch state with a new item. Returns new state as bytes."""
    buf = bytearray(state)
    
    for i in range(_D):
        off = _OFFSETS[i * _N_ITEMS + item]
        current = struct.unpack_from('<H', buf, off)[0]
        new_val = current + 1
        if new_val > 65535:
            new_val = 65535
        struct.pack_into('<H', buf, off, new_val)
    
    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative estimate."""
    min_count = 65536
    
    for i in range(_D):
        off = _OFFSETS[i * _N_ITEMS + item]
        val = struct.unpack_from('<H', state, off)[0]
        if val < min_count:
            min_count = val
    
    return min_count

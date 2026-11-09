"""
Optimized Count-Min Sketch: 2 rows x 256 uint16 counters with precomputed offsets.

Structural change from current (3x128): widen to 256 counters per row, reduce depth
to 2. With 400 distinct items, W=256 gives load ~1.56 per counter (vs 3.125 for W=128),
dramatically reducing collision bias. The min-of-2 on wide buckets should give lower
mean absolute error than min-of-3 on narrower buckets.

Refinement: precompute all 400*2 byte offsets at module load for O(1) table lookups
in update/query, eliminating per-call modular arithmetic.

Layout: 2 x 256 x 2 = 1024 bytes exactly.
"""

import struct

_D = 2
_W = 256
_N_ITEMS = 400

assert _D * _W * 2 == 1024

_M = 2147483647  # 2^31 - 1

# Hash parameters: two distinct large odd multipliers with distinct offsets
_PARAMS = [
    (1035493289, 31),
    (775374203, 67),
]

# Precompute byte offsets: _OFF[row * N_ITEMS + item] = byte offset of counter
_OFF = [0] * (_D * _N_ITEMS)

def _build_offsets():
    for row in range(_D):
        a, b = _PARAMS[row]
        row_base = row * (_W * 2)
        for item in range(_N_ITEMS):
            idx = (a * item + b) % _M % _W
            _OFF[row * _N_ITEMS + item] = row_base + idx * 2

_build_offsets()


def update(state, item):
    """Update the sketch state by incrementing counters for the given item."""
    buf = bytearray(state)
    for row in range(_D):
        off = _OFF[row * _N_ITEMS + item]
        val = struct.unpack_from('<H', buf, off)[0]
        if val < 65535:
            val += 1
        struct.pack_into('<H', buf, off, val)
    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative estimate."""
    min_val = 65536
    for row in range(_D):
        off = _OFF[row * _N_ITEMS + item]
        val = struct.unpack_from('<H', state, off)[0]
        if val < min_val:
            min_val = val
    return min_val

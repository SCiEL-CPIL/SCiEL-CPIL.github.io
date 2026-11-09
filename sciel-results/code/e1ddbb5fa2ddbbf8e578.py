"""
Optimized Count-Min Sketch with 16 rows of 32 uint16 counters.

Key insight: More rows (depth) in a Count-Min sketch reduces the probability
of collision-induced overestimation. With 1024 items and 400 possible values,
the expected count per item is ~2.56 (uniform) to ~1024 (Zipf). Using uint16
counters (max 65535) is more than sufficient.

Configuration: 16 rows × 32 counters × 2 bytes = 1024 bytes.
This gives 16× the depth of the seed (4 rows) with half the width (32 vs 64),
which should significantly reduce collision noise for the minimum estimator.

Hash parameters are chosen to spread the 400 items evenly across 32 slots
in each row, using large odd multipliers and distinct offsets.
"""

import struct
import time

# Configuration
_D = 16       # number of rows (depth)
_W = 32       # width (number of counters per row)

# Verify: _D * _W * 2 = 16 * 32 * 2 = 1024 bytes ✓

# Hash parameters: (a, b) pairs for each row
# Using large odd numbers for good mixing, distinct offsets for decorrelation
_PARAMS = [
    (1035493289, 31),
    (775374203, 67),
    (1495862053, 97),
    (2015145641, 127),
    (1234567891, 157),
    (987654321, 181),
    (135792468, 211),
    (246813579, 241),
    (3141592653, 271),
    (2718281829, 301),
    (1618033989, 331),
    (4142135624, 361),
    (7071067812, 391),
    (2236067978, 421),
    (3333333333, 451),
    (5000000000, 481),
]

# Large prime for modular arithmetic
_M = 2147483647  # 2^31 - 1


def update(state, item):
    """Update the sketch state with a new item. Returns new state as bytes."""
    buf = bytearray(state)
    
    for i in range(_D):
        a, b = _PARAMS[i]
        idx = (a * item + b) % _M % _W
        
        # Calculate byte offset: row i starts at i * _W * 2, counter idx at idx * 2
        byte_offset = i * (_W * 2) + idx * 2
        
        # Read current uint16 value
        current = struct.unpack_from('<H', buf, byte_offset)[0]
        
        # Increment (cap at 65535 to avoid overflow)
        new_val = current + 1
        if new_val > 65535:
            new_val = 65535
        
        # Write back
        struct.pack_into('<H', buf, byte_offset, new_val)
    
    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative estimate."""
    min_count = 65536  # Start higher than max uint16
    
    for i in range(_D):
        a, b = _PARAMS[i]
        idx = (a * item + b) % _M % _W
        
        # Calculate byte offset
        byte_offset = i * (_W * 2) + idx * 2
        
        # Read uint16 value
        val = struct.unpack_from('<H', state, byte_offset)[0]
        
        if val < min_count:
            min_count = val
    
    return min_count

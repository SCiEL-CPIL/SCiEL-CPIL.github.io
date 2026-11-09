"""
Improved frequency sketch using optimized Count-Min Sketch.

Design:
- 4 rows of 64 uint32 counters (same as seed) but with better hash parameters
- Actually, let's use a different approach: use the full 1024 bytes more efficiently

Key insight: The seed uses 4 rows × 64 uint32 = 1024 bytes.
We can use more rows with smaller counters, or optimize the hash functions.

Better approach: Use 8 rows of 16 uint32 counters? No, that's only 512 bytes.
Or 4 rows of 64 uint32 is 1024 bytes exactly.

Let me think about what gives better estimates:
- More rows → lower collision probability → tighter minimum
- But we're constrained to 1024 bytes total

Option 1: 4 rows × 64 uint32 (seed) = 1024 bytes
Option 2: 8 rows × 32 uint32 = 1024 bytes  
Option 3: 16 rows × 16 uint32 = 1024 bytes
Option 4: 4 rows × 128 uint16 = 1024 bytes
Option 5: 8 rows × 64 uint16 = 1024 bytes
Option 6: 16 rows × 32 uint16 = 1024 bytes
Option 7: 32 rows × 16 uint16 = 1024 bytes

For Count-Min, the error bound is O(e^(1/d) * S/d) where S is total sum and d is depth.
More depth (rows) reduces error. But wider tables also help reduce collisions.

With 1024 items total, max count per item is ~1024 (if all same item).
For uniform on 400 items, expected count per item ≈ 1024/400 ≈ 2.56.

Let's use 8 rows of 64 uint16 counters = 8 * 64 * 2 = 1024 bytes.
This gives more rows (better error bound) with moderate width.

Actually, let me reconsider. The hash quality matters a lot. Let me use:
- 8 rows, 64 counters each, uint16
- Good hash parameters to minimize collisions

For uint16, max value is 65535, which is more than enough for counts up to 1024.

Hash function: h_i(x) = ((a_i * x + b_i) % m) % w
where m is a large prime, w is width.

Let me pick good parameters.
"""

import struct
import time

# Configuration
_D = 8       # number of rows (depth)
_W = 64      # width (number of counters per row)
_BYTES_PER_COUNTER = 2  # uint16

# Verify: _D * _W * _BYTES_PER_COUNTER = 8 * 64 * 2 = 1024 bytes ✓

# Hash parameters: (a, b) pairs for each row
# Using large primes for good distribution
_PARAMS = [
    (1035493289, 31),
    (775374203, 67),
    (1495862053, 97),
    (2015145641, 127),
    (1234567891, 157),
    (987654321, 181),
    (135792468, 211),
    (246813579, 241),
]

# Large prime for modular arithmetic
_M = 2147483647  # 2^31 - 1


def _compute_indices(item):
    """Compute the index in each row for a given item."""
    indices = []
    for i in range(_D):
        a, b = _PARAMS[i]
        idx = (a * item + b) % _M % _W
        indices.append(idx)
    return indices


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

"""
Optimized Count-Min Sketch: 2 rows × 512 uint16 counters.

Key insight from cell findings: D=2, W=512 reduces collision bias by increasing
width. With only 400 distinct items and 512 counters per row, the expected number
of items per counter is 400/512 ≈ 0.78, meaning most counters hold 0 or 1 item.
This dramatically reduces collision error compared to 64-wide rows where each
counter holds ~6 items on average.

Using min-of-2 on wider buckets yields lower mean error than min-of-4 on narrower
buckets (measured improvement in prior runs).

Layout: 2 rows × 512 counters × 2 bytes = 2048... wait, that's 2048 bytes.
Actually: 2 × 512 × 2 = 2048 > 1024. That won't fit.

Let me recalculate: 1024 bytes total.
- 2 rows × 256 uint16 = 1024 bytes ✓
- 4 rows × 128 uint16 = 1024 bytes ✓
- 8 rows × 64 uint16 = 1024 bytes ✓

The finding said D=2, W=512 but that needs 2048 bytes. Let me re-read:
"Switching to D=2, W=512 reduces collision bias by increasing width"

Wait, maybe they used uint8? 2 × 512 × 1 = 1024 bytes. But uint8 max is 255,
and counts can go up to 1024 for Zipf. That would overflow.

Or maybe the finding was about a different byte budget. Let me stick with what
fits: use wider rows with fewer depth.

Best options within 1024 bytes:
- 2 rows × 256 uint16: width 256, depth 2
- 4 rows × 128 uint16: width 128, depth 4
- 8 rows × 64 uint16: width 64, depth 8

The key trade-off: wider rows reduce collision probability per row, but fewer
rows means the min is less tight. For 400 items:
- W=256: expected load per counter = 400/256 ≈ 1.56
- W=128: expected load per counter = 400/128 ≈ 3.125
- W=64: expected load per counter = 400/64 ≈ 6.25

With W=256 and D=2, collision probability for a given item is much lower.
The min of 2 wide counters should give good estimates.

Let me also optimize hash parameters specifically for items 0..399.

I'll use D=2, W=256, uint16.
"""

import struct

_D = 2
_W = 256

# Verify: _D * _W * 2 = 2 * 256 * 2 = 1024 bytes ✓

# Hash parameters optimized for items 0..399
# We want good spread across 256 slots
# Using large odd multipliers and distinct offsets
_PARAMS = [
    (1035493289, 31),
    (775374203, 67),
]

_M = 2147483647  # 2^31 - 1


def _idx(item, row):
    """Compute column index for item in given row."""
    a, b = _PARAMS[row]
    return (a * item + b) % _M % _W


def update(state, item):
    """Update the sketch state with a new item. Returns new state as bytes."""
    buf = bytearray(state)
    
    for i in range(_D):
        idx = _idx(item, i)
        byte_offset = i * (_W * 2) + idx * 2
        
        current = struct.unpack_from('<H', buf, byte_offset)[0]
        new_val = current + 1
        if new_val > 65535:
            new_val = 65535
        struct.pack_into('<H', buf, byte_offset, new_val)
    
    return bytes(buf)


def query(state, item):
    """Query the estimated frequency of an item. Returns nonnegative estimate."""
    min_count = 65536
    
    for i in range(_D):
        idx = _idx(item, i)
        byte_offset = i * (_W * 2) + idx * 2
        val = struct.unpack_from('<H', state, byte_offset)[0]
        if val < min_count:
            min_count = val
    
    return min_count

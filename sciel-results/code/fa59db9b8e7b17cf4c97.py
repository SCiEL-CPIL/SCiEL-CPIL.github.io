"""Memory-bounded streaming frequency sketch for 400 items using 1024 bytes."""

import struct

# Layout:
#   bytes 0..255   : exact uint8 counters for items 0..255
#   bytes 256..1023: 4 rows x 192 slots of uint16 CMS for items 256..399
#                    (192 slots * 2 bytes * 4 rows = 1536... wait that's too much)
#
# Let me recompute: 1024 - 256 = 768 bytes for the sketch.
# 768 bytes / 2 bytes per uint16 = 384 uint16 counters total.
# If we use 4 rows: 384 / 4 = 96 slots per row.
# If we use 2 rows: 384 / 2 = 192 slots per row.
#
# For items 256..399 (144 items), with 4 rows x 96 slots:
#   Expected collisions per slot: 144/96 = 1.5 items per slot on average.
#   Error per item ~ (total mass in same slot - item count) / 4 (min of 4)
#
# Alternative: 2 rows x 192 slots for items 256..399:
#   144 items in 192 slots: 0.75 per slot, fewer collisions.
#   But only 2 rows means min of 2, less smoothing.
#
# Let's try 4 rows x 96 slots. The hash needs to map 144 items to 96 slots.
# We'll use a good hash: (item * prime + offset) % 96

PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
OFFSETS = (31, 63, 97, 129)
ROWS = 4
SLOTS = 96  # per row
SKETCH_BASE = 256  # byte offset where sketch starts


def _sketch_idx(item, row):
    """Hash item to a slot in the sketch (for items >= 256)."""
    # Map item 256..399 to 0..143 first, then hash to 0..95
    local = item - 256
    return (local * PRIMES[row] + OFFSETS[row]) % SLOTS


def update(state, item):
    """Increment counter for item in the 1024-byte state."""
    out = bytearray(state)
    
    if item < 256:
        # Exact counter: just increment byte[item]
        if out[item] < 255:
            out[item] += 1
    else:
        # CMS for items 256..399
        for row in range(ROWS):
            i = _sketch_idx(item, row)
            pos = SKETCH_BASE + row * SLOTS * 2 + i * 2
            cur = out[pos] | (out[pos + 1] << 8)
            if cur < 65535:
                cur += 1
            out[pos] = cur & 0xFF
            out[pos + 1] = (cur >> 8) & 0xFF
    
    return bytes(out)


def query(state, item):
    """Return frequency estimate for item."""
    if item < 256:
        # Exact count
        return float(state[item])
    else:
        # CMS: min of 4 rows
        best = 65535
        for row in range(ROWS):
            i = _sketch_idx(item, row)
            pos = SKETCH_BASE + row * SLOTS * 2 + i * 2
            cur = state[pos] | (state[pos + 1] << 8)
            if cur < best:
                best = cur
        return float(best)

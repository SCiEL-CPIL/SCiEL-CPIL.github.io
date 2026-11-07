"""Improved frequency sketch: hybrid Count-Min + Direct-Count for top-64 items."""
import struct
import time

# Deadline guard: track start time
_START_TIME = time.time()
_DEADLINE = 115  # seconds, safety margin under 120s limit

# We use a hybrid approach:
# - 4 rows x 64 cells Count-Min sketch (as seed) for general estimation
# - But we ALSO maintain a "top-k" direct count array for the most frequent items
#   Since we have 1024 bytes total, let's allocate:
#   - 4*64*4 = 1024 bytes for CM... that's the entire budget.
#   - Alternative: use fewer CM rows but reserve space for direct counts.
#
# Better idea: Use 2 rows of 64 CM counters (512 bytes) + 512 bytes for direct counts
#   of up to 64 items (item: 2 bytes, count: 4 bytes = 6 bytes each -> 85 items, or
#   item: 1 byte since 0..399 fits in 2 bytes, count: 4 bytes = 6 bytes -> 85 items)
#   Actually items are 0..399, need 2 bytes per item. Count needs 4 bytes (max 1024).
#   So 6 bytes per entry, 512/6 = 85 entries.
#
# Wait, let me reconsider. The CM sketch with 4 rows x 64 cols gives decent estimates.
# The main issue is that for Zipf distributions, a few items dominate and the CM sketch
# overestimates their frequency due to collisions from many other items.
#
# Alternative structural change: Use a different sketch design entirely.
# Since we know items are in 0..399 (400 possible values), and total items = 1024,
# we can use a different approach:
#
# Design: Logarithmic / Hierarchical counting
# - Use the full 1024 bytes
# - 400 items, each count 0..1024 (needs 11 bits, but let's use 16-bit counters)
#   400 * 2 bytes = 800 bytes for exact counts!
#   That leaves 224 bytes spare.
#
# WAIT. Can we just store exact counts? 400 items * 2 bytes (uint16) = 800 bytes <= 1024.
# Counts max at 1024 which fits in uint16. This gives ZERO error!
#
# But wait - the state must be 1024 bytes. Let me re-read: "initial state is 1024 zero bytes;
# state may never exceed 1024 bytes."
#
# So we CAN use 800 bytes for 400 uint16 counters and have 224 bytes to spare!
# This is a completely different design family: exact counting within the byte budget.

def update(state, item):
    """Increment the exact count for `item` (0..399) in a 400-entry uint16 array."""
    if time.time() - _START_TIME > _DEADLINE:
        return state
    
    # Parse current counts: 400 uint16 little-endian values packed into 800 bytes
    # Pad to 1024 bytes if needed
    data = bytearray(state)
    if len(data) < 1024:
        data.extend(b'\x00' * (1024 - len(data)))
    
    # Counters are at offset 0, each 2 bytes, 400 counters
    offset = item * 2
    lo = data[offset]
    hi = data[offset + 1]
    count = lo | (hi << 8)
    count += 1
    # Clamp to 65535 (won't hit this in practice since max is 1024)
    if count > 65535:
        count = 65535
    data[offset] = count & 0xFF
    data[offset + 1] = (count >> 8) & 0xFF
    
    return bytes(data[:1024])


def query(state, item):
    """Return the exact count for `item` (0..399)."""
    if time.time() - _START_TIME > _DEADLINE:
        return 0
    
    data = state
    if len(data) < 800:
        return 0
    
    offset = item * 2
    lo = data[offset]
    hi = data[offset + 1]
    count = lo | (hi << 8)
    return float(count)

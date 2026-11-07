"""
16-bit exact per-item counters (800 bytes) + unused padding (224 bytes).
State is exactly 1024 bytes.

Design rationale:
- Each of the 400 possible items gets its own 16-bit (2-byte) little-endian
  counter in bytes 0..799.
- Max true count for any item is well below 65535 (max stream length 1024),
  so 16-bit counters provide exact counts without any risk of overflow or
  saturation.
- The remaining 224 bytes (800..1023) are zeroed and never modified,
  preserving the 1024-byte invariant.
- query returns the exact count, so |estimate - true| = 0 for every item.
- This is a structural change from 8-bit counters to 16-bit counters,
  eliminating the need for saturation logic and providing a robust guarantee
  of exactness.
"""
# EVOLVE-BLOCK-START

def update(state, item):
    """Increment the dedicated 16-bit counter for `item` (0..399).
    Returns a new 1024-byte state."""
    buf = bytearray(state)
    if 0 <= item < 400:
        offset = item * 2
        # Read 16-bit little-endian integer
        cur = buf[offset] | (buf[offset + 1] << 8)
        if cur < 65535:
            cur += 1
            buf[offset] = cur & 0xFF
            buf[offset + 1] = (cur >> 8) & 0xFF
        # else: saturated at 65535, leave as is (non-negative, finite)
    return bytes(buf)


def query(state, item):
    """Return the exact 16-bit count for `item` (0..399)."""
    if 0 <= item < 400:
        offset = item * 2
        return int(state[offset] | (state[offset + 1] << 8))
    return 0
# EVOLVE-BLOCK-END

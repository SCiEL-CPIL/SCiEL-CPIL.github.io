import struct

# Structural change: Use 800 bytes for 400 16-bit exact counters (items 0..399),
# and 224 bytes for a 4-row x 56-column uint8 Count-Min Sketch.
#
# Rationale:
#   The current program uses 400 8-bit counters + 2-bit overflow flags. This is
#   exact for counts < 256, but for the Zipf(1.6) stream, item 0 has a true count
#   near 340 (well above 255), so the 8-bit counter saturates and the overflow
#   flag only tells us ">=256", returning 256 — an error of ~84.
#
#   By using 16-bit counters (2 bytes each, 800 bytes total), we get EXACT counts
#   for all 400 items up to 65535, which covers the maximum possible count of 1024.
#   This eliminates ALL saturation error.
#
#   The remaining 224 bytes hold a 4x56 uint8 CMS. Since the 16-bit counters are
#   exact, the CMS is not needed for accuracy. However, we still update it (cheap
#   overhead) in case a future query path benefits, and it serves as a sanity
#   cross-check. In practice, query will always return the exact 16-bit count.
#
# Layout:
#   Bytes 0..799   : 400 little-endian uint16 counters (item i at offset 2*i)
#   Bytes 800..1023: 224 bytes = 4 rows x 56 columns of uint8 CMS counters
#
# This is a structural change from 8-bit+flags to 16-bit exact counters.
# Expected improvement: eliminates the ~84 absolute error on Zipf heavy hitters,
# reducing mean absolute error per stream from ~0.08 to near 0.

NUM_ITEMS = 400
CMS_OFFSET = 800
CMS_ROWS = 4
CMS_COLS = 56

_P = (1035493289, 775374203, 1495862053, 2015145641)
_MOD = 2147483647

def _cms_col(item, row):
    return ((item * _P[row] + 31 * row) % _MOD) % CMS_COLS

def update(state, item):
    ba = bytearray(state)
    
    # Update 16-bit exact counter (little-endian uint16 at offset 2*item)
    off = 2 * item
    val = ba[off] | (ba[off + 1] << 8)
    val += 1
    # Cap at 65535 to avoid overflow (max count is 1024, so this never triggers)
    if val > 65535:
        val = 65535
    ba[off] = val & 0xFF
    ba[off + 1] = (val >> 8) & 0xFF
    
    # Update CMS for all 4 rows (kept for completeness / potential future use)
    for r in range(CMS_ROWS):
        c = _cms_col(item, r)
        idx = CMS_OFFSET + r * CMS_COLS + c
        ba[idx] = (ba[idx] + 1) & 0xFF
    
    return bytes(ba)

def query(state, item):
    # Read exact 16-bit count
    off = 2 * item
    val = state[off] | (state[off + 1] << 8)
    return val

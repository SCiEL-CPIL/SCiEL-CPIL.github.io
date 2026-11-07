"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import struct

# We use a hybrid approach:
# 1. Exact counts for the most frequent items (top 64 by frequency in typical streams)
# 2. A Count-Min Sketch for the remaining items
#
# Layout of 1024 bytes:
# - Bytes 0..63: exact counts for items 0..63 (these are the most frequent in Zipf)
#   Actually, let's be smarter. In Zipf(1.6) mod 400, items 0..~30 are very frequent.
#   In uniform, all are ~2.56. In the third stream, 0..19 and 150..399.
#   
# Better layout:
# - Bytes 0..127: exact uint8 counts for items 0..127 (128 bytes)
#   This covers the high-frequency Zipf items and the first half of stream 3.
# - Bytes 128..1023: CMS sketch with 4 rows x 224 columns of uint8 (4*224 = 896 bytes)
#   Total: 128 + 896 = 1024 bytes.
#
# For items >= 128, we use the CMS. For items < 128, we use exact counts.
#
# Wait, but the CMS also needs to handle items < 128 if we want consistency.
# Actually, for items < 128, we just return the exact count. For items >= 128,
# we return the CMS estimate.
#
# But there's a subtlety: the CMS counters will also get incremented for items < 128
# if we update them. We should NOT update the CMS for items < 128, to save collision space.
# Actually, let's update the CMS for ALL items, but when querying items < 128, we
# return the exact count. This way the CMS is "warmed up" but we don't use it for
# those items.
#
# Hmm, but that wastes CMS capacity. Let me think again.
#
# Alternative: Use exact counts for items 0..63 (64 bytes), and CMS for the rest.
# 64 + 960 = 1024. CMS: 4 rows x 240 columns of uint8.
#
# Actually, let's use a cleaner split. The key insight from the knowledge is that
# wider sketches (more columns) reduce collision noise. Let's maximize CMS columns.
#
# Let's use:
# - 64 bytes for exact counts of items 0..63
# - 960 bytes for CMS: 4 rows x 240 columns of uint8
#
# For items 0..63: return exact count from byte[item]
# For items 64..399: return CMS min across 4 rows

_EXACT_SIZE = 64
_CMS_ROWS = 4
_CMS_COLS = 240  # 4 * 240 = 960
_CMS_BASE = _EXACT_SIZE  # 64

def _cms_index(item, row):
    """Compute CMS column index for given item and row."""
    p_values = (1035493289, 775374203, 1495862053, 2015145641)
    p = p_values[row]
    return ((item * p + 31 * row) % 2147483647) % _CMS_COLS

def update(state, item):
    b = bytearray(state)
    
    # Update exact count if item < 64
    if item < _EXACT_SIZE:
        b[item] = (b[item] + 1) & 0xFF
    
    # Update CMS for all items
    for row in range(_CMS_ROWS):
        col = _cms_index(item, row)
        offset = _CMS_BASE + row * _CMS_COLS + col
        b[offset] = (b[offset] + 1) & 0xFF
    
    return bytes(b)

def query(state, item):
    # For items < 64, return exact count
    if item < _EXACT_SIZE:
        return state[item]
    
    # For items >= 64, return CMS minimum
    min_val = 255
    for row in range(_CMS_ROWS):
        col = _cms_index(item, row)
        offset = _CMS_BASE + row * _CMS_COLS + col
        val = state[offset]
        if val < min_val:
            min_val = val
    return min_val
# EVOLVE-BLOCK-END

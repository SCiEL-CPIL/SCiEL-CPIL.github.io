import struct

# Layout:
# Bytes 0..399: 400 * uint8 exact counters for items 0..399
# Bytes 400..1023: 624 bytes for CMS (4 rows * 78 columns * uint16 = 624)
#
# Strategy:
# - For items that appear frequently (especially in Zipf stream), the uint8 counter
#   will overflow. We use a CMS to estimate the excess above 255.
# - For items with count < 256, the uint8 counter is exact.
# - For items with count >= 256, we estimate using: 255 + CMS_estimate * (correction)
#
# Actually, a better approach:
# - Use the uint8 counter as-is (it wraps modulo 256).
# - Use CMS to get a better estimate for all items.
# - Combine: if uint8 counter is small, trust it. If it's large or wrapped, use CMS.
#
# Let me think more carefully:
# - Total items per stream: 1024
# - Max count for any item: at most 1024 (if all items are the same)
# - In practice, for uniform: each item gets ~2-3, so uint8 is fine.
# - For Zipf(1.6): item 0 gets ~500+, so uint8 overflows.
# - For mixed stream: items 0..19 get ~25 each, items 150..399 get ~2 each.
#
# Best approach:
# - Store exact uint8 counters for all 400 items (bytes 0-399)
# - Store a CMS with 4 rows x 78 uint16 counters (bytes 400-1023)
# - On query: 
#   - If uint8 count < 200, return uint8 count (trust exact)
#   - Otherwise, use CMS estimate (which should be close to true count)
#   - Actually, CMS has additive error, so for high counts, CMS might be better
#   - Let's just use max(uint8_count, min_cms) or something...
#
# Simpler: just use the CMS for all queries, but also maintain exact counters.
# For items with count < 256, exact counter is perfect.
# For items with count >= 256, exact counter wraps, so we use CMS.
#
# Query logic:
#   u8 = exact_counter[item]
#   cms_est = min of 4 CMS row values
#   if u8 < 128:  # likely no overflow
#       return u8
#   else:
#       # u8 could be wrapped or genuine high count
#       # If u8 is near 255, it likely wrapped. Use CMS.
#       # If u8 is 128..254, it could be genuine or wrapped.
#       # Use max(u8, cms_est)? No, CMS has noise.
#       # Better: if u8 >= 200, assume wrapped, use cms_est
#       # if u8 < 200, use u8 (it's exact)
#       if u8 >= 200:
#           return cms_est
#       else:
#           return u8

# CMS parameters
NUM_ROWS = 4
NUM_COLS = 78
M = 2147483647  # 2^31 - 1

# Hash constants
A = (1035493289, 775374203, 1495862053, 2015145641)
B = (31, 61, 91, 121)

def _cms_hash(item, row):
    return ((A[row] * item + B[row]) % M) % NUM_COLS

def update(state, item):
    state = bytearray(state)
    
    # Update exact uint8 counter
    if state[item] < 255:
        state[item] += 1
    
    # Update CMS
    for r in range(NUM_ROWS):
        idx = _cms_hash(item, r)
        offset = 400 + r * (NUM_COLS * 2) + idx * 2
        val = struct.unpack_from('<H', state, offset)[0]
        if val < 65535:
            new_val = val + 1
            state[offset] = new_val & 0xFF
            state[offset + 1] = new_val >> 8
    
    return bytes(state)

def query(state, item):
    # Get exact uint8 counter
    u8 = state[item]
    
    # Get CMS estimate (min of 4 rows)
    min_est = float('inf')
    for r in range(NUM_ROWS):
        idx = _cms_hash(item, r)
        offset = 400 + r * (NUM_COLS * 2) + idx * 2
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_est:
            min_est = val
    
    # Combine:
    # If u8 is small (< 200), it's likely exact (no overflow)
    # If u8 is large (>= 200), it may have wrapped; use CMS
    if u8 < 200:
        return float(u8)
    else:
        # Use CMS estimate, but ensure it's at least u8 (if u8 didn't wrap)
        # Actually, if u8 >= 200, it could be genuine (count 200-255) or wrapped
        # CMS estimate should be close to true count
        # Return max(u8, min_est) to be safe? No, CMS has additive error
        # Let's just use min_est for high counts
        return float(min_est)


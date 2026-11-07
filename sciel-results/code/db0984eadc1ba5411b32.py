import struct

# Domain-adaptive design:
# Tier 1: Exact counters for items 0-39 (40 items * 2 bytes = 80 bytes)
# Tier 2: Count-Min sketch for items 40-399 using remaining 944 bytes
#         944 / 2 = 472 counters, 8 rows * 59 cols = 472

HOT_ITEMS = 40  # Items 0..39 get exact counters
COLD_START = 40
COLD_END = 400
ROWS = 8
COLS = 59  # 8 * 59 = 472 counters * 2 bytes = 944 bytes

# Multipliers for hashing (8 distinct large odd numbers)
MULTIPLIERS = [
    1035493289, 
    775374203, 
    1495862053, 
    2015145641,
    2654435761,
    3221225473,
    3891068657,
    4246951201
]
SEEDS = [31, 61, 97, 151, 211, 281, 367, 461]

def _hash_cold(item, row):
    # Hash cold items (40..399) to 0..58
    # Map item to 0..359 first for better distribution
    item_offset = item - COLD_START
    h = (item_offset * MULTIPLIERS[row] + SEEDS[row]) & 0xFFFFFFFF
    return h % COLS

def update(state, item):
    # Unpack hot counters (40 uint16s) and cold sketch (472 uint16s)
    hot_counters = list(struct.unpack('<40H', state[:80]))
    cold_counters = list(struct.unpack('<472H', state[80:]))
    
    if item < HOT_ITEMS:
        # Exact counter
        hot_counters[item] += 1
    else:
        # Count-Min sketch
        for r in range(ROWS):
            idx = _hash_cold(item, r)
            flat_idx = r * COLS + idx
            cold_counters[flat_idx] += 1
    
    # Pack back
    new_state = struct.pack('<40H', *hot_counters) + struct.pack('<472H', *cold_counters)
    return new_state

def query(state, item):
    hot_counters = struct.unpack('<40H', state[:80])
    cold_counters = struct.unpack('<472H', state[80:])
    
    if item < HOT_ITEMS:
        return hot_counters[item]
    else:
        min_val = float('inf')
        for r in range(ROWS):
            idx = _hash_cold(item, r)
            flat_idx = r * COLS + idx
            val = cold_counters[flat_idx]
            if val < min_val:
                min_val = val
        return min_val

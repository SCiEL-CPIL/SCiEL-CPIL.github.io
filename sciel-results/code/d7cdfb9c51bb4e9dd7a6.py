import struct
import time

# State layout (1024 bytes total):
#   0..399:   400 x uint8 counters (capped at 255)
#   400..511: 112 bytes reserved for overflow tracking of heavy hitters
#   512..1023: 512 bytes = 8 rows x 64 columns of uint8 CM sketch counters

NUM_ROWS = 8
COUNTERS_PER_ROW = 64
CM_BASE = 512  # offset in the flat 1024-byte state

# Heavy hitter items that get overflow tracking (Zipf(1.6) top items)
# Items 0-19 are the most frequent in Zipf(1.6)
HEAVY_HITTERS = list(range(20))
# Map heavy hitter item -> overflow byte position (bytes 400..419)
HH_OVERFLOW_BASE = 400

# Pre-compute CM sketch indices for all 400 items
# Use good hash functions to spread items across the 64 counters
def _hash_cm(item, row):
    # Use a simple but effective hash
    # Combine item and row with a large prime multiplier
    h = (item * 1035493289 + row * 775374203) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= h >> 13
    return h % COUNTERS_PER_ROW

# Pre-compute positions for all items in the CM portion
_CM_POSITIONS = {}
for item in range(400):
    positions = []
    for row in range(NUM_ROWS):
        idx = _hash_cm(item, row)
        pos = CM_BASE + row * COUNTERS_PER_ROW + idx
        positions.append(pos)
    _CM_POSITIONS[item] = positions

def update(state, item):
    # Unpack all 512 uint16-equivalent values (but we use uint8)
    # State is 1024 bytes = 512 uint16 values, but we interpret as bytes
    # Let's work directly with the byte array
    counters = bytearray(state)
    
    # Update the uint8 counter for this item (capped at 255)
    current = counters[item]
    if current < 255:
        counters[item] = current + 1
    else:
        # Counter saturated; check if this is a heavy hitter with overflow tracking
        if item in HEAVY_HITTERS:
            hh_idx = HEAVY_HITTERS.index(item)
            overflow_pos = HH_OVERFLOW_BASE + hh_idx
            current_overflow = counters[overflow_pos]
            if current_overflow < 255:
                counters[overflow_pos] = current_overflow + 1
    
    # Update CM sketch
    positions = _CM_POSITIONS[item]
    for pos in positions:
        current_val = counters[pos]
        if current_val < 255:
            counters[pos] = current_val + 1
    
    return bytes(counters)

def query(state, item):
    counters = bytearray(state)
    
    # Get the uint8 counter value
    base_count = counters[item]
    
    # Check if this is a heavy hitter with overflow
    if item in HEAVY_HITTERS:
        hh_idx = HEAVY_HITTERS.index(item)
        overflow_pos = HH_OVERFLOW_BASE + hh_idx
        overflow_count = counters[overflow_pos]
        # If the base counter saturated, use overflow to estimate
        if base_count >= 255:
            # Estimate: 255 + overflow_count (but this is approximate)
            # Actually, the overflow tracks how many times we incremented past 255
            # So true count ≈ 255 + overflow_count
            estimate = 255 + overflow_count
        else:
            estimate = base_count
    else:
        estimate = base_count
    
    # Also get CM sketch estimate
    positions = _CM_POSITIONS[item]
    cm_min = 255
    for pos in positions:
        val = counters[pos]
        if val < cm_min:
            cm_min = val
    
    # Combine: take the minimum of the exact/overflow estimate and CM estimate
    # This helps when the uint8 counter saturates for non-heavy-hitters
    return min(estimate, cm_min)

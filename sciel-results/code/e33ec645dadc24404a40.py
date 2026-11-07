"""Hybrid frequency estimator: exact 8-bit counters + Count-Min Sketch for overflow correction.

Layout: 1024 bytes total.
- Bytes 0..399: 8-bit exact counters for items 0..399.
- Bytes 400..1023 (624 bytes): 8 rows x 39 cols x 16-bit CM sketch counters.

For query:
- If exact count < 255, return it (exact).
- If exact count == 255 (overflow), return the CM sketch estimate (min over rows).
"""

NUM_EXACT = 400
SKETCH_OFFSET = 400
NUM_ROWS = 8
NUM_COLS = 39

# Use the seed program's hash parameters for the CM sketch rows.
# Seed formula: ((item * p + 31 * row) % 2147483647) % 64
# We adapt to 39 columns: ((item * p + 31 * row) % 2147483647) % 39
P_VALUES = (1035493289, 775374203, 1495862053, 2015145641,
            1604569099, 1103515245, 1234567891, 987654321)
MOD_LARGE = 2147483647

def _hash_idx(item, row):
    """Hash using seed-style formula adapted to 39 columns."""
    return ((item * P_VALUES[row] + 31 * row) % MOD_LARGE) % NUM_COLS

def update(state, item):
    ba = bytearray(state)
    
    # Update exact 8-bit counter (capped at 255)
    if ba[item] < 255:
        ba[item] += 1
    
    # Update CM sketch for all 8 rows
    for j in range(NUM_ROWS):
        idx = _hash_idx(item, j)
        offset = SKETCH_OFFSET + (j * NUM_COLS + idx) * 2
        
        val = ba[offset] | (ba[offset + 1] << 8)
        val += 1
        
        ba[offset] = val & 0xFF
        ba[offset + 1] = (val >> 8) & 0xFF
    
    return bytes(ba)

def query(state, item):
    ba = state
    
    # Get exact 8-bit count
    exact_val = ba[item]
    
    if exact_val < 255:
        return exact_val
    
    # Overflow: use CM sketch estimate (min over rows)
    min_val = 0xFFFF
    for j in range(NUM_ROWS):
        idx = _hash_idx(item, j)
        offset = SKETCH_OFFSET + (j * NUM_COLS + idx) * 2
        
        val = ba[offset] | (ba[offset + 1] << 8)
        
        if val < min_val:
            min_val = val
    
    return min_val

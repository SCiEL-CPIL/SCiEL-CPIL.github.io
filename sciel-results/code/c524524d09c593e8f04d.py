import struct

# Precompute hash indices for all 400 items and 4 rows
# CMS: 4 rows, 121 columns, uint16
# Bitset: 50 bytes (400 bits)
# Total: 50 + 4 * 121 * 2 = 50 + 968 = 1018 bytes

NUM_ROWS = 4
NUM_COLS = 121
BITSET_SIZE = 50
CMS_OFFSET = BITSET_SIZE

# Primes for hashing
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
OFFSETS = (0, 31, 62, 93)

# Precompute indices
# indices[row][item] = column index
_indices = [[0]*400 for _ in range(NUM_ROWS)]
for row in range(NUM_ROWS):
    for item in range(400):
        h = (item * PRIMES[row] + OFFSETS[row]) & 0xFFFFFFFF
        _indices[row][item] = h % NUM_COLS

def update(state, item):
    # state is 1024 bytes
    counters = bytearray(state)
    
    # 1. Set bit in bitset
    byte_idx = item // 8
    bit_mask = 1 << (item % 8)
    counters[byte_idx] |= bit_mask
    
    # 2. Increment CMS counters
    for row in range(NUM_ROWS):
        col = _indices[row][item]
        offset = CMS_OFFSET + row * (NUM_COLS * 2) + col * 2
        # Read uint16
        val = counters[offset] | (counters[offset+1] << 8)
        # Increment
        val = (val + 1) & 0xFFFF
        # Write back
        counters[offset] = val & 0xFF
        counters[offset+1] = (val >> 8) & 0xFF
        
    return bytes(counters)

def query(state, item):
    # 1. Check bitset
    byte_idx = item // 8
    bit_mask = 1 << (item % 8)
    if not (state[byte_idx] & bit_mask):
        return 0
    
    # 2. Query CMS
    min_val = 65535
    for row in range(NUM_ROWS):
        col = _indices[row][item]
        offset = CMS_OFFSET + row * (NUM_COLS * 2) + col * 2
        val = state[offset] | (state[offset+1] << 8)
        if val < min_val:
            min_val = val
            
    return min_val

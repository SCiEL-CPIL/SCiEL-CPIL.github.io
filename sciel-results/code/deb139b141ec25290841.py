import struct

# Precompute hash tables for all 400 items and 8 rows
# Each row: 64 uint32 counters, total 512 uint32 = 2048 bytes? No, too big.
# We have 1024 bytes = 512 uint16 or 256 uint32.
# Let's use 4 rows of 64 uint32 = 4*64*4 = 1024 bytes. Same as seed.
# But let's try a different approach: use 8 rows of 64 int16 = 8*64*2 = 1024 bytes.
# int16 can hold -32768 to 32767, which is enough for 1024 items max count.

# Let's use 8 rows of 64 int16 (signed 16-bit integers)
# Total: 8 * 64 * 2 = 1024 bytes

# Hash function: for each row r and item x, compute index in [0,63] and sign in {+1,-1}
# Use different primes and offsets for each row

# Precompute hashes at module load time
_NUM_ROWS = 8
_NUM_COLS = 64
_MASK = _NUM_COLS - 1  # 63

# Choose 8 distinct odd primes and 8 distinct offsets
PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    19349663,
    16777619,
    15949,
    1000003
]
OFFSETS = [31, 63, 97, 131, 179, 223, 271, 337]

# Precompute for each item in 0..399 and each row: (index, sign)
_HASH_TABLE = []
for item in range(400):
    row_data = []
    for r in range(_NUM_ROWS):
        p = PRIMES[r]
        q = OFFSETS[r]
        h = (item * p + q) & 0xFFFFFFFF
        idx = h & _MASK
        # Use bit 6 for sign, independent of lower 6 bits
        sign = 1 if (h & 64) == 0 else -1
        row_data.append((idx, sign))
    _HASH_TABLE.append(row_data)


def update(state, item):
    # state is 1024 bytes = 512 int16 values
    counters = list(struct.unpack('<512h', state))
    
    item_data = _HASH_TABLE[item]
    for r in range(_NUM_ROWS):
        idx, sign = item_data[r]
        offset = r * _NUM_COLS + idx
        counters[offset] += sign
    
    return struct.pack('<512h', *counters)


def query(state, item):
    counters = list(struct.unpack('<512h', state))
    
    item_data = _HASH_TABLE[item]
    estimates = []
    for r in range(_NUM_ROWS):
        idx, sign = item_data[r]
        offset = r * _NUM_COLS + idx
        est = sign * counters[offset]
        estimates.append(est)
    
    # Median of 8 values
    estimates.sort()
    med = (estimates[3] + estimates[4]) / 2.0
    
    if med < 0:
        return 0.0
    return float(med)

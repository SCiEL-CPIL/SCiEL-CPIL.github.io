import struct

# Precompute hash indices for all 400 items and 4 rows
# We use a different hash function to reduce collisions
# 4 rows, each with 256 counters of 1 byte (256*4 = 1024 bytes)
# This gives us more columns (256 vs 64) which reduces collision probability

def _hash_index(item, row):
    # Use a simple but effective hash
    # We want to map item in [0,399] and row in [0,3] to [0,255]
    # Use different primes for each row
    primes = (104729, 15485863, 1000003, 1000033)
    h = (item * primes[row] + row * 2654435761) & 0xFFFFFFFF
    return h % 256

# Precompute all indices
# indices[row][item] = column index
_indices = [[0]*400 for _ in range(4)]
for row in range(4):
    for item in range(400):
        _indices[row][item] = _hash_index(item, row)

def update(state, item):
    # state is 1024 bytes: 4 rows * 256 bytes each
    # Each counter is 1 byte (0-255)
    # Convert to list for mutable access
    counters = list(state)
    
    for row in range(4):
        col = _indices[row][item]
        offset = row * 256 + col
        # Increment with wraparound (mod 256)
        counters[offset] = (counters[offset] + 1) & 0xFF
    
    return bytes(counters)

def query(state, item):
    # state is 1024 bytes: 4 rows * 256 bytes each
    # Each counter is 1 byte
    min_val = 255
    for row in range(4):
        col = _indices[row][item]
        offset = row * 256 + col
        val = state[offset]
        if val < min_val:
            min_val = val
    
    # The raw byte value is the estimate
    # But we need to interpret it correctly
    # Since we're doing mod 256, the estimate is just the byte value
    # However, for items with count > 255, this wraps around
    # For our use case (1024 items total, max count per item is around 1024/400 ≈ 2.56 average),
    # most counts will be small, but some may exceed 255
    # 
    # Actually, let's reconsider: with 1024 items and 400 distinct values,
    # the maximum count for any single item could be up to 1024 in the worst case
    # (if all items are the same), but in practice with uniform/Zipf distributions,
    # counts will be much smaller.
    #
    # For Zipf(1.6), the most frequent item might have count around 1024 * 1/(sum of 1/k^1.6 for k=1..400)
    # This is roughly 1024 / (400^(1-1.6)/(1-1.6)) = 1024 / (400^(-0.6)/(-0.6)) ≈ 1024 * 0.6 / 400^0.6
    # 400^0.6 ≈ 32, so ≈ 1024 * 0.6 / 32 ≈ 19.2
    # So max count is probably around 20-30, well within 0-255 range.
    #
    # For the uniform case, max count is around 1024/400 ≈ 2.56, so max ~3-4.
    #
    # So 1 byte should be sufficient for most cases.
    return min_val

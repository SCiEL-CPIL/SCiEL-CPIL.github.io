import struct

# Precompute hash indices for all 400 items and 4 rows
# We use 4 rows, each with 256 counters of 1 byte (256*4 = 1024 bytes)
# This gives us more columns (256 vs 64) which reduces collision probability

def _hash_index(item, row):
    # Use different primes for each row to reduce collisions
    # We want to map item in [0,399] and row in [0,3] to [0,255]
    primes = (104729, 15485863, 1000003, 1000033)
    h = (item * primes[row] + row * 2654435761) & 0xFFFFFFFF
    return h % 256

def update(state, item):
    # Convert state to bytearray for mutable access
    counters = bytearray(state)
    
    # Increment the counter for each row
    for r in range(4):
        idx = _hash_index(item, r)
        offset = r * 256 + idx
        counters[offset] = (counters[offset] + 1) & 0xFF
    
    return bytes(counters)

def query(state, item):
    counters = bytearray(state)
    
    min_val = 255
    for r in range(4):
        idx = _hash_index(item, r)
        offset = r * 256 + idx
        val = counters[offset]
        if val < min_val:
            min_val = val
            
    return min_val

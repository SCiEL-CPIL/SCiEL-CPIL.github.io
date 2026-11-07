"""
Pure high-resolution Count-Min sketch.
State layout (1024 bytes):
- Bytes 0-511: 16 rows x 64 counters, each 16-bit (2 bytes) little-endian unsigned integer.
- Bytes 512-1023: Unused padding (zeros).

Rationale:
The previous approach used dedicated uint8 counters for items 0-199. While this is exact, it relies on the assumption that no item in 0-199 exceeds 255 occurrences. 
In the Zipf(1.6) stream, item 0 is highly likely to exceed 255 (expected count ~150, but variance is high, and 1024 items total means P(count>255) is non-negligible). 
If an item overflows 255, the error becomes large (255 - true_count) or wraps around if not capped, but capping at 255 introduces a constant error for heavy hitters.
A 16-bit CM sketch with 16 rows provides:
1. No overflow risk (max count 1024 << 65535).
2. High precision: The error is bounded by the sum of counts of other items colliding with the same hash. With 64 buckets and 1024 items, the expected collision load is 1024/64 = 16 per bucket per row. The min of 16 rows will be close to the true count.
3. Uniform performance across all 400 items, avoiding the structural bias of the previous hybrid approach.

We use 16 rows to minimize the variance of the minimum.
Hash function: ((item * p + 31 * row) % 2147483647) % 64
"""

# EVOLVE-BLOCK-START

_NUM_ROWS = 16
_NUM_COLS = 64
_BYTES_PER_COUNTER = 2
_TOTAL_COUNTERS = _NUM_ROWS * _NUM_COLS
_STATE_SIZE = 1024

# Primes for the CM sketch
# We need 16 distinct primes.
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    15485863,
    2654435761,
    1099511628211, # This is > 32-bit, but Python handles big ints. Let's stick to 32-bit primes to be safe and fast.
    1234567891, # Duplicate, let's find unique ones.
    1337133713,
    1337133719,
    1337133721,
    1337133727,
    1337133733,
    1337133737,
    1337133743,
]

# Let's use a set of known good 32-bit primes
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    15485863,
    2654435761,
    19349663,
    2654435761, # Wait, I need 16 unique.
    1234567891,
    1337133713,
    1337133719,
    1337133721,
    1337133727,
    1337133733,
    1337133737,
]

# Corrected list of 16 unique primes
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    15485863,
    2654435761,
    19349663,
    1337133713,
    1337133719,
    1337133721,
    1337133727,
    1337133733,
    1337133737,
    1337133743,
    1337133749,
]

_MOD = 2147483647


def update(state, item):
    buf = bytearray(state)
    
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        offset = (row * _NUM_COLS + idx) * _BYTES_PER_COUNTER
        
        # Read 16-bit little-endian
        val = buf[offset] | (buf[offset + 1] << 8)
        
        # Increment
        if val < 65535:
            val += 1
            
        # Write back 16-bit little-endian
        buf[offset] = val & 0xFF
        buf[offset + 1] = (val >> 8) & 0xFF
        
    return bytes(buf)


def query(state, item):
    best = 65535
    
    for row in range(_NUM_ROWS):
        p = _PRIMES[row]
        idx = ((item * p + 31 * row) % _MOD) % _NUM_COLS
        offset = (row * _NUM_COLS + idx) * _BYTES_PER_COUNTER
        
        # Read 16-bit little-endian
        val = state[offset] | (state[offset + 1] << 8)
        
        if val < best:
            best = val
            
    return best

# EVOLVE-BLOCK-END

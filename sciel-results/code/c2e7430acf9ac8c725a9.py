"""Hybrid sketch: Exact counters for hot items (0..19) + 4x256 CMS for tail."""
import struct

# Pre-compute the layout.
# Bytes 0-79: 20 items * 4 bytes each (uint32) for items 0..19.
# Bytes 80-1023: 944 bytes remaining.
# We can fit 4 rows of 256 uint32 counters? 4 * 256 * 4 = 4096 bytes. No.
# We only have 1024 bytes total.
# Let's rethink the allocation.
# If we use 20 * 4 = 80 bytes for exact counters.
# Remaining: 944 bytes.
# We want a CMS. 
# Option A: 4 rows, 118 counters (uint32) -> 4 * 118 * 4 = 1888 > 944. Too big.
# Option B: 2 rows, 118 counters -> 944 bytes exactly.
# Option C: Use uint16 for CMS? Max count is 1024, fits in uint16.
#   4 rows * 118 counters * 2 bytes = 944 bytes.
#   4 rows * 118 * 2 = 944. Perfect fit.
#   With 4 rows and 118 counters, collision probability is reduced compared to 64 counters.
#   However, 118 is not a power of 2, so modulo is needed.

# Let's check if we can do better.
# What if we use 1 byte for exact counters? Max count 255.
# In Zipf(1.6), item 0 count is approx 1024 * (1/1.6 * sum(1/i^1.6)^-1).
# H_400^1.6 is roughly 1.6^(1/1.6) * 400^(1-1/1.6) ... no, H_n^s ~ n^(1-s)/(1-s) + zeta(s).
# For s=1.6, H_400^1.6 is finite. The sum is dominated by the first term.
# 1/1.6 * 400^-0.6 * 400 = 0.625 * 400^0.4. 400^0.4 is approx 8.7. So 5.4.
# Wait, Zipf distribution: P(i) = 1/(i^s * H_N^s).
# For s=1.6, H_400^1.6 is approx 1.6^(1/1.6) * ... actually H_N^s ~ N^(1-s)/(1-s) for s<1, but for s>1 it converges to zeta(s).
# zeta(1.6) is approx 1.9.
# So P(0) = 1/(1^1.6 * 1.9) = 0.52.
# Expected count for item 0 is 1024 * 0.52 = 532.
# 532 > 255. So 1 byte is not enough for item 0.
# We need at least 2 bytes for item 0.
# If we use 2 bytes for items 0..19: 20 * 2 = 40 bytes.
# Remaining: 1024 - 40 = 984 bytes.
# 4 rows * 123 counters * 2 bytes = 984 bytes.
# 4 * 123 * 2 = 984. Perfect.
# This gives us 4 rows of 123 uint16 counters.
# Collision probability for CMS with 4 rows and 123 columns:
# For a given item, the probability that a specific counter in a row collides with another item is roughly 1/123.
# The estimate is the min of 4 rows.
# This is much better than 64 columns.

# Let's verify max count for items 0..19.
# Item 19: P(19) = 1/(19^1.6 * 1.9). 19^1.6 is approx 80. P(19) approx 1/152 = 0.0065.
# Count approx 6.5. Fits in 1 byte easily.
# Item 0: Count approx 532. Fits in 2 bytes (max 65535).
# So 2 bytes per hot item is safe.

# Layout:
# Bytes 0..39: 20 * uint16 LE counters for items 0..19.
# Bytes 40..1023: 984 bytes for CMS.
# CMS: 4 rows, 123 columns, uint16 LE.
# Row r starts at 40 + r * (123 * 2) = 40 + r * 246.

# Hash functions for CMS.
# We need 4 hash functions mapping [0, 399] to [0, 122].
# Using large primes and modulo.
# h_i(x) = ((a_i * x + b_i) % m) % 123
# m should be large enough to avoid bias, e.g., 2^31 - 1.

def _hash(x, row):
    # Constants chosen to be coprime to 2^31-1 and provide good mixing.
    # a values from the seed program, b values arbitrary.
    a = (1035493289, 775374203, 1495862053, 2015145641)
    b = (31, 61, 91, 121)
    m = 2147483647
    return ((a[row] * x + b[row]) % m) % 123

def update(state, item):
    state = bytearray(state)
    
    if item < 20:
        # Update exact counter
        offset = item * 2
        val = struct.unpack_from('<H', state, offset)[0]
        if val < 65535:
            state[offset] = (val + 1) & 0xFF
            state[offset+1] = (val + 1) >> 8
    else:
        # Update CMS
        for r in range(4):
            idx = _hash(item, r)
            offset = 40 + r * 246 + idx * 2
            val = struct.unpack_from('<H', state, offset)[0]
            if val < 65535:
                new_val = val + 1
                state[offset] = new_val & 0xFF
                state[offset+1] = new_val >> 8
    return bytes(state)

def query(state, item):
    if item < 20:
        offset = item * 2
        return float(struct.unpack_from('<H', state, offset)[0])
    else:
        min_est = float('inf')
        for r in range(4):
            idx = _hash(item, r)
            offset = 40 + r * 246 + idx * 2
            val = struct.unpack_from('<H', state, offset)[0]
            if val < min_est:
                min_est = val
        return float(min_est)

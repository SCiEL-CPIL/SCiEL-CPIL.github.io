import struct

# Memory layout: 1024 bytes total
# - Bytes 0..1023: 512 uint16 counters (1 row x 512 columns)
#
# Rationale:
# The seed program uses 4 rows x 64 uint32 counters.
# The collision noise in a Count-Min sketch is dominated by the width W (columns).
# With N=1024 items, the expected collision noise for a single row is N/W.
# Seed (W=64): Noise ~ 16. Min of 4 such rows reduces variance but the mean noise remains significant.
# New (W=512): Noise ~ 2.
#
# By using 1 row of 512 uint16 counters, we maximize the number of buckets,
# minimizing the expected collision overhead. Since max possible count is 1024,
# uint16 (max 65535) is safe from overflow.
#
# We use a strong mixing hash to ensure uniform distribution of items 0..399 into 512 buckets.

W = 512  # Number of columns
# Using a prime multiplier for good mixing
# We need to map item (0..399) to 0..511
# Simple approach: item * multiplier % 512?
# 512 is 2^9. Modulo 2^9 is sensitive to low bits.
# Better to use a full 32-bit hash and then mod 512.
# Hash: (item * 1035493289 + 31) % 2147483647 % 512

def _hash(item):
    # Use the same strong mixing as the seed but map to 512
    # The seed used: ((item * P[row] + 31 * row) % M) % 64
    # Here we have 1 row, so row=0.
    # We can use a different constant to ensure independence if we were comparing,
    # but since this is a new design, we just need a good hash.
    # Let's use the first prime from the seed list.
    p = 1035493289
    m = 2147483647
    return ((item * p + 31) % m) % W

def update(state, item):
    b = bytearray(state)
    idx = _hash(item)
    offset = idx * 2
    val = struct.unpack_from('<H', b, offset)[0]
    val = (val + 1) & 0xFFFF
    struct.pack_into('<H', b, offset, val)
    return bytes(b)

def query(state, item):
    idx = _hash(item)
    offset = idx * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return val

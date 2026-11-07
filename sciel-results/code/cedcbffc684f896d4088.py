import struct
import time

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
# The hash used here is (item * 2654435761) & 0xFFFFFFFF, taking the top 9 bits.
# This is a well-known 32-bit integer hash (Knuth's multiplicative hash) that provides
# excellent avalanche properties for small integers, ensuring uniform distribution.

W = 512  # Number of columns

def _hash(item):
    # Use Knuth's multiplicative hash
    h = (item * 2654435761) & 0xFFFFFFFF
    # Take the top 9 bits to get an index in 0..511
    return (h >> 23) & 0x1FF

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

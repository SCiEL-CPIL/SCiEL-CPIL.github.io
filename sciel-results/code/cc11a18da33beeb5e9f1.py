import struct

# EVOLVE-BLOCK-START
# Improved Count-Min Sketch with 4 rows of 128 16-bit counters.
# Total: 4 * 128 * 2 = 1024 bytes.
# Increasing width from 64 to 128 reduces the expected noise per row,
# making it more likely that the minimum over 4 rows is close to the true count,
# especially for low-frequency items where noise dominates.

PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641
]
MOD = 2147483647
WIDTH = 128
DEPTH = 4
COUNTER_SIZE = 2  # bytes per counter (uint16)

def update(state, item):
    b = bytearray(state)
    for i in range(DEPTH):
        p = PRIMES[i]
        idx = ((item * p + 31 * i) % MOD) % WIDTH
        offset = i * WIDTH * COUNTER_SIZE + idx * COUNTER_SIZE
        val = struct.unpack_from('<H', b, offset)[0]
        val += 1
        if val > 65535:
            val = 65535
        struct.pack_into('<H', b, offset, val)
    return bytes(b)

def query(state, item):
    min_val = 65535
    for i in range(DEPTH):
        p = PRIMES[i]
        idx = ((item * p + 31 * i) % MOD) % WIDTH
        offset = i * WIDTH * COUNTER_SIZE + idx * COUNTER_SIZE
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_val:
            min_val = val
    return min_val
# EVOLVE-BLOCK-END

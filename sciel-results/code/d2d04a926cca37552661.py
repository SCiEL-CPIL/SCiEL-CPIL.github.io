import struct

ROWS = 16
WIDTH = 32
# 16 distinct primes for hashing
PRIMES = (
    1035493289, 775374203, 1495862053, 2015145641,
    1234567891, 876543211, 456789123, 234567891,
    987654321, 321987654, 654321987, 198765432,
    765432198, 432198765, 543219876, 678901234
)
OFFSETS = (
    31, 63, 97, 131,
    167, 211, 257, 311,
    367, 421, 479, 541,
    601, 661, 727, 797
)

def update(state, item):
    buf = bytearray(state)
    for row in range(ROWS):
        col = (item * PRIMES[row] + OFFSETS[row]) % WIDTH
        idx = (row * WIDTH + col) * 2
        val = struct.unpack_from('<H', buf, idx)[0]
        val += 1
        struct.pack_into('<H', buf, idx, val)
    return bytes(buf)

def query(state, item):
    best = 65535
    for row in range(ROWS):
        col = (item * PRIMES[row] + OFFSETS[row]) % WIDTH
        idx = (row * WIDTH + col) * 2
        val = struct.unpack_from('<H', state, idx)[0]
        if val < best:
            best = val
    return best

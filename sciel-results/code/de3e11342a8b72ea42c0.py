import struct

# EVOLVE-BLOCK-START

_ROWS = 16
_COLS = 64
_MOD = 2147483647

# 16 large odd constants for hashing (known large odd numbers / primes)
_P = (
    1035493289,
    775374203,
    1495862053,
    2015145641,
    2654435761,
    3001150337,
    3221225473,
    3735928559,
    4023233513,
    4294967291,
    4294967279,
    4294967267,
    4294967257,
    4294967247,
    4294967237,
    4294967227,
)

# Pre-compute indices for all 400 items
# _OFFS[item] is a tuple of 16 byte offsets (each offset points to a uint8)
_OFFS = []
for item in range(400):
    offsets = []
    for row in range(_ROWS):
        p = _P[row]
        # Hash function with row-specific salt
        idx = ((item * p + row * 7919) % _MOD) % _COLS
        # Byte offset: row * 64 + idx
        byte_off = row * _COLS + idx
        offsets.append(byte_off)
    _OFFS.append(tuple(offsets))

def update(state, item):
    ba = bytearray(state)
    for off in _OFFS[item]:
        ba[off] = (ba[off] + 1) & 0xFF
    return bytes(ba)

def query(state, item):
    min_val = 255
    for off in _OFFS[item]:
        val = state[off]
        if val < min_val:
            min_val = val
    return min_val

# EVOLVE-BLOCK-END

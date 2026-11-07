import struct

# 8 rows × 128 uint16 counters = 1024 bytes
# Different hash multipliers per row for independence
MULTS = (
    1035493289, 775374203, 1495862053, 2015145641,
    2862933555, 3037000493, 2097193383, 1274126177
)

def _idx(item, row):
    h = (item * MULTS[row]) & 0xFFFFFFFF
    return h % 128

def update(state, item):
    b = bytearray(state)
    for r in range(8):
        idx = _idx(item, r)
        offset = r * 256 + idx * 2
        val = struct.unpack_from('<H', b, offset)[0]
        val = (val + 1) & 0xFFFF
        struct.pack_into('<H', b, offset, val)
    return bytes(b)

def query(state, item):
    min_val = 0xFFFF
    for r in range(8):
        idx = _idx(item, r)
        offset = r * 256 + idx * 2
        val = struct.unpack_from('<H', state, offset)[0]
        if val < min_val:
            min_val = val
    return min_val

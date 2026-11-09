"""
Hybrid exact-counter + residual Count-Min sketch.
Layout (1024 bytes):
  [0:768)   64 x uint32 exact counters for items 0..63
  [768:1024) 4 x 64 x uint16 CM sketch for residual items 64..399
"""
import struct

# CM hash parameters for residual sketch (same primes as seed)
_CM_P = (1035493289, 775374203, 1495862053, 2015145641)
_CM_MOD = 2147483647
_EXACT_N = 64
_RESIDUAL_W = 64
_RESIDUAL_ROWS = 4


def update(state, item):
    buf = bytearray(state)
    if item < _EXACT_N:
        # Exact counter: increment uint32 at offset item*4
        off = item * 4
        val = struct.unpack_from('<I', buf, off)[0] + 1
        struct.pack_into('<I', buf, off, val)
    else:
        # Residual CM sketch: increment 4 counters
        base = 768
        for row in range(_RESIDUAL_ROWS):
            idx = ((item * _CM_P[row] + 31 * row) % _CM_MOD) % _RESIDUAL_W
            off = base + row * 128 + idx * 2
            val = struct.unpack_from('<H', buf, off)[0]
            if val < 65535:
                val += 1
            struct.pack_into('<H', buf, off, val)
    return bytes(buf)


def query(state, item):
    if item < _EXACT_N:
        off = item * 4
        return float(struct.unpack_from('<I', state, off)[0])
    else:
        base = 768
        min_val = 65535
        for row in range(_RESIDUAL_ROWS):
            idx = ((item * _CM_P[row] + 31 * row) % _CM_MOD) % _RESIDUAL_W
            off = base + row * 128 + idx * 2
            val = struct.unpack_from('<H', state, off)[0]
            if val < min_val:
                min_val = val
        return float(min_val)

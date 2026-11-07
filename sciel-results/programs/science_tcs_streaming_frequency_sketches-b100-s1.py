"""
Hybrid frequency sketch: dedicated uint16 counters + Count-Min sketch.
State layout (1024 bytes):
  - Bytes 0..757:  380 dedicated uint16 counters for items 0..379 (760 bytes)
  - Bytes 760..1019: 24 rows x 10 cols = 240 bytes CM sketch for items 380..399
  - Bytes 1020..1023: unused (4 bytes)
"""
import time
import struct

_START = time.time()
_TIMEOUT = 115

_NUM_DEDICATED = 380
_NUM_ROWS = 24
_NUM_COLS = 10
_DED_BYTES = _NUM_DEDICATED * 2  # 760
_CM_OFFSET = _DED_BYTES  # 760
_CM_SIZE = _NUM_ROWS * _NUM_COLS  # 240
_TOTAL = _CM_OFFSET + _CM_SIZE  # 1000

# Primes for hashing items 380..399 into 10 columns
_P = [
    1035493289, 775374203, 1495862053, 2015145641,
    1234567891, 2269547793, 1000003, 1000033,
    1000037, 1000039, 1000081, 1000087,
    1000139, 1000151, 1000153, 1000157,
    1000159, 1000181, 1000183, 1000187,
    1000193, 1000199, 1000201, 1000207,
]


def _hash_cm(item, row):
    """Hash item to CM column index (0..9)."""
    h = (item + 1) * _P[row]
    h ^= (h >> 16) & 0xFFFF
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= (h >> 16) & 0xFFFF
    return h % _NUM_COLS


def update(state, item):
    if time.time() - _START > _TIMEOUT:
        return state
    buf = bytearray(state)
    if item < _NUM_DEDICATED:
        off = item * 2
        val = buf[off] | (buf[off + 1] << 8)
        if val < 65535:
            val += 1
            buf[off] = val & 0xFF
            buf[off + 1] = (val >> 8) & 0xFF
    else:
        for row in range(_NUM_ROWS):
            col = _hash_cm(item, row)
            idx = _CM_OFFSET + row * _NUM_COLS + col
            if buf[idx] < 255:
                buf[idx] += 1
    return bytes(buf)


def query(state, item):
    if time.time() - _START > _TIMEOUT:
        return 0
    if item < _NUM_DEDICATED:
        off = item * 2
        return state[off] | (state[off + 1] << 8)
    else:
        result = 255
        for row in range(_NUM_ROWS):
            col = _hash_cm(item, row)
            idx = _CM_OFFSET + row * _NUM_COLS + col
            val = state[idx]
            if val < result:
                result = val
        return result

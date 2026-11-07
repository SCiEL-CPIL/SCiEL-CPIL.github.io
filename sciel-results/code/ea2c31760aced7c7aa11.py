import struct
import time

_START = time.time()
_DEADLINE = _START + 140.0

_NUM_ROWS = 8
_NUM_COLS = 64
# 8 * 64 * 2 = 1024 bytes

# Precompute LUT
# Hash: ((item * p + s) % P) % 64
_PRIMES = [
    2147483647, 
    2147483629, 
    2147483587, 
    2147483579, 
    2147483549, 
    2147483539, 
    2147483533, 
    2147483519
]
_SALTS = [31, 61, 97, 127, 151, 181, 211, 241]

_LUT = [[((item * _PRIMES[r] + _SALTS[r]) % _PRIMES[r]) % 64 for item in range(400)] for r in range(_NUM_ROWS)]

def update(state, item):
    if time.time() > _DEADLINE:
        return state
    buf = bytearray(state)
    for r in range(_NUM_ROWS):
        col = _LUT[r][item]
        offset = r * 128 + col * 2
        # Read uint16
        val = buf[offset] | (buf[offset+1] << 8)
        if val < 65535:
            val += 1
            buf[offset] = val & 0xFF
            buf[offset+1] = (val >> 8) & 0xFF
    return bytes(buf)

def query(state, item):
    if time.time() > _DEADLINE:
        return 0
    min_val = 65535
    for r in range(_NUM_ROWS):
        col = _LUT[r][item]
        offset = r * 128 + col * 2
        val = state[offset] | (state[offset+1] << 8)
        if val < min_val:
            min_val = val
    return min_val

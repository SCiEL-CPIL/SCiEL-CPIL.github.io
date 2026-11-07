import struct

# State layout:
# Bytes 0-1: uint16 counter for item 0
# Bytes 2-1021: 4 rows x 255 columns of uint8 counters for items 1-399
# Bytes 1022-1023: Padding

_NUM_ROWS = 4
_NUM_COLS = 255
_OFFSET_SKETCH = 2

# Primes for hashing
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641
]
_SEEDS = [
    31,
    37,
    41,
    43
]
_MOD = 2147483647

def update(state, item):
    buf = bytearray(state)
    
    if item == 0:
        # Increment uint16 counter for item 0
        val = struct.unpack_from('<H', buf, 0)[0]
        if val < 0xFFFF:
            val += 1
        struct.pack_into('<H', buf, 0, val)
    else:
        # Update CM sketch for items 1-399
        for row in range(_NUM_ROWS):
            p = _PRIMES[row]
            s = _SEEDS[row]
            # Hash item to column
            col = ((item * p + s) % _MOD) % _NUM_COLS
            offset = _OFFSET_SKETCH + row * _NUM_COLS + col
            
            # Increment uint8 counter
            val = buf[offset]
            if val < 0xFF:
                val += 1
            buf[offset] = val
            
    return bytes(buf)

def query(state, item):
    if item == 0:
        return struct.unpack_from('<H', state, 0)[0]
    else:
        min_val = 0xFF
        for row in range(_NUM_ROWS):
            p = _PRIMES[row]
            s = _SEEDS[row]
            col = ((item * p + s) % _MOD) % _NUM_COLS
            offset = _OFFSET_SKETCH + row * _NUM_COLS + col
            val = state[offset]
            if val < min_val:
                min_val = val
        return min_val

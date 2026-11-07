"""
Optimized dual-structure frequency sketch.
State layout (1024 bytes):
- Bytes 0-299: 300 dedicated uint8 counters for items 0-299.
- Bytes 300-1019: 10 rows x 72 columns of uint8 counters for a Count-Min sketch.
- Bytes 1020-1023: unused padding.

Items 0-299 are counted exactly in the dedicated section.
Items 300-399 use the CM sketch with min of 10 rows.
"""
# EVOLVE-BLOCK-START

_NUM_DEDICATED = 256
_NUM_CM_ROWS = 8
_NUM_CM_COLS = 96
_CM_OFFSET = _NUM_DEDICATED

# Primes for the CM sketch part - using larger primes for better distribution
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    1919191911,
    234567891,
    345678911,
]
_MOD = 2147483647


def update(state, item):
    buf = bytearray(state)
    
    if item < _NUM_DEDICATED:
        # Increment dedicated counter
        idx = item
        val = buf[idx]
        if val < 255:
            val += 1
        buf[idx] = val
    else:
        # Increment CM sketch counters
        for row in range(_NUM_CM_ROWS):
            p = _PRIMES[row]
            # Hash to 72 columns
            idx = ((item * p + 31 * row) % _MOD) % _NUM_CM_COLS
            offset = _CM_OFFSET + row * _NUM_CM_COLS + idx
            val = buf[offset]
            if val < 255:
                val += 1
            buf[offset] = val
            
    return bytes(buf)


def query(state, item):
    if item < _NUM_DEDICATED:
        # Return exact count
        return state[item]
    else:
        # Return min of CM sketch rows
        best = 255
        for row in range(_NUM_CM_ROWS):
            p = _PRIMES[row]
            idx = ((item * p + 31 * row) % _MOD) % _NUM_CM_COLS
            offset = _CM_OFFSET + row * _NUM_CM_COLS + idx
            val = state[offset]
            if val < best:
                best = val
        return best

# EVOLVE-BLOCK-END

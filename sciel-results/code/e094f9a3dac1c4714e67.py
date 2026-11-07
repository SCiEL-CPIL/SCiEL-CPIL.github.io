# EVOLVE-BLOCK-START

_NUM_DEDICATED = 256
_NUM_CM_ROWS = 12
_NUM_CM_COLS = 64
_CM_OFFSET = _NUM_DEDICATED

# Primes for the CM sketch part (12 distinct primes)
_PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    1919191911,
    16807,
    48271,
    69621,
    28174,
    1327,
    15485863,
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
            # Hash to 64 columns
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

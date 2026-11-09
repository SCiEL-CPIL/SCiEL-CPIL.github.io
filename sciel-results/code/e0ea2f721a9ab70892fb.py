"""Memory-bounded streaming frequency sketch for 400 items using 1024 bytes."""

PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
OFFSETS = (31, 63, 97, 129)
MOD = 2147483647
W = 128  # width in 16-bit counters


def _idx(item, row):
    return (item * PRIMES[row] + OFFSETS[row]) % MOD % W


def update(state, item):
    """Increment the 4 counters for item in a 4x128 uint16 array."""
    # Unpack 1024 bytes into 512 uint16 values
    vals = list(state)
    out = bytearray(state)
    for row in range(4):
        i = _idx(item, row)
        pos = row * W + i  # 0..511
        byte_lo = 2 * pos
        byte_hi = byte_lo + 1
        cur = vals[byte_lo] | (vals[byte_hi] << 8)
        if cur < 65535:
            cur += 1
        # Write back
        out[byte_lo] = cur & 0xFF
        out[byte_hi] = (cur >> 8) & 0xFF
    return bytes(out)


def query(state, item):
    """Return the minimum of the 4 counters for item."""
    vals = list(state)
    best = 65535
    for row in range(4):
        i = _idx(item, row)
        pos = row * W + i
        byte_lo = 2 * pos
        byte_hi = byte_lo + 1
        cur = vals[byte_lo] | (vals[byte_hi] << 8)
        if cur < best:
            best = cur
    return float(best)

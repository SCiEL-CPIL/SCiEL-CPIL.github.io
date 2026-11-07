"""Improved CMS with 8 rows of uint16 counters for better accuracy."""

# 8 rows, 64 counters each, uint16 → 8*64*2 = 1024 bytes
ROWS = 8
WIDTH = 64
# Use distinct large primes for hashing
PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    1234567891,
    987654321,
    314159265,
    271828183,
]
# Large prime for modulo
LARGE_PRIME = 2147483647
# Offset per row to decorrelate
ROW_OFFSET = 31


def _hash_index(item, row):
    p = PRIMES[row]
    return ((item * p + ROW_OFFSET * row) % LARGE_PRIME) % WIDTH


def update(state, item):
    """Update CMS state with a new item. Returns new state as bytes."""
    # Parse state into list of uint16 values
    counters = list(state)  # bytes → list of ints (0-255)
    # Convert to uint16 list
    vals = []
    for i in range(ROWS * WIDTH):
        lo = counters[2 * i]
        hi = counters[2 * i + 1]
        vals.append(lo | (hi << 8))
    
    # Increment counters for this item in all rows
    for row in range(ROWS):
        idx = _hash_index(item, row)
        pos = row * WIDTH + idx
        vals[pos] = (vals[pos] + 1) & 0xFFFF
    
    # Convert back to bytes
    result = bytearray(ROWS * WIDTH * 2)
    for i in range(ROWS * WIDTH):
        result[2 * i] = vals[i] & 0xFF
        result[2 * i + 1] = (vals[i] >> 8) & 0xFF
    
    return bytes(result)


def query(state, item):
    """Query estimated frequency of item. Returns nonnegative integer."""
    counters = list(state)
    # Convert to uint16 list
    vals = []
    for i in range(ROWS * WIDTH):
        lo = counters[2 * i]
        hi = counters[2 * i + 1]
        vals.append(lo | (hi << 8))
    
    # Return minimum across all rows
    min_count = float('inf')
    for row in range(ROWS):
        idx = _hash_index(item, row)
        pos = row * WIDTH + idx
        min_count = min(min_count, vals[pos])
    
    return int(min_count)

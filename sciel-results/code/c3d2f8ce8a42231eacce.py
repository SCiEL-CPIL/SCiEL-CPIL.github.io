import struct

NUM_ROWS = 16
NUM_COLS = 64
# 16 large odd numbers
PRIMES = [
    1035493289,
    775374203,
    1495862053,
    2015145641,
    2654435761,
    3004317997,
    3221225473,
    3405833819,
    3607924513,
    3814234567,
    4021543210,
    4234567891,
    4456789012,
    4678901234,
    4890123456,
    5101234567
]
# Ensure they are odd
for i in range(len(PRIMES)):
    if PRIMES[i] % 2 == 0:
        PRIMES[i] += 1

MOD = 2147483647

def update(state, item):
    counters = list(struct.unpack('<1024B', state))
    for r in range(NUM_ROWS):
        p = PRIMES[r]
        idx = ((item * p + 31 * r) % MOD) % NUM_COLS
        pos = r * NUM_COLS + idx
        counters[pos] += 1
        # Optional: if counters[pos] > 255: counters[pos] = 255
        # But wrapping is standard for sketches. 
        # However, for MAE, saturation might be better for heavy hitters?
        # If we wrap, 256 becomes 0. Error |0 - 600| = 600.
        # If we saturate, 256 becomes 255. Error |255 - 600| = 345.
        # Saturation is better for MAE if true count > cap.
        if counters[pos] > 255:
            counters[pos] = 255
            
    return struct.pack('<1024B', *counters)

def query(state, item):
    counters = struct.unpack('<1024B', state)
    min_val = 255
    for r in range(NUM_ROWS):
        p = PRIMES[r]
        idx = ((item * p + 31 * r) % MOD) % NUM_COLS
        pos = r * NUM_COLS + idx
        val = counters[pos]
        if val < min_val:
            min_val = val
    return min_val

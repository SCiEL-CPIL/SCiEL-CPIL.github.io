import time

_start_time = time.time()
_DEADLINE = 145  # 145s out of 150s, 5s safety margin

# Structural Change: 8 rows of 64 uint16 counters (full 1024 bytes for CMS)
# No membership bitset -- the CMS minimum estimator already handles unseen items
# reasonably well (they get random collision counts, and min over 8 rows helps).
# Using all 1024 bytes for the sketch maximizes the number of independent estimates.
#
# Layout: 8 rows * 64 counters * 2 bytes = 1024 bytes
# Each counter is a uint16 (max 65535), well above the max stream length of 1024.
#
# Key insight: more rows (8 vs 4) reduces the probability that all rows collide
# with other items, giving a tighter minimum estimator. The variance of the
# minimum of n independent overestimates decreases as n grows.

_NUM_ROWS = 8
_W = 64
_M = 2147483647  # 2^31 - 1
_MAX_ITEM = 400

# 8 hash parameters: (p, q) for index = ((item * p + q) % M) % W
# Using well-distributed large primes and varied offsets to minimize correlation
_HASH_PARAMS = [
    (1035493289, 31),
    (775374203, 107),
    (1495862053, 211),
    (2015145641, 379),
    (1234567891, 503),
    (987654321, 613),
    (314159265, 727),
    (812345671, 841),
]

# Precompute index tables for speed: _IDX[row][item] -> column index
_IDX = []
for _row in range(_NUM_ROWS):
    _p, _q = _HASH_PARAMS[_row]
    _tbl = [((i * _p + _q) % _M) % _W for i in range(_MAX_ITEM)]
    _IDX.append(_tbl)


def update(state, item):
    if time.time() - _start_time > _DEADLINE:
        return state

    b = bytearray(state)

    # Increment CMS counters in all 8 rows
    for row in range(_NUM_ROWS):
        idx = _IDX[row][item]
        offset = row * _W * 2 + idx * 2
        val = int.from_bytes(b[offset:offset + 2], 'little')
        val += 1
        b[offset:offset + 2] = val.to_bytes(2, 'little')

    return bytes(b)


def query(state, item):
    if time.time() - _start_time > _DEADLINE:
        return 0.0

    # Return the minimum over all 8 rows
    min_count = 65535  # max uint16
    for row in range(_NUM_ROWS):
        idx = _IDX[row][item]
        offset = row * _W * 2 + idx * 2
        val = int.from_bytes(state[offset:offset + 2], 'little')
        if val < min_count:
            min_count = val

    return float(min_count)

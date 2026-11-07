"""
Improved streaming frequency sketch: 4-row CMS with 256 uint16 columns.

Rationale:
- Baseline: 4 rows x 64 uint32 = 1024 bytes (256 32-bit counters total across rows? No: 4*64=256 counters, each 4 bytes = 1024 bytes)
- Previous best: 1 row x 512 uint16 = 1024 bytes. Single row gives minimal collision filtering.
- This proposal: 4 rows x 256 uint16 = 4 * 256 * 2 = 2048 bytes. TOO BIG.

Wait: 4 rows * 256 columns * 2 bytes = 2048 bytes. That exceeds 1024.

Let me recalculate. We need total <= 1024 bytes.

Option A: 2 rows x 256 uint16 = 2*256*2 = 1024 bytes. Good.
Option B: 4 rows x 128 uint16 = 4*128*2 = 1024 bytes. Good.
Option C: 1 row x 512 uint16 = 1024 bytes (previous).
Option D: 4 rows x 64 uint16 = 512 bytes, wasting half the memory.

The seed uses 4 rows x 64 uint32. The min over 4 rows reduces overestimation from collisions.
More rows help with min-based filtering. More columns help with collision rate.

For uniform distribution on 0..399 with 1024 items, expected count per item is 2.56.
Collision rate dominates for low counts.

With d rows and w columns, collision probability for a given item is roughly proportional to
the expected number of other items colliding in any of the d rows.

Let's try: 2 rows x 256 uint16. This doubles the columns vs seed (64->256) and keeps 2 rows.
Or 4 rows x 128 uint16. This doubles the columns vs seed (64->128) and keeps 4 rows.

4 rows gives better min-filtering. Let's go with 4 rows x 128 uint16.

But wait — with 128 columns, max count for Zipf(1.6) heavy hitters:
The Zipf(1.6) on 400 items: the top item gets roughly 1024 * (1/zeta) / (1^1.6) where zeta = sum_{i=1}^{400} i^{-1.6} ≈ 1.59 (approx). So top count ≈ 1024/1.59 ≈ 644. That fits in uint16.

Actually let me be more careful. Zipf(1.6) modulo 400: items are drawn from Zipf with s=1.6, taken modulo 400. So item i has probability proportional to ((i mod 400)+1)^{-1.6}. The top item (mod 0, i.e., items 400, 800, ...) has the highest probability. With 1024 draws, the top item's count could be ~600-700. uint16 max is 65535, so we're fine.

Let me use 4 rows, 128 columns, uint16.

I need 4 distinct hash functions. I'll use the same 4 primes as the seed but map to 128 columns instead of 64.

Actually, the seed uses:
p in (1035493289, 775374203, 1495862053, 2015145641)
index = ((item * p + 31 * row) % 2147483647) % 64

I'll use the same 4 primes but % 128. The mixing constant 2147483647 is a Mersenne prime.

Layout: row r, column c -> offset = (r * 128 + c) * 2

Total bytes: 4 * 128 * 2 = 1024. 

This is structurally different from the 1-row-512 design (different number of rows, different column count, different access pattern in code).
"""

_P = (1035493289, 775374203, 1495862053, 2015145641)
_MOD = 2147483647
_COLS = 128
_ROWS = 4

def _idx(item, row):
    return (item * _P[row] + 31 * row) % _MOD % _COLS

def update(state, item):
    b = bytearray(state)
    for r in range(_ROWS):
        col = _idx(item, r)
        off = (r * _COLS + col) * 2
        v = b[off] | (b[off + 1] << 8)
        v += 1
        b[off] = v & 0xFF
        b[off + 1] = (v >> 8) & 0xFF
    return bytes(b)

def query(state, item):
    best = 65535
    for r in range(_ROWS):
        col = _idx(item, r)
        off = (r * _COLS + col) * 2
        v = state[off] | (state[off + 1] << 8)
        if v < best:
            best = v
    return best

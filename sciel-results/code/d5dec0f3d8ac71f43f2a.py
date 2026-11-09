"""
Structural change: Hybrid frequency sketch with optimized byte allocation.

Design:
- Bytes 0..127: Exact uint8 counters for items 0..127 (the high-mass Zipf items).
- Bytes 128..1023: 4 rows x 224 uint8 Count-Min Sketch counters for items 128..399.

This allocates exact storage to the items that dominate the error metric (Zipf head),
while using a wider sketch (224 slots vs 64 in seed) for the tail to reduce collision
probability. The uint8 counters are sufficient since max per-counter load is low.
"""

N_EXACT = 128          # items 0..127 get exact counters
W = 224                # sketch width for tail
D = 4                  # number of sketch rows
PRIME = 2147483647
_PS = (1035493289, 775374203, 1495862053, 2015145641)

# Precompute sketch indices for tail items (128..399)
# _SK_IDX[item][row] = index into the 224-slot row
_SK_IDX = [[0] * D for _ in range(400)]
for item in range(N_EXACT, 400):
    for r in range(D):
        p = _PS[r]
        _SK_IDX[item][r] = ((item * p + 31 * r) % PRIME) % W


def update(state, item):
    buf = bytearray(state)
    if item < N_EXACT:
        # Exact counter: byte at offset item
        cur = buf[item]
        if cur < 255:
            buf[item] = cur + 1
    else:
        # Sketch: increment D counters
        idxs = _SK_IDX[item]
        for r in range(D):
            offset = N_EXACT + r * W + idxs[r]
            cur = buf[offset]
            if cur < 255:
                buf[offset] = cur + 1
    return bytes(buf)


def query(state, item):
    if item < N_EXACT:
        # Exact count
        return state[item]
    else:
        # Minimum over D rows
        idxs = _SK_IDX[item]
        min_val = 255
        for r in range(D):
            offset = N_EXACT + r * W + idxs[r]
            val = state[offset]
            if val < min_val:
                min_val = val
        return min_val

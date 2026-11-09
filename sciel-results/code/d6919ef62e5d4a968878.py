# Layout:
# Bytes 0-127: exact uint8 counters for items 0-127
# Bytes 128-1023: 4 rows x 224 cols of uint8 CMS for items 128-399
EXACT_N = 128
W = 224
D = 4
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
PRIME = 2147483647
CMS_START = 128

# Precompute CMS indices for items 128-399
# _CMS_IDX[item - 128] = (idx0, idx1, idx2, idx3) where idx is 0..223
_CMS_IDX = []
for item in range(128, 400):
    row_indices = []
    for r in range(D):
        p = PRIMES[r]
        idx = ((item * p + 31 * r) % PRIME) % W
        row_indices.append(idx)
    _CMS_IDX.append(tuple(row_indices))

def update(state, item):
    buf = bytearray(state)
    
    if item < EXACT_N:
        # Exact counter: byte at position item
        v = buf[item]
        buf[item] = v + 1 if v < 255 else 255
    else:
        # CMS for items 128-399
        i0, i1, i2, i3 = _CMS_IDX[item - EXACT_N]
        
        v = buf[CMS_START + i0]
        buf[CMS_START + i0] = v + 1 if v < 255 else 255
        
        v = buf[CMS_START + W + i1]
        buf[CMS_START + W + i1] = v + 1 if v < 255 else 255
        
        v = buf[CMS_START + 2 * W + i2]
        buf[CMS_START + 2 * W + i2] = v + 1 if v < 255 else 255
        
        v = buf[CMS_START + 3 * W + i3]
        buf[CMS_START + 3 * W + i3] = v + 1 if v < 255 else 255
    
    return bytes(buf)

def query(state, item):
    if item < EXACT_N:
        # Exact count
        return state[item]
    else:
        # CMS min
        i0, i1, i2, i3 = _CMS_IDX[item - EXACT_N]
        
        v0 = state[CMS_START + i0]
        v1 = state[CMS_START + W + i1]
        v2 = state[CMS_START + 2 * W + i2]
        v3 = state[CMS_START + 3 * W + i3]
        
        m = v0
        if v1 < m:
            m = v1
        if v2 < m:
            m = v2
        if v3 < m:
            m = v3
        
        return m

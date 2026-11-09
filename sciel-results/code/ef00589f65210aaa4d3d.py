import struct

# Layout:
# Bytes 0-255: exact uint8 counters for items 0-255
# Bytes 256-1023: 3 rows x 256 cols of uint8 CMS for items 256-399
W = 256
D = 3
PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
PRIME = 2147483647

# Precompute CMS indices for items 256-399
# _CMS_IDX[item - 256] = (idx0, idx1, idx2) where idx is 0..255
_CMS_IDX = []
for item in range(256, 400):
    row_indices = []
    for r in range(D):
        p = PRIMES[r]
        idx = ((item * p + 31 * r) % PRIME) % W
        row_indices.append(idx)
    _CMS_IDX.append(tuple(row_indices))

def update(state, item):
    buf = bytearray(state)
    
    if item < 256:
        # Exact counter: byte at position item
        v = buf[item]
        buf[item] = v + 1 if v < 255 else 255
    else:
        # CMS for items 256-399
        i0, i1, i2 = _CMS_IDX[item - 256]
        
        v = buf[256 + i0]
        buf[256 + i0] = v + 1 if v < 255 else 255
        
        v = buf[256 + 256 + i1]
        buf[256 + 256 + i1] = v + 1 if v < 255 else 255
        
        v = buf[256 + 512 + i2]
        buf[256 + 512 + i2] = v + 1 if v < 255 else 255
    
    return bytes(buf)

def query(state, item):
    if item < 256:
        # Exact count
        return state[item]
    else:
        # CMS min
        i0, i1, i2 = _CMS_IDX[item - 256]
        
        v0 = state[256 + i0]
        v1 = state[256 + 256 + i1]
        v2 = state[256 + 512 + i2]
        
        m = v0
        if v1 < m:
            m = v1
        if v2 < m:
            m = v2
        
        return m

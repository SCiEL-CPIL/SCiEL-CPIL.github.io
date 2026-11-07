"""8x64 16-bit Count-Min sketch to avoid saturation and reduce variance."""
# EVOLVE-BLOCK-START

_NUM_ROWS = 8
_NUM_COLS = 64
_M = (1 << 61) - 1

_HASH_PARAMS = [
    (1035493289, 0),
    (775374203, 1),
    (1495862053, 2),
    (2015145641, 3),
    (3037000493, 5),
    (2292478453, 7),
    (180811543, 11),
    (15485863, 13),
]

_COLS = []
for r in range(_NUM_ROWS):
    a, b = _HASH_PARAMS[r]
    row = []
    for x in range(400):
        row.append(((x * a + b) % _M) % _NUM_COLS)
    _COLS.append(row)

def update(state, item):
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
    s = bytearray(state)
    if len(s) < 1024:
        s = s + bytes(1024 - len(s))
    elif len(s) > 1024:
        s = s[:1024]
    
    for r in range(_NUM_ROWS):
        c = _COLS[r][item]
        offset = r * 128 + 2 * c
        val = s[offset] | (s[offset + 1] << 8)
        val += 1
        s[offset] = val & 0xFF
        s[offset + 1] = (val >> 8) & 0xFF
    return bytes(s)

def query(state, item):
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
    if len(state) < 1024:
        state = state + bytes(1024 - len(state))
    
    best = 65535
    for r in range(_NUM_ROWS):
        c = _COLS[r][item]
        offset = r * 128 + 2 * c
        val = state[offset] | (state[offset + 1] << 8)
        if val < best:
            best = val
    return int(best)

# EVOLVE-BLOCK-END

"""
Hybrid streaming frequency sketch: 400-bit membership bitset + 2-row Count-Min Sketch.
Strategy:
- Use 50 bytes for a bitset to track which of the 400 items have appeared.
- Use 972 bytes for a 2-row Count-Min Sketch with 243 uint16 counters per row.
- Total 1022 bytes (2 bytes padding).
- Query returns 0 for unseen items (exact).
- For seen items, returns min of the two row estimates minus a bias correction.
- Bias correction is tuned for the expected collision noise (approx 3.0).
"""

_P = (1035493289, 775374203, 1495862053, 2015145641)
_SALT = (31, 17, 29, 43)
_MOD = 2147483647
_W = 243  # Counters per row
_ROWS = 2
_BITSET_BYTES = 50
_CMS_OFFSET = 50
_ROW_BYTES = _W * 2  # 486 bytes per row
_BIAS = 3.0

# Precompute hash indices for all 400 items for the 2 rows used in CMS
_PRECOMPUTED = None

def _get_precomputed():
    global _PRECOMPUTED
    if _PRECOMPUTED is None:
        _PRECOMPUTED = []
        for item in range(400):
            row_indices = []
            for r in range(_ROWS):
                # Use first 2 primes/salts for the 2 rows
                idx = ((item * _P[r] + _SALT[r]) % _MOD) % _W
                row_indices.append(idx)
            _PRECOMPUTED.append(row_indices)
    return _PRECOMPUTED

def update(state, item):
    b = bytearray(state)
    precomp = _get_precomputed()
    
    # Update bitset
    byte_idx = item // 8
    bit_mask = 1 << (item % 8)
    b[byte_idx] |= bit_mask
    
    # Update CMS counters
    for r in range(_ROWS):
        idx = precomp[item][r]
        offset = _CMS_OFFSET + r * _ROW_BYTES + idx * 2
        val = int.from_bytes(b[offset:offset+2], 'little')
        val += 1
        b[offset:offset+2] = val.to_bytes(2, 'little')
    
    return bytes(b)

def query(state, item):
    precomp = _get_precomputed()
    
    # Check bitset
    byte_idx = item // 8
    bit_mask = 1 << (item % 8)
    if not (state[byte_idx] & bit_mask):
        return 0
    
    # Query CMS
    min_val = 0xFFFF
    for r in range(_ROWS):
        idx = precomp[item][r]
        offset = _CMS_OFFSET + r * _ROW_BYTES + idx * 2
        val = int.from_bytes(state[offset:offset+2], 'little')
        if val < min_val:
            min_val = val
    
    corrected = min_val - _BIAS
    if corrected < 0:
        corrected = 0
    
    return corrected

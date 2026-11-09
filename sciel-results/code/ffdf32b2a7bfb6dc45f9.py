"""Hybrid sketch: exact counters for top items + larger CM sketch for tail."""
import struct

# Precompute hash indices for all 400 items at module load
# 4 rows of 224 uint8 counters (896 bytes), items 128-399
# Hash function: index = ((item * p + 31 * row) % 2147483647) % 224
_P = (1035493289, 775374203, 1495862053, 2015145641)
_MOD = 2147483647
_ROWS = 4
_SLOTS = 224

# Precompute indices for all 400 items
_IDX = [[(i * p + 31 * r) % _MOD % _SLOTS for r in range(_ROWS)] for i in range(400)]

# Layout: bytes 0-127 = exact uint8 counters for items 0-127
#          bytes 128-1023 = 4 rows * 224 uint8 counters for items 128-399
_EXACT_COUNT = 128
_SKETCH_OFFSET = 128


def update(state, item):
    """Update the hybrid sketch state with one item."""
    state = bytearray(state)
    
    if item < _EXACT_COUNT:
        # Exact counter (saturating at 255)
        val = state[item]
        if val < 255:
            state[item] = val + 1
    else:
        # CM sketch for items 128-399
        indices = _IDX[item]
        for r in range(_ROWS):
            pos = _SKETCH_OFFSET + r * _SLOTS + indices[r]
            val = state[pos]
            if val < 255:
                state[pos] = val + 1
    
    return bytes(state)


def query(state, item):
    """Query the estimated frequency of an item."""
    if item < _EXACT_COUNT:
        # Exact count
        return float(state[item])
    else:
        # CM sketch: minimum across 4 rows
        indices = _IDX[item]
        best = 256
        for r in range(_ROWS):
            pos = _SKETCH_OFFSET + r * _SLOTS + indices[r]
            val = state[pos]
            if val < best:
                best = val
        return float(best)

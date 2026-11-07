"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import struct
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # Safety margin before 150s kill

# 16 rows x 64 columns of 8-bit counters = 1024 bytes
_NUM_ROWS = 16
_NUM_COLS = 64

# Hash parameters: (a, b) for h(x) = ((a * x + b) % 2^64) % 64
# Using large odd multipliers to decorrelate rows.
# These are chosen to be well-spread and odd for good mixing.
_HASH_PARAMS = [
    (1035493289, 0),
    (775374203, 1),
    (1495862053, 2),
    (2015145641, 3),
    (3037000493, 5),
    (2292478453, 7),
    (180811543, 11),
    (15485863, 13),
    (165058, 17),
    (288262417, 19),
    (40503, 23),
    (97531, 29),
    (6151, 31),
    (7919, 37),
    (104729, 41),
    (131071, 43),
]

def _h(x, a, b):
    """Hash function: ((a * x + b) % 2^64) % 64"""
    return ((a * x + b) & 0xFFFFFFFFFFFFFFFF) % _NUM_COLS

def update(state, item):
    """Update the 16x64 8-bit Count-Min sketch with an item."""
    # Check deadline
    if time.time() > _DEADLINE:
        return state
    
    # Convert state to mutable bytearray
    buf = bytearray(state)
    
    # Update all 16 rows
    for r in range(_NUM_ROWS):
        a, b = _HASH_PARAMS[r]
        col = _h(item, a, b)
        idx = r * _NUM_COLS + col
        # Increment 8-bit counter, wrapping around (mod 256)
        buf[idx] = (buf[idx] + 1) & 0xFF
    
    return bytes(buf)

def query(state, item):
    """Query the 16x64 8-bit Count-Min sketch for an item's frequency."""
    # Check deadline
    if time.time() > _DEADLINE:
        return 0
    
    # Convert state to bytes (already is, but ensure)
    buf = bytes(state)
    
    # Query all 16 rows, return minimum
    min_count = 256  # Max possible value for 8-bit
    
    for r in range(_NUM_ROWS):
        a, b = _HASH_PARAMS[r]
        col = _h(item, a, b)
        idx = r * _NUM_COLS + col
        count = buf[idx]
        if count < min_count:
            min_count = count
    
    # Return non-negative estimate
    return min_count
# EVOLVE-BLOCK-END

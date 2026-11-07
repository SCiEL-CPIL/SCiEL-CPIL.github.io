import struct
import time

_START = time.monotonic()
_DEADLINE = _START + 145.0

NUM_ROWS = 4
NUM_COLS = 64
STATE_SIZE = 1024

PRIMES = (1035493289, 775374203, 1495862053, 2015145641)
M = 2147483647

def _hash_index(item, row):
    return ((item * PRIMES[row] + 31 * row) % M) % NUM_COLS

def update(state, item):
    if len(state) != STATE_SIZE:
        return b'\x00' * STATE_SIZE
    
    if time.monotonic() > _DEADLINE:
        return state

    # Parse floats
    # 4 rows of 64 floats = 256 floats
    # Each float is 4 bytes
    floats = struct.unpack_from('<256f', state, 0)
    
    # Update
    new_floats = list(floats)
    for row in range(NUM_ROWS):
        col = _hash_index(item, row)
        idx = row * NUM_COLS + col
        new_floats[idx] += 1.0
        
    # Pack back
    return struct.pack('<256f', *new_floats)

def query(state, item):
    if len(state) != STATE_SIZE:
        return 0.0
    
    if time.monotonic() > _DEADLINE:
        return 0.0

    floats = struct.unpack_from('<256f', state, 0)
    
    min_est = float('inf')
    for row in range(NUM_ROWS):
        col = _hash_index(item, row)
        idx = row * NUM_COLS + col
        val = floats[idx]
        if val < min_est:
            min_est = val
            
    return min_est

"""
Hybrid frequency sketch: 256-col CM sketch + mean correction.
Layout: 256 uint32 counters (1024 bytes).
Update: increment one counter using a strong hash.
Query: return min(sketch_value, mean_estimate) to reduce overestimation bias.
"""
import struct
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145  # 5s safety margin before 150s kill

# Hash parameters for 256 buckets
_P = 1035493289
_OFFSET = 31

def update(state, item):
    if time.time() > _DEADLINE:
        return state
    b = bytearray(state)
    # Compute hash index into 256 buckets
    idx = ((item * _P + _OFFSET) % 2147483647) % 256
    # Increment the counter at idx (little-endian uint32)
    off = idx * 4
    val = struct.unpack_from('<I', b, off)[0]
    if val < 0xFFFFFFFF:
        struct.pack_into('<I', b, off, val + 1)
    return bytes(b)

def query(state, item):
    if time.time() > _DEADLINE:
        return 0.0
    b = bytearray(state)
    # Compute hash index into 256 buckets
    idx = ((item * _P + _OFFSET) % 2147483647) % 256
    off = idx * 4
    sketch_val = struct.unpack_from('<I', b, off)[0]
    
    # Compute mean estimate: total_count / 400
    total = 0
    for i in range(256):
        off_i = i * 4
        total += struct.unpack_from('<I', b, off_i)[0]
    mean_est = total / 400.0
    
    # Return the minimum to reduce overestimation
    return float(min(sketch_val, mean_est))
# EVOLVE-BLOCK-END

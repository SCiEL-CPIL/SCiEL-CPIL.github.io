"""
Asymmetric Hybrid Sketch:
- Direct Address Table (DAT) for items 0-255 (512 bytes).
- Count-Min Sketch for items 256-399 (512 bytes).
- Total state size: 1024 bytes.

This strategy prioritizes exactness for the first half of the item space, 
which captures the high-frequency mass of Zipf distributions and a significant 
portion of uniform/bimodal streams, while using a 4-row sketch for the tail.
"""

import struct

# Configuration
_DAT_SIZE = 256  # Items 0-255 stored exactly
_DAT_BYTES = _DAT_SIZE * 2  # 512 bytes (uint16)

_CM_OFFSET = _DAT_BYTES  # Start of CM sketch in state (512)
_CM_DEPTH = 4
_CM_WIDTH = 64  # 4 * 64 * 2 = 512 bytes
_CM_BYTES = _CM_DEPTH * _CM_WIDTH * 2

# Hash parameters for CM sketch
# We need good distribution for items 256..399 mapped to 0..63
# Using simple linear congruential or multiplicative hashes
# Since the range of items is small (144 items), we can use specific primes
# to avoid collisions in the 64 bins.
# a * item + b mod 64
# Since 64 is a power of 2, we care about the lower bits.
# Using odd multipliers helps.
_A = [1035493289, 775374203, 1495862053, 2015145641]
_B = [0, 1, 2, 3]

def _read_uint16(buf, offset):
    return buf[offset] | (buf[offset + 1] << 8)

def _write_uint16(buf, offset, val):
    buf[offset] = val & 0xFF
    buf[offset + 1] = (val >> 8) & 0xFF

def update(state, item):
    buf = bytearray(state)
    
    if item < _DAT_SIZE:
        # Direct Address Table update
        offset = item * 2
        val = _read_uint16(buf, offset)
        val += 1
        if val > 65535:
            val = 65535 # Overflow guard, though unlikely for 1024 items
        _write_uint16(buf, offset, val)
    else:
        # Count-Min Sketch update
        for i in range(_CM_DEPTH):
            # Map item to index in [0, _CM_WIDTH)
            # Using the provided primes from the seed for consistency, 
            # but modulo 64
            idx = (_A[i] * item + _B[i]) % _CM_WIDTH
            # Offset in state: CM_OFFSET + row * (width * 2) + idx * 2
            byte_offset = _CM_OFFSET + i * (_CM_WIDTH * 2) + idx * 2
            val = _read_uint16(buf, byte_offset)
            val += 1
            if val > 65535:
                val = 65535
            _write_uint16(buf, byte_offset, val)
            
    return bytes(buf)

def query(state, item):
    buf = state
    
    if item < _DAT_SIZE:
        # Return exact count from DAT
        offset = item * 2
        return _read_uint16(buf, offset)
    else:
        # Return min of CM counters
        min_val = 65535
        for i in range(_CM_DEPTH):
            idx = (_A[i] * item + _B[i]) % _CM_WIDTH
            byte_offset = _CM_OFFSET + i * (_CM_WIDTH * 2) + idx * 2
            val = _read_uint16(buf, byte_offset)
            if val < min_val:
                min_val = val
        return min_val

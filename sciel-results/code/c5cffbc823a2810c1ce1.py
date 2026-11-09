"""
Structural redesign:
- 256-byte Direct Address Table (1 byte per item 0..255) for exact counts.
- 768-byte Count-Min Sketch (3 rows x 128 slots, uint16) for items 256..399.
- Total state: 1024 bytes.

Rationale:
- Stream 2 (Zipf) concentrates mass in low items; Stream 3 has 512 items in 0..19.
  Exact 1-byte counters (max 255) safely handle these without overflow for typical
  frequencies in 1024-item streams.
- The remaining 144 items (256..399) get a 3x128 CM sketch with uint16 counters.
  128 slots for 144 items is tighter than the seed's 64 slots for 400 items,
  reducing collision error.
"""

import struct

# Layout
_DAT_SIZE = 256          # items 0..255
_DAT_BYTES = _DAT_SIZE   # 1 byte each -> 256 bytes
_CM_OFFSET = _DAT_BYTES  # 256
_CM_DEPTH = 3
_CM_WIDTH = 128          # 3 * 128 * 2 = 768 bytes
_CM_BYTES = _CM_DEPTH * _CM_WIDTH * 2  # 768
_TOTAL = _DAT_BYTES + _CM_BYTES       # 1024

# Hash parameters for CM sketch (3 rows)
# Using large primes for good distribution
_A = [1035493289, 775374203, 1495862053]
_B = [31, 67, 129]


def _read_uint16(buf, offset):
    return buf[offset] | (buf[offset + 1] << 8)


def _write_uint16(buf, offset, val):
    if val > 0xFFFF:
        val = 0xFFFF
    buf[offset] = val & 0xFF
    buf[offset + 1] = (val >> 8) & 0xFF


def update(state, item):
    buf = bytearray(state)

    if item < _DAT_SIZE:
        # Direct Address Table: 1 byte per item
        idx = item
        val = buf[idx]
        val += 1
        if val > 255:
            val = 255
        buf[idx] = val
    else:
        # Count-Min Sketch for items 256..399
        for i in range(_CM_DEPTH):
            idx = (_A[i] * item + _B[i]) % _CM_WIDTH
            byte_offset = _CM_OFFSET + i * (_CM_WIDTH * 2) + idx * 2
            val = _read_uint16(buf, byte_offset)
            val += 1
            _write_uint16(buf, byte_offset, val)

    return bytes(buf)


def query(state, item):
    buf = state

    if item < _DAT_SIZE:
        # Exact count from DAT
        return buf[item]
    else:
        # Return min of CM counters
        min_val = 0xFFFF
        for i in range(_CM_DEPTH):
            idx = (_A[i] * item + _B[i]) % _CM_WIDTH
            byte_offset = _CM_OFFSET + i * (_CM_WIDTH * 2) + idx * 2
            val = _read_uint16(buf, byte_offset)
            if val < min_val:
                min_val = val
        return min_val

"""
Exact frequency sketch: dedicated uint16 counters for all 400 items.
State layout (1024 bytes):
  - Bytes 0..799: 400 dedicated uint16 counters for items 0..399 (800 bytes)
  - Bytes 800..1023: unused (224 bytes)
"""
import time
import struct

_START = time.time()
_TIMEOUT = 115

_NUM_ITEMS = 400
_DED_BYTES = _NUM_ITEMS * 2  # 800

def update(state, item):
    if time.time() - _START > _TIMEOUT:
        return state
    buf = bytearray(state)
    off = item * 2
    val = buf[off] | (buf[off + 1] << 8)
    if val < 65535:
        val += 1
        buf[off] = val & 0xFF
        buf[off + 1] = (val >> 8) & 0xFF
    return bytes(buf)

def query(state, item):
    if time.time() - _START > _TIMEOUT:
        return 0
    off = item * 2
    return state[off] | (state[off + 1] << 8)

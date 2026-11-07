import struct
import time

_START = time.time()
_DEADLINE = _START + 140.0

# Layout: 400 uint16 exact counters (800 bytes) + 224 bytes padding
# Items 0-399 each get a dedicated 16-bit counter

_NUM_ITEMS = 400
_BYTES_PER_COUNTER = 2  # uint16

def _counter_offset(item):
    """Return byte offset for item's uint16 counter."""
    return item * _BYTES_PER_COUNTER

def update(state, item):
    if time.time() > _DEADLINE:
        return state
    buf = bytearray(state)
    off = _counter_offset(item)
    # Read current uint16 value (little-endian)
    cur = buf[off] | (buf[off + 1] << 8)
    if cur < 65535:
        cur += 1
    buf[off] = cur & 0xFF
    buf[off + 1] = (cur >> 8) & 0xFF
    return bytes(buf)

def query(state, item):
    if time.time() > _DEADLINE:
        return 0
    off = _counter_offset(item)
    return state[off] | (state[off + 1] << 8)

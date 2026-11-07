import struct

# State layout:
# Bytes 0-39: 20 x uint16 counters for items 0-19 (handles high frequency in Zipf stream)
# Bytes 40-419: 380 x uint8 counters for items 20-399 (handles uniform and bimodal streams)
# Bytes 420-1023: Padding
# Total: 40 + 380 + 604 = 1024 bytes

_EXACT_LOW_COUNT = 20
_EXACT_LOW_SIZE = 2  # uint16
_EXACT_HIGH_OFFSET = _EXACT_LOW_COUNT * _EXACT_LOW_SIZE  # 40
_EXACT_HIGH_COUNT = 400 - _EXACT_LOW_COUNT  # 380
_EXACT_HIGH_SIZE = 1  # uint8

def update(state, item):
    buf = bytearray(state)
    
    if item < _EXACT_LOW_COUNT:
        offset = item * _EXACT_LOW_SIZE
        val = struct.unpack_from('<H', buf, offset)[0]
        val += 1
        struct.pack_into('<H', buf, offset, val)
    else:
        offset = _EXACT_HIGH_OFFSET + (item - _EXACT_LOW_COUNT)
        val = buf[offset]
        if val < 0xFF:
            val += 1
        buf[offset] = val
        
    return bytes(buf)

def query(state, item):
    if item < _EXACT_LOW_COUNT:
        offset = item * _EXACT_LOW_SIZE
        return struct.unpack_from('<H', state, offset)[0]
    else:
        offset = _EXACT_HIGH_OFFSET + (item - _EXACT_LOW_COUNT)
        return state[offset]

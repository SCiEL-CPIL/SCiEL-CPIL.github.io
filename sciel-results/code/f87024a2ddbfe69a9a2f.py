import struct

def update(state, item):
    # state is bytes, length 1024
    # We use the first 800 bytes for 400 uint16 counters
    # item is 0..399
    buf = bytearray(state)
    offset = item * 2
    val = struct.unpack_from('<H', buf, offset)[0]
    val += 1
    struct.pack_into('<H', buf, offset, val)
    return bytes(buf)

def query(state, item):
    offset = item * 2
    val = struct.unpack_from('<H', state, offset)[0]
    return val

def update(state, item):
    b = bytearray(state)
    offset = item * 2
    val = b[offset] | (b[offset + 1] << 8)
    val += 1
    b[offset] = val & 0xFF
    b[offset + 1] = (val >> 8) & 0xFF
    return bytes(b)

def query(state, item):
    b = bytearray(state)
    offset = item * 2
    val = b[offset] | (b[offset + 1] << 8)
    return val

def update(state, item):
    b = bytearray(state)
    bit_pos = item * 11
    byte_idx = bit_pos // 8
    bit_offset = bit_pos % 8
    
    val = 0
    for j in range(11):
        bi = byte_idx + (bit_offset + j) // 8
        bo = (bit_offset + j) % 8
        val |= ((b[bi] >> bo) & 1) << j
    
    val += 1
    
    for j in range(11):
        bi = byte_idx + (bit_offset + j) // 8
        bo = (bit_offset + j) % 8
        if (val >> j) & 1:
            b[bi] |= (1 << bo)
        else:
            b[bi] &= ~(1 << bo)
    
    return bytes(b)

def query(state, item):
    bit_pos = item * 11
    byte_idx = bit_pos // 8
    bit_offset = bit_pos % 8
    
    val = 0
    for j in range(11):
        bi = byte_idx + (bit_offset + j) // 8
        bo = (bit_offset + j) % 8
        val |= ((state[bi] >> bo) & 1) << j
    
    return val

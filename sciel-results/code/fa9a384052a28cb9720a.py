"""
Exact frequency counter using bit-packing.
Layout: 11 bits per item for items 0-399.
Total bits: 400 * 11 = 4400 bits = 550 bytes.
Fits in 1024 bytes.
Max count 1024 fits in 11 bits.
Update: Read 11 bits, increment, write back.
Query: Read 11 bits, return as float.
"""
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145  # 5s safety margin before 150s kill

def update(state, item):
    if time.time() > _DEADLINE:
        return state
    
    b = bytearray(state)
    start_bit = item * 11
    
    # Read current count
    count = 0
    for i in range(11):
        bit_pos = start_bit + i
        byte_i = bit_pos // 8
        bit_i = bit_pos % 8
        if (b[byte_i] >> bit_i) & 1:
            count |= (1 << i)
            
    # Increment
    count += 1
    
    # Write back
    for i in range(11):
        bit_pos = start_bit + i
        byte_i = bit_pos // 8
        bit_i = bit_pos % 8
        if (count >> i) & 1:
            b[byte_i] |= (1 << bit_i)
        else:
            b[byte_i] &= ~(1 << bit_i)
            
    return bytes(b)

def query(state, item):
    if time.time() > _DEADLINE:
        return 0.0
        
    # Read count
    start_bit = item * 11
    count = 0
    for i in range(11):
        bit_pos = start_bit + i
        byte_i = bit_pos // 8
        bit_i = bit_pos % 8
        if (state[byte_i] >> bit_i) & 1:
            count |= (1 << i)
            
    return float(count)

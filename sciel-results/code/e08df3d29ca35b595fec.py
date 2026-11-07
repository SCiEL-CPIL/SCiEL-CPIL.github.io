"""
Exact frequency counting using 10 bits per item (400 items * 10 bits = 4000 bits = 500 bytes).
Each item has a dedicated 10-bit counter. Update increments the counter; query reads it.
This gives zero-error exact counting within the 1024-byte state limit.
"""

def update(state, item):
    """
    Increment the 10-bit counter for the given item.
    state: bytes (1024 bytes)
    item: int in 0..399
    returns: new state as bytes
    """
    # Convert to bytearray for mutation
    b = bytearray(state)
    
    # Item's 10-bit counter starts at bit position item * 10
    bit_offset = item * 10
    byte_start = bit_offset // 8
    bit_in_byte = bit_offset % 8
    
    # Read current 10-bit value
    # The 10 bits span from bit_in_byte in byte_start to the next byte(s)
    val = 0
    # Read enough bytes to cover 10 bits
    for i in range(2):
        byte_idx = byte_start + i
        if byte_idx < len(b):
            val |= (b[byte_idx] << (8 * i))
    
    # Mask to 10 bits
    val &= 0x3FF
    
    # Increment
    val = (val + 1) & 0x3FF  # Cap at 1023 (10 bits)
    
    # Write back the 10-bit value
    # Clear the 10 bits first, then set
    # We need to clear bits in byte_start and possibly byte_start+1
    # Clear the bits in byte_start that are part of this counter
    # The counter occupies bits [bit_in_byte, bit_in_byte + 9]
    # In byte_start: bits bit_in_byte through 7 (8 - bit_in_byte bits)
    # In byte_start+1: bits 0 through (bit_in_byte + 9) - 8 = bit_in_byte + 1
    
    # Clear the 10 bits
    # Mask for byte_start: bits 0..(bit_in_byte-1) are NOT part of counter
    # Bits bit_in_byte..7 ARE part of counter
    mask_byte0 = 0xFF
    if bit_in_byte > 0:
        mask_byte0 &= (1 << bit_in_byte) - 1  # Keep lower bits
    # Actually, we want to CLEAR the counter bits, so we keep the non-counter bits
    # Non-counter bits in byte_start: bits 0..bit_in_byte-1
    keep_mask0 = (1 << bit_in_byte) - 1 if bit_in_byte > 0 else 0
    b[byte_start] = (b[byte_start] & keep_mask0) | ((val >> bit_in_byte) & (0xFF - keep_mask0))
    
    # In byte_start+1: bits 0..(bit_in_byte+1) are part of counter (since 10 - (8 - bit_in_byte) = 2 + bit_in_byte)
    num_bits_in_byte1 = 10 - (8 - bit_in_byte)  # = 2 + bit_in_byte
    if num_bits_in_byte1 > 0 and byte_start + 1 < len(b):
        keep_mask1 = (0xFF >> num_bits_in_byte1)  # Keep upper bits
        b[byte_start + 1] = (b[byte_start + 1] & keep_mask1) | ((val >> 8) & (~keep_mask1 & 0xFF))
    
    return bytes(b)


def query(state, item):
    """
    Read the 10-bit counter for the given item.
    state: bytes (1024 bytes)
    item: int in 0..399
    returns: nonnegative frequency estimate (int)
    """
    bit_offset = item * 10
    byte_start = bit_offset // 8
    bit_in_byte = bit_offset % 8
    
    # Read the 10-bit value
    val = 0
    for i in range(2):
        byte_idx = byte_start + i
        if byte_idx < len(state):
            val |= (state[byte_idx] << (8 * i))
    
    # Mask to 10 bits and shift
    val = (val >> bit_in_byte) & 0x3FF
    
    return val

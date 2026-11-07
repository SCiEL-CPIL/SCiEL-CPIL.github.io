"""
Bit-sliced exact frequency counters for 400 items, each with 11 bits (0-2047).
Total storage: 400 * 11 / 8 = 550 bytes (padded to 1024).
This gives EXACT counts up to 2047, well above the max possible count of 1024.
"""
# EVOLVE-BLOCK-START

def update(state, item):
    """Increment the exact count for `item` using bit-sliced storage."""
    state = bytearray(state)
    
    if item < 0 or item >= 400:
        # Should not happen per contract; return unchanged state
        return bytes(state)
    
    # Each item uses 11 bits: 400 items * 11 bits = 4400 bits = 550 bytes
    # Bit layout for item i:
    #   byte_offset = i * 11 // 8 = i * 11 >> 3
    #   bit_offset  = i * 11 % 8
    # We store bit 0 (LSB) at the lowest address.
    
    base_bit = item * 11  # global bit index for LSB of this item's count
    carry = 1  # We are incrementing by 1
    
    bit_idx = base_bit
    while carry and bit_idx < 550 * 8:
        byte_i = bit_idx // 8
        bit_in_byte = bit_idx % 8
        mask = 1 << bit_in_byte
        
        current_bit = (state[byte_i] & mask) != 0
        if not current_bit:
            state[byte_i] |= mask
            carry = 0  # No more carry
        else:
            state[byte_i] &= ~mask
            # Carry continues to next bit
        
        bit_idx += 1
    
    return bytes(state)


def query(state, item):
    """Return the exact frequency estimate for `item`."""
    if item < 0 or item >= 400:
        return 0.0
    
    # Read 11 bits for this item
    base_bit = item * 11
    count = 0
    for b in range(11):
        bit_idx = base_bit + b
        byte_i = bit_idx // 8
        bit_in_byte = bit_idx % 8
        if state[byte_i] & (1 << bit_in_byte):
            count |= (1 << b)
    
    return float(count)

# EVOLVE-BLOCK-END

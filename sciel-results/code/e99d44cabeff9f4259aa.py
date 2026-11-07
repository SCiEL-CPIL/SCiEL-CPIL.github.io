import struct

# State layout:
# Bytes 0-799: 400 uint16 counters for items 0-399
# Bytes 800-1023: Unused padding

_NUM_ITEMS = 400
_BYTES_PER_ITEM = 2
_TOTAL_SIZE = _NUM_ITEMS * _BYTES_PER_ITEM  # 800 bytes

def update(state, item):
    buf = bytearray(state)
    
    # Calculate offset for the item's counter
    offset = item * _BYTES_PER_ITEM
    
    # Unpack the current count
    # Using little-endian uint16
    count = struct.unpack_from('<H', buf, offset)[0]
    
    # Increment the count
    # Max possible count is 1024 (stream length), which fits in uint16 (max 65535)
    # No overflow check strictly needed for the given constraints, but good practice
    if count < 0xFFFF:
        count += 1
    
    # Pack the new count back
    struct.pack_into('<H', buf, offset, count)
    
    return bytes(buf)

def query(state, item):
    # Calculate offset for the item's counter
    offset = item * _BYTES_PER_ITEM
    
    # Unpack the count
    count = struct.unpack_from('<H', state, offset)[0]
    
    return count

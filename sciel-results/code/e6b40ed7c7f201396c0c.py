import struct

# 256 pairs of (uint16 item, uint16 count)
# 256 * 2 * 2 = 1024 bytes

def update(state, item):
    # Unpack 512 uint16s
    data = struct.unpack('<512H', state)
    data = list(data)
    
    found = False
    min_idx = 0
    min_val = data[1] # count of first pair
    
    # Search for item and find min count
    # We iterate over the 256 pairs
    for i in range(256):
        item_idx = i * 2
        count_idx = i * 2 + 1
        
        if data[item_idx] == item:
            data[count_idx] += 1
            found = True
            break
        
        if data[count_idx] < min_val:
            min_val = data[count_idx]
            min_idx = item_idx # Store the item index of the min count pair
            
    if not found:
        # Replace the min count entry
        data[min_idx] = item
        data[min_idx + 1] = min_val + 1
        
    return struct.pack('<512H', *data)

def query(state, item):
    data = struct.unpack('<512H', state)
    for i in range(256):
        if data[i * 2] == item:
            return float(data[i * 2 + 1])
    return 0.0

import struct

def update(state, item):
    # Unpack 400 uint16s from the first 800 bytes
    counts = list(struct.unpack('<400H', state[:800]))
    
    # Increment the count for the specific item
    counts[item] += 1
    
    # Pack back to 800 bytes and pad to 1024 bytes
    new_counts = struct.pack('<400H', *counts)
    return new_counts + b'\x00' * 224

def query(state, item):
    # Unpack 400 uint16s from the first 800 bytes
    counts = struct.unpack('<400H', state[:800])
    return float(counts[item])

import struct

def update(state, item):
    vals = list(struct.unpack('<512H', state))
    
    # Find item
    found = False
    for i in range(256):
        if vals[2*i] == item:
            vals[2*i + 1] += 1
            found = True
            break
            
    if not found:
        # Find min count
        min_count = 65535
        min_idx = 0
        for i in range(256):
            c = vals[2*i + 1]
            if c < min_count:
                min_count = c
                min_idx = i
                
        vals[2*min_idx] = item
        vals[2*min_idx + 1] = min_count + 1
        
    return struct.pack('<512H', *vals)

def query(state, item):
    vals = struct.unpack('<512H', state)
    
    # Check if item exists
    for i in range(256):
        if vals[2*i] == item:
            return float(vals[2*i + 1])
            
    # Item not found
    # Calculate variance of counts to decide estimate
    counts = [vals[2*i + 1] for i in range(256)]
    
    # Quick mean/variance
    total = sum(counts)
    mean = total / 256.0
    var = 0.0
    for c in counts:
        var += (c - mean)**2
    var /= 256.0
    
    if var < 2.0:
        est = min(counts)
    else:
        est = 0
        
    return float(est)

import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    
    # Generate all possible vectors as bits
    # Shape: (2^17, 17)
    # Using vectorized bit extraction
    max_val = 1 << N
    vectors_int = np.arange(max_val, dtype=np.uint32)
    
    # Extract bits
    # (2^17, 1) >> (17,) -> (2^17, 17)
    bits = ((vectors_int[:, None] >> np.arange(N, dtype=np.uint32)) & 1).astype(np.uint8)
    
    # Calculate weights (popcount)
    weights = bits.sum(axis=1)
    
    # Sort indices by weight (ascending)
    # Ties are broken by integer value (stable sort)
    sorted_indices = np.argsort(weights, kind='stable')
    
    selected_indices = []
    selected_bits = None
    
    for idx in sorted_indices:
        v = bits[idx]
        
        if not selected_indices:
            selected_indices.append(idx)
            selected_bits = v.reshape(1, -1)
        else:
            # Check distance to all selected bits
            # XOR and sum along axis 1
            dists = (selected_bits ^ v).sum(axis=1)
            
            if np.min(dists) >= MIN_DIST:
                selected_indices.append(idx)
                selected_bits = np.vstack((selected_bits, v))
                
    return [int(i) for i in selected_indices]

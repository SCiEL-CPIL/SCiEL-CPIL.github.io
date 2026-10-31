import numpy as np

def priority(vector, n):
    # Calculate Hamming weight (number of non-zero elements)
    # Vectors with more non-zero elements are "denser" and might be less likely 
    # to form simple low-weight lines early in the greedy process compared to sparse vectors.
    # We prioritize higher Hamming weight.
    
    # Count non-zero elements
    weight = sum(1 for x in vector if x != 0)
    
    # To break ties deterministically but non-lexicographically, we add a small 
    # pseudo-random component derived from a hash.
    # This ensures that within the same weight class, the order is "randomized" 
    # rather than lexicographic, which can be suboptimal.
    import hashlib
    v_str = str(vector)
    h = int(hashlib.md5(v_str.encode('utf-8')).hexdigest(), 16)
    # Scale hash to [0, 1) and multiply by a small factor (e.g., 0.5) 
    # so that weight (int) is the dominant term.
    # Max weight is n. So epsilon should be < 1.
    epsilon = float(h) / float(0xFFFFFFFFFFFFFFFF) * 0.5
    
    return float(weight) + epsilon

# EVOLVE-BLOCK-END

import numpy as np
from math import lcm
from functools import reduce

def construct():
    # Calculate LCM of 1..30
    L = reduce(lcm, range(1, 31))
    
    # Weights w_i = L // i
    weights = np.array([L // i for i in range(1, 31)], dtype=np.int64)
    
    # Split into two halves
    w1 = weights[:15]
    w2 = weights[15:]
    
    def generate_sums(weights_half):
        # Returns sums (np.array) and configs (np.array)
        # configs are base-3 encoded integers
        # Mapping: digit 0 -> -1, 1 -> 0, 2 -> 1
        # We process weights in order.
        # If we do:
        # new_sums = concat(sums - w, sums, sums + w)
        # new_configs = concat(configs * 3 + 0, configs * 3 + 1, configs * 3 + 2)
        # Then the last processed weight is in the least significant digit.
        # If we process w[0] first, then w[1], ... w[14],
        # w[0] is in the most significant digit (if we multiply by 3 each step).
        # Let's trace:
        # Start: sums=[0], configs=[0]
        # Process w[0]:
        #   sums = [-w0, 0, w0]
        #   configs = [0, 1, 2]
        # Process w[1]:
        #   sums = [-w0-w1, -w0, -w0+w1, -w1, 0, w1, w0-w1, w0, w0+w1]
        #   configs:
        #     0*3+0=0, 0*3+1=1, 0*3+2=2
        #     1*3+0=3, 1*3+1=4, 1*3+2=5
        #     2*3+0=6, 2*3+1=7, 2*3+2=8
        #   So config 0 corresponds to -w0-w1.
        #   Digits: 0 (for w0), 0 (for w1).
        #   Digit 0 of config (LSB) is for w[1] (last processed).
        #   Digit 1 of config (next) is for w[0] (first processed).
        
        sums = [0]
        configs = [0]
        
        for w in weights_half:
            # To maintain alignment, we must expand carefully.
            # Old sums: S. New sums: S-w, S, S+w.
            # Old configs: C. New configs: C*3+0, C*3+1, C*3+2.
            # This implies the new digit (LSB) corresponds to the current weight w.
            # So if we process w[0] then w[1], w[1] is LSB, w[0] is next.
            
            sums_arr = np.array(sums, dtype=np.int64)
            configs_arr = np.array(configs, dtype=np.int64)
            
            new_sums = np.concatenate([
                sums_arr - w,
                sums_arr,
                sums_arr + w
            ])
            
            new_configs = np.concatenate([
                configs_arr * 3 + 0,
                configs_arr * 3 + 1,
                configs_arr * 3 + 2
            ])
            
            sums = new_sums
            configs = new_configs
            
        return np.array(sums, dtype=np.int64), np.array(configs, dtype=np.int64)

    sums1, configs1 = generate_sums(w1)
    sums2, configs2 = generate_sums(w2)
    
    # Sort sums2 and align configs2
    order = np.argsort(sums2)
    sums2_sorted = sums2[order]
    configs2_sorted = configs2[order]
    
    # For each sum in sums1, find best match in sums2_sorted
    # We want to minimize |s1 + s2| != 0
    
    # Vectorized search
    # target = -sums1
    targets = -sums1
    
    # searchsorted returns index where target would be inserted to keep order
    # We check idx and idx-1
    idx = np.searchsorted(sums2_sorted, targets)
    
    # Clip indices to valid range
    # We need to check idx and idx-1
    # Create arrays for indices to check
    idx1 = np.clip(idx, 0, len(sums2_sorted) - 1)
    idx2 = np.clip(idx - 1, 0, len(sums2_sorted) - 1)
    
    # Get candidate sums
    s2_cand1 = sums2_sorted[idx1]
    s2_cand2 = sums2_sorted[idx2]
    
    # Calculate diffs
    diff1 = sums1 + s2_cand1
    diff2 = sums1 + s2_cand2
    
    # We need min |diff| where diff != 0
    # Combine candidates
    all_diffs = np.concatenate([diff1, diff2])
    all_abs = np.abs(all_diffs)
    
    # Mask out zeros
    valid_mask = (all_diffs != 0)
    all_abs[~valid_mask] = np.iinfo(np.int64).max
    
    min_val = np.min(all_abs)
    min_idx = np.argmin(all_abs)
    
    # Recover indices
    if min_idx < len(sums1):
        # Came from diff1
        i1 = min_idx
        i2 = idx1[i1]
    else:
        # Came from diff2
        i1 = min_idx - len(sums1)
        i2 = idx2[i1]
        
    config1 = configs1[i1]
    config2 = configs2_sorted[i2]
    
    # Decode configs to coefficients
    # Config is base-3.
    # For w1 (indices 0..14 in w1 array, corresponding to 1..15 in original):
    # LSB corresponds to w1[14] (index 15 in original? No, w1 is weights[0:15], so indices 0..14 correspond to 1..15).
    # Wait, w1[i] corresponds to original index i+1.
    # In generate_sums, we processed w1[0], w1[1], ..., w1[14].
    # LSB (digit 0) corresponds to w1[14] (original index 15).
    # Digit 1 corresponds to w1[13] (original index 14).
    # ...
    # Digit 14 corresponds to w1[0] (original index 1).
    
    coeffs = [0] * 30
    
    # Decode config1
    for k in range(15):
        digit = config1 % 3
        config1 //= 3
        # digit 0 -> -1, 1 -> 0, 2 -> 1
        c = digit - 1
        # This digit corresponds to w1[14-k]
        # w1[14-k] is original index (14-k) + 1 = 15-k
        # So coeffs[15-k-1] = coeffs[14-k]
        coeffs[14 - k] = c
        
    # Decode config2
    for k in range(15):
        digit = config2 % 3
        config2 //= 3
        c = digit - 1
        # This digit corresponds to w2[14-k]
        # w2[14-k] is original index 15 + (14-k) = 29-k
        # So coeffs[29-k] = c
        coeffs[29 - k] = c
        
    return coeffs


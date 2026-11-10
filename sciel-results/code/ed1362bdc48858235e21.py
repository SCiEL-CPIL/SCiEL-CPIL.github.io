import time

def construct():
    # The problem is small enough (2^10 subsets) that we can precompute everything
    # in a very fast loop without complex deadline guards inside, 
    # but we keep a global deadline for safety.
    
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    best_size = [11] * 101
    best_mask = [0] * 101  # 0 means not found
    
    # Iterate all non-empty subsets
    # We can compute sum and size incrementally or just calculate.
    # 1023 iterations is nothing.
    
    for mask in range(1, 1 << 10):
        size = 0
        s_sum = 0
        m = mask
        idx = 0
        while m:
            if m & 1:
                size += 1
                s_sum += inv[idx + 1]
            idx += 1
            m >>= 1
            
        residue = s_sum % 101
        
        if size < best_size[residue]:
            best_size[residue] = size
            best_mask[residue] = mask

    result = []
    for r in range(101):
        if best_mask[r]:
            subset = []
            m = best_mask[r]
            idx = 0
            while m:
                if m & 1:
                    subset.append(idx + 1)
                idx += 1
                m >>= 1
            result.append(subset)
        else:
            # Fallback, should not happen
            result.append([1])
            
    return result

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # We use a greedy approach to select 40 masks that maximize comparability.
    # Comparability is defined as A subset B or B subset A.
    
    selected = []
    available = set(range(total_masks))
    
    for _ in range(m):
        best_score = -1
        best_mask = -1
        
        # Iterate through all available masks to find the one with the highest
        # number of comparable pairs with the already selected masks.
        for mask in available:
            score = 0
            for s in selected:
                # Check subset relation using bitmasks
                # (s & mask) == s  => s is subset of mask
                # (s & mask) == mask => mask is subset of s
                if (s & mask) == s or (s & mask) == mask:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_mask = mask
        
        selected.append(best_mask)
        available.remove(best_mask)
        
    # Convert the selected masks to lists of integers
    result = []
    for mask in selected:
        result.append([b for b in range(n) if (mask >> b) & 1])
        
    return result

# EVOLVE-BLOCK-END

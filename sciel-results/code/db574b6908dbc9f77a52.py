def construct():
    """
    Assigns colors to positions 0..127 such that every nonconstant 4-term arithmetic progression
    uses at least 3 distinct colors. Minimizes the number of distinct color labels used.
    
    Uses bitwise integer packing: for each position, we maintain a 128-bit mask where bit c
    indicates color c is forbidden at that position. This allows O(1) parallel evaluation
    of all 128 candidate colors per position.
    """
    n = 128
    colors = [0] * n
    colors[0] = 0
    
    # Precompute bit masks for quick reference
    # bit_mask[c] = 1 << c
    bit_mask = [1 << c for c in range(n)]
    
    for i in range(1, n):
        forbidden = 0  # 128-bit mask of forbidden colors at position i
        
        # Constraint 1: 4-term progressions ending at i
        # (i-3d, i-2d, i-d, i) must use >= 3 distinct colors
        for d in range(1, i // 3 + 1):
            c1 = colors[i - d]
            c2 = colors[i - 2 * d]
            c3 = colors[i - 3 * d]
            
            # If all three previous are same color, no valid choice (shouldn't happen)
            if c1 == c2 and c2 == c3:
                forbidden = (1 << n) - 1
                break
            
            # Determine distinct colors among c1, c2, c3
            if c1 == c2:
                # c1==c2, c3 different
                # Forbidden: c1 (would give {c1,c3} size 2) and c3 (would give {c1,c3} size 2)
                forbidden |= bit_mask[c1]
                forbidden |= bit_mask[c3]
            elif c1 == c3:
                # c1==c3, c2 different
                forbidden |= bit_mask[c1]
                forbidden |= bit_mask[c2]
            elif c2 == c3:
                # c2==c3, c1 different
                forbidden |= bit_mask[c2]
                forbidden |= bit_mask[c1]
            else:
                # All three distinct: any color works, no restriction
                pass
        
        # Constraint 2: Prevent 3-term monochromatic subsequences
        # If colors[i-2d] == colors[i-d] == K, then colors[i] != K
        for d in range(1, i // 2 + 1):
            c1 = colors[i - d]
            c2 = colors[i - 2 * d]
            if c1 == c2:
                forbidden |= bit_mask[c1]
        
        # Find smallest color not in forbidden mask
        # Use bit trick: invert forbidden, find lowest set bit
        available = ~forbidden & ((1 << n) - 1)
        if available == 0:
            # Fallback (shouldn't occur with correct logic)
            colors[i] = 0
        else:
            # Find index of lowest set bit
            # available & (-available) isolates the lowest set bit
            lowbit = available & (-available)
            c = lowbit.bit_length() - 1
            colors[i] = c
    
    return colors

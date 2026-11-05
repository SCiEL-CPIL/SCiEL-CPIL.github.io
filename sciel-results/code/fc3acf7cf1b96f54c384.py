"""Optimized program using bitwise integer packing for fast constraint checking."""
# EVOLVE-BLOCK-START
def construct():
    n = 128
    # colors[i] stores the color (0..127) assigned to position i (0-indexed)
    colors = [0] * n
    # For each position, we'll track which colors are blocked
    # Use a list of 128-bit masks: blocked_mask[i] has bit c set if color c is blocked at position i
    
    for x in range(n):
        # Start with all colors allowed (all bits set)
        blocked = 0  # bit c set means color c is blocked
        
        # Check all 4-term progressions ending at position x
        # A 4-term progression: (a, a+d, a+2d, a+3d) where a+3d = x
        # So a = x - 3d, and we need a >= 0, i.e., d <= x//3
        for d in range(1, x // 3 + 1):
            a = x - 3 * d
            # The three previous positions in this progression:
            p1 = a + d      # = x - 2d
            p2 = a + 2 * d  # = x - d
            p3 = a + 3 * d  # = x (current, not yet set)
            # Wait, the progression is (a, a+d, a+2d, a+3d)
            # Previous three are: a, a+d, a+2d
            # Current is a+3d = x
            # So we need colors at positions a, a+d, a+2d
            
            c_a = colors[a]
            c_ad = colors[a + d]
            c_a2d = colors[a + 2 * d]
            
            # The set of colors used in the previous three positions
            # We need |{new_color, c_a, c_ad, c_a2d}| >= 3
            
            # Count distinct colors among c_a, c_ad, c_a2d
            # If all three same: distinct = 1, must use a color different from c_a
            # If two same, one different: distinct = 2, must use a color not in {the two values}
            # If all different: distinct = 3, any color works
            
            # Build bitmask of the three colors
            mask_prev = (1 << c_a) | (1 << c_ad) | (1 << c_a2d)
            num_distinct = bin(mask_prev).count('1')
            
            if num_distinct <= 2:
                # Block all colors that are in the previous set
                # Because if new_color is in {c_a, c_ad, c_a2d}, then the total distinct
                # colors will be at most num_distinct <= 2
                blocked |= mask_prev
        
        # Find the smallest color not blocked
        # We want to minimize the number of distinct colors used overall
        # Greedy: pick the smallest available color
        available = ~blocked & ((1 << 128) - 1)
        
        # Find the lowest set bit
        if available == 0:
            # Should not happen, but fallback
            colors[x] = 0
        else:
            # Find index of lowest set bit
            # available & (-available) isolates the lowest set bit
            lowest = available & (-available)
            # Get the bit position
            color = lowest.bit_length() - 1
            colors[x] = color
    
    return colors
# EVOLVE-BLOCK-END

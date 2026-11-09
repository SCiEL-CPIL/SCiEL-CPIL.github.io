"""Colorings with three colors in every four-term progression."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = 80  # safety margin within 90s wall-clock
    n = 128
    block_size = 16
    num_blocks = n // block_size  # 8 blocks

    # Precompute all 4-term APs within a block of size 16
    # AP: (a, a+d, a+2d, a+3d) where 1 <= a, a+3d <= 16
    def get_internal_aps(block_size):
        aps = []
        for a in range(1, block_size - 2):
            max_d = (block_size - a) // 3
            for d in range(1, max_d + 1):
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
        return aps

    internal_aps = get_internal_aps(block_size)

    # Precompute valid patterns for a block with k colors
    def get_valid_block_patterns(k_colors, block_size):
        """Find all valid colorings of a block that satisfy internal AP constraints."""
        valid_patterns = []
        # Try all combinations is too expensive, use backtracking
        def backtrack(pos, current, used_colors):
            if pos == block_size:
                valid_patterns.append(tuple(current))
                return
            for color in range(k_colors):
                current.append(color)
                # Check if adding this color violates any AP that ends at pos
                valid = True
                for (a, b, c, d) in internal_aps:
                    if d == pos:  # 0-indexed
                        colors_ap = [current[a], current[b], current[c], current[d]]
                        if len(set(colors_ap)) < 3:
                            valid = False
                            break
                if valid:
                    backtrack(pos + 1, current, used_colors | {color})
                current.pop()
        
        backtrack(0, [], set())
        return valid_patterns

    # Try with increasing number of colors
    for k_colors in range(3, 8):
        if time.time() > deadline - 10:
            break
        
        # Get valid patterns for this number of colors
        valid_patterns = get_valid_block_patterns(k_colors, block_size)
        
        if not valid_patterns:
            continue
        
        # Now combine blocks: each block can use any valid pattern
        # We need to check cross-boundary APs
        # A cross-boundary AP spans multiple blocks
        
        # For now, try to find a combination that works
        # Use backtracking over blocks
        def check_cross_boundary_ap(colors, a, d):
            """Check if AP (a, a+d, a+2d, a+3d) has at least 3 distinct colors."""
            idx = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1]
            if any(i < 0 or i >= n for i in idx):
                return True
            c = [colors[i] for i in idx]
            return len(set(c)) >= 3

        def is_valid_global(colors):
            """Check all APs globally."""
            for a in range(1, n - 2):
                max_d = (n - a) // 3
                for d in range(1, max_d + 1):
                    if not check_cross_boundary_ap(colors, a, d):
                        return False
            return True

        # Backtracking: assign patterns to blocks
        def combine_blocks(block_assignments, block_idx, current_colors):
            if block_idx == num_blocks:
                return current_colors
            
            for pattern in valid_patterns:
                new_colors = current_colors + list(pattern)
                # Check APs that end in this block and started in earlier blocks
                valid = True
                # Check APs that span from earlier blocks to this block
                # An AP (a, a+d, a+2d, a+3d) with a+3d in this block
                block_start = block_idx * block_size
                block_end = block_start + block_size
                for a in range(1, n - 2):
                    max_d = (n - a) // 3
                    for d in range(1, max_d + 1):
                        a3d = a + 3 * d
                        if block_start < a3d < block_end:
                            # This AP ends in current block, check if valid
                            if not check_cross_boundary_ap(new_colors, a, d):
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    result = combine_blocks(block_assignments, block_idx + 1, new_colors)
                    if result is not None:
                        return result
            
            return None

        result = combine_blocks([], 0, [])
        if result is not None and is_valid_global(result):
            return result

    # Fallback: use the original greedy approach
    def greedy_simple(num_colors):
        colors = [0] * n
        colors[0] = 0
        for x in range(2, n + 1):
            valid_colors = set(range(num_colors))
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1:
                    break
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                existing = {c1, c2, c3}
                if len(existing) >= 3:
                    pass
                elif len(existing) == 2:
                    for c in list(valid_colors):
                        if c in existing:
                            valid_colors.discard(c)
                else:
                    for c in list(valid_colors):
                        if c == list(existing)[0]:
                            valid_colors.discard(c)
            if not valid_colors:
                return None
            colors[x - 1] = min(valid_colors)
        
        # Verify
        for a in range(1, n - 2):
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                c4 = colors[a + 3 * d - 1]
                if len({c1, c2, c3, c4}) < 3:
                    return None
        return colors

    # Try greedy with increasing colors
    for k in range(3, 8):
        if time.time() > deadline - 5:
            break
        result = greedy_simple(k)
        if result is not None:
            return result

    # Last resort: return a simple pattern
    colors = []
    for x in range(1, 129):
        colors.append((x - 1) % 3)
    return colors

# EVOLVE-BLOCK-END

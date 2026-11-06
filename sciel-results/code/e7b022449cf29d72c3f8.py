"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s safety margin
    
    n = 128
    
    # Precompute all nonconstant 4-term APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n + 1):
            a3 = a + 3 * d
            if a3 <= n:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a3 - 1))
    
    num_aps = len(aps)
    
    def count_colors(colors):
        return len(set(colors))
    
    def check_valid(colors):
        """Check if coloring is valid: every 4-term AP has >= 3 distinct colors"""
        for idx1, idx2, idx3, idx4 in aps:
            c = {colors[idx1], colors[idx2], colors[idx3], colors[idx4]}
            if len(c) < 3:
                return False
        return True
    
    def check_valid_fast(colors, ap_list):
        """Vectorized-ish check"""
        for idx1, idx2, idx3, idx4 in ap_list:
            c1 = colors[idx1]
            c2 = colors[idx2]
            c3 = colors[idx3]
            c4 = colors[idx4]
            # Count distinct
            if c1 == c2 == c3 == c4:
                return False
            # Check if exactly 2 distinct
            if c1 == c2:
                if c3 == c4:
                    return False  # AA BB
                elif c3 == c1 or c4 == c1:
                    return False  # AA A B or AA B A
                elif c3 == c4:
                    return False  # AA B B
            elif c1 == c3:
                if c2 == c4:
                    return False  # ABAB
                elif c2 == c1 or c4 == c1:
                    return False
                elif c2 == c4:
                    return False
            elif c1 == c4:
                if c2 == c3:
                    return False  # A B B A
                elif c2 == c1 or c3 == c1:
                    return False
                elif c2 == c3:
                    return False
            elif c2 == c3:
                if c1 == c4:
                    return False
                elif c1 == c2 or c4 == c2:
                    return False
                elif c1 == c4:
                    return False
            elif c2 == c4:
                if c1 == c3:
                    return False
                elif c1 == c2 or c3 == c2:
                    return False
                elif c1 == c3:
                    return False
            elif c3 == c4:
                if c1 == c2:
                    return False
                elif c1 == c3 or c2 == c3:
                    return False
                elif c1 == c2:
                    return False
        return True
    
    # Try to find a valid coloring with k colors for k = 3, 4, 5, ...
    for k in [3, 4, 5, 6]:
        if time.time() > deadline - 15:
            break
        
        # Initialize with random assignment using k colors
        best_colors = None
        best_num_colors = k + 1
        
        # Multiple restarts
        num_restarts = 50 if k == 3 else 30
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Local search: try to fix violations
            for iteration in range(2000):
                if time.time() > deadline - 5:
                    break
                
                # Find a violated AP
                violated = False
                for idx1, idx2, idx3, idx4 in aps:
                    c = {colors[idx1], colors[idx2], colors[idx3], colors[idx4]}
                    if len(c) < 3:
                        violated = True
                        # Try to fix by changing one of the positions
                        fixed = False
                        for pos in [idx1, idx2, idx3, idx4]:
                            old_color = colors[pos]
                            for new_color in range(k):
                                if new_color == old_color:
                                    continue
                                colors[pos] = new_color
                                # Check if this fixes the violation without creating new ones nearby
                                # Just accept for now, we'll re-check
                                fixed = True
                                break
                            if fixed:
                                break
                        break
                
                if not violated:
                    # Valid coloring found!
                    actual_colors = count_colors(colors)
                    if actual_colors < best_num_colors:
                        best_num_colors = actual_colors
                        best_colors = list(colors)
                    break
            
            # Check if current is valid
            if check_valid(colors):
                actual_colors = count_colors(colors)
                if actual_colors < best_num_colors:
                    best_num_colors = actual_colors
                    best_colors = list(colors)
        
        if best_colors is not None and best_num_colors <= k:
            return best_colors
    
    # Fallback: use the greedy seed approach
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    
    return colors

# EVOLVE-BLOCK-END

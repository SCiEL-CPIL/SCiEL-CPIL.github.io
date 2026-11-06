import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Try K=3, 4, 5...
    # We assume the seed uses at least 3.
    # If we find a solution with K colors, we return it.
    # We prioritize smaller K.
    
    for K in range(3, 10):
        # Allocate some time to this K
        # If K=3, give it more time? Or just iterate?
        # Let's do a fixed number of restarts per K, bounded by time.
        
        max_restarts = 5000
        found = False
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Randomized Greedy
            colors = [None] * 128
            used_colors = set()
            success = True
            
            for i in range(128):
                pos = i + 1
                valid = set(range(K))
                
                # Check all 4-APs ending at pos
                # pos, pos-d, pos-2d, pos-3d
                # d >= 1, pos-3d >= 1 => d <= (pos-1)//3
                max_d = (pos - 1) // 3
                for d in range(1, max_d + 1):
                    i1 = pos - 3*d
                    i2 = pos - 2*d
                    i3 = pos - d
                    
                    c1 = colors[i1-1]
                    c2 = colors[i2-1]
                    c3 = colors[i3-1]
                    
                    # For each candidate color c for pos, check if {c1, c2, c3, c} has size >= 3
                    # If not, c is invalid.
                    # We can optimize this by precomputing invalid colors for this AP.
                    # But K is small, so just loop.
                    for c in list(valid):
                        if len({c1, c2, c3, c}) < 3:
                            valid.discard(c)
                            
                if not valid:
                    success = False
                    break
                    
                # Choose color
                # Bias towards used colors
                preferred = valid & used_colors
                if preferred:
                    c = random.choice(list(preferred))
                else:
                    c = random.choice(list(valid))
                    
                colors[i] = c
                used_colors.add(c)
                
            if success:
                # We found a valid coloring with K colors
                # The task says "Minimize the number of labels actually used".
                # Our algorithm uses a subset of 0..K-1.
                # We should return the actual labels used, or just the list?
                # "construct() -> n=128 integer color labels in 0..127"
                # The labels are the values in the list.
                # We don't need to remap them to 0..M-1, just return the list.
                # However, if we used only colors {0, 2} from K=3, the number of labels is 2.
                # But the condition "at least 3 distinct colors" implies we must use at least 3.
                # So used_colors will have size >= 3.
                return colors
        
        # If loop finished without success, try next K
        # But we might have broken due to time
        if time.time() > deadline:
            break
            
    # If we ran out of time or didn't find a solution with K < 10
    # Fall back to the seed algorithm or a deterministic greedy
    # The seed algorithm is deterministic and likely valid.
    # Let's just run the seed algorithm if we didn't find a better one?
    # Or return a known valid coloring?
    # The seed program is:
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors

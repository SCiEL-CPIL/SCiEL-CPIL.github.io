import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin out of 60s
    
    n = 128
    
    # Precompute 4-term progressions ending at each index x (0-indexed)
    # A progression is (x-3d, x-2d, x-d, x)
    ap_ending = [[] for _ in range(n)]
    for x in range(n):
        max_d = x // 3
        for d in range(1, max_d + 1):
            i3 = x - d
            i2 = x - 2 * d
            i1 = x - 3 * d
            if i1 >= 0:
                ap_ending[x].append((i1, i2, i3))

    # Lookup table for popcount for small masks (up to 2^10)
    popcount = [0] * 1024
    for i in range(1024):
        popcount[i] = bin(i).count('1')

    # Try increasing number of colors
    for K in range(3, 15):
        # Try many random restarts
        # Number of restarts can be adjusted based on time
        max_restarts = 5000
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            colors = [-1] * n
            success = True
            
            for x in range(n):
                if time.time() > deadline:
                    success = False
                    break
                
                # Find all valid colors for position x
                valid_colors = []
                for c in range(K):
                    ok = True
                    for (i1, i2, i3) in ap_ending[x]:
                        # Calculate mask of colors used in the AP
                        # colors[i1], colors[i2], colors[i3] are already assigned
                        mask = (1 << colors[i1]) | (1 << colors[i2]) | (1 << colors[i3]) | (1 << c)
                        # Check if at least 3 distinct colors
                        if popcount[mask] < 3:
                            ok = False
                            break
                    if ok:
                        valid_colors.append(c)
                
                if not valid_colors:
                    success = False
                    break
                
                # Choose a color randomly to diversify search
                colors[x] = random.choice(valid_colors)
            
            if success:
                # Final verification (optional, but good for safety)
                valid = True
                for x in range(n):
                    for (i1, i2, i3) in ap_ending[x]:
                        mask = (1 << colors[i1]) | (1 << colors[i2]) | (1 << colors[i3]) | (1 << colors[x])
                        if popcount[mask] < 3:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return colors

    # Fallback: If no solution found within time/budget, return a deterministic greedy attempt
    # This might fail, but better than crashing or timeout without output
    # Or return a list of zeros (invalid, score 0)
    # Let's try one last deterministic greedy without randomization
    colors = [-1] * n
    for x in range(n):
        valid_colors = []
        for c in range(10):
            ok = True
            for (i1, i2, i3) in ap_ending[x]:
                mask = (1 << colors[i1]) | (1 << colors[i2]) | (1 << colors[i3]) | (1 << c)
                if popcount[mask] < 3:
                    ok = False
                    break
            if ok:
                valid_colors.append(c)
        if valid_colors:
            colors[x] = valid_colors[0]
        else:
            colors[x] = 0 # Placeholder
    
    return colors

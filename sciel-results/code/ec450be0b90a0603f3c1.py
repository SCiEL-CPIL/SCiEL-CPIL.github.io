import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s kill
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) where 1 <= a, a+3d <= n
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, precompute which APs it's part of and its role
    pos_aps = [[] for _ in range(n + 1)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_aps[a].append((idx, 0))
        pos_aps[b].append((idx, 1))
        pos_aps[c].append((idx, 2))
        pos_aps[d].append((idx, 3))
    
    def count_colors(colors, pos):
        """Count distinct colors in a 4-term AP given current coloring."""
        a, b, c, d = aps[pos]
        return len({colors[a-1], colors[b-1], colors[c-1], colors[d-1]})
    
    def is_valid_ap(colors, pos):
        return count_colors(colors, pos) >= 3
    
    def total_violations(colors):
        v = 0
        for pos in range(len(aps)):
            if not is_valid_ap(colors, pos):
                v += 1
        return v
    
    def num_distinct_colors(colors):
        return len(set(colors))
    
    # Try different strategies
    best_result = None
    best_num_colors = 128
    
    # Strategy 1: Greedy with lookahead - try to minimize colors
    # Start with a random assignment using few colors, then repair
    for num_colors_attempt in range(2, 8):  # try 2 to 7 colors
        if time.time() > deadline - 10:
            break
            
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initial assignment
            colors = [random.randint(0, num_colors_attempt - 1) for _ in range(n)]
            
            # Check if valid
            if total_violations(colors) == 0:
                nc = num_distinct_colors(colors)
                if nc < best_num_colors:
                    best_num_colors = nc
                    best_result = colors[:]
                break
            
            # Local search: fix violations by changing colors
            improved = True
            iterations = 0
            max_iterations = 5000
            
            while improved and iterations < max_iterations and time.time() < deadline - 5:
                improved = False
                iterations += 1
                
                # Find violated APs
                violated = [pos for pos in range(len(aps)) if not is_valid_ap(colors, pos)]
                
                if not violated:
                    nc = num_distinct_colors(colors)
                    if nc < best_num_colors:
                        best_num_colors = nc
                        best_result = colors[:]
                    break
                
                # Pick a random violated AP
                pos = random.choice(violated)
                a, b, c, d = aps[pos]
                current_colors = [colors[a-1], colors[b-1], colors[c-1], colors[d-1]]
                distinct = set(current_colors)
                
                # Try to fix by changing one position's color
                for role in range(4):
                    positions_in_ap = [a, b, c, d]
                    p = positions_in_ap[role]
                    old_color = colors[p-1]
                    
                    # Try new colors
                    for new_color in range(num_colors_attempt):
                        if new_color == old_color:
                            continue
                        colors[p-1] = new_color
                        # Check if this AP is now valid
                        new_distinct = len({colors[a-1], colors[b-1], colors[c-1], colors[d-1]})
                        if new_distinct >= 3:
                            # Check we didn't create new violations in other APs involving p
                            new_violations = 0
                            for (ap_idx, _) in pos_aps[p]:
                                if not is_valid_ap(colors, ap_idx):
                                    new_violations += 1
                            if new_violations <= 1:  # Allow slight increase
                                improved = True
                                break
                            else:
                                colors[p-1] = old_color
                    if improved:
                        break
                    colors[p-1] = old_color
            
            # After local search, check result
            if total_violations(colors) == 0:
                nc = num_distinct_colors(colors)
                if nc < best_num_colors:
                    best_num_colors = nc
                    best_result = colors[:]
    
    # Strategy 2: Greedy construction with minimal color introduction
    if time.time() < deadline - 5:
        for trial in range(100):
            if time.time() > deadline - 3:
                break
            
            colors = [None] * n
            used_colors = set()
            
            for i in range(n):
                pos = i + 1
                # Try existing colors first, then new colors
                candidates = list(used_colors)
                # Add some new color options
                for c in range(max(used_colors) + 1, max(used_colors) + 1 + 3):
                    if c not in candidates:
                        candidates.append(c)
                random.shuffle(candidates)
                
                found = False
                for c in candidates:
                    colors[i] = c
                    # Check all APs involving this position
                    valid = True
                    for (ap_idx, _) in pos_aps[pos]:
                        a, b, cd, d = aps[ap_idx]
                        # Only check if all positions are assigned
                        if all(colors[j-1] is not None for j in (a, b, cd, d)):
                            if len({colors[a-1], colors[b-1], colors[cd-1], colors[d-1]}) < 3:
                                valid = False
                                break
                    if valid:
                        used_colors.add(c)
                        found = True
                        break
                
                if not found:
                    # Backtrack or fail
                    break
            
            if all(c is not None for c in colors):
                # Verify full validity
                if total_violations(colors) == 0:
                    nc = num_distinct_colors(colors)
                    if nc < best_num_colors:
                        best_num_colors = nc
                        best_result = colors[:]
    
    # Strategy 3: Start from seed-like greedy but optimize
    if time.time() < deadline - 3 and (best_result is None or best_num_colors > 5):
        # Greedy: pick smallest valid color
        colors = []
        for x in range(1, n + 1):
            for color in range(128):
                ok = True
                # Check 4-term APs where x is the 4th term
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 3 * d
                    if a >= 1:
                        b = a + d
                    else:
                        continue
                    # a, b, c=x-d, x
                    c = x - d
                    if a >= 1 and b >= 1 and c >= 1:
                        if len({color, colors[a-1], colors[b-1], colors[c-1]}) < 3:
                            ok = False
                            break
                if ok:
                    # Check 4-term APs where x is 3rd term
                    for d in range(1, (x - 1) // 2 + 1):
                        a = x - 2 * d
                        if a >= 1:
                            b = a + d
                        else:
                            continue
                        # a, b, x, x+d
                        if x + d <= n:
                            # Can't check x+d yet, skip
                            pass
                    # Check 4-term APs where x is 2nd term
                    for d in range(1, (x - 1) // 1 + 1):
                        a = x - d
                        if a >= 1:
                            # a, x, x+d, x+2d
                            if x + 2 * d <= n:
                                # Can't check future, skip
                                pass
                    colors.append(color)
                    break
        
        if len(colors) == n:
            if total_violations(colors) == 0:
                nc = num_distinct_colors(colors)
                if nc < best_num_colors:
                    best_num_colors = nc
                    best_result = colors[:]
    
    # Final: if we have a valid result, return it; otherwise return the seed-like result
    if best_result is not None:
        # Ensure labels are in 0..127
        return best_result
    
    # Fallback: use the original greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors

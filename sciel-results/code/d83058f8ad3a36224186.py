import time
import random

def construct():
    N = 128
    # Precompute APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    M = len(aps)
    
    # Map position to list of AP indices
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_distinct_colors(c0, c1, c2, c3):
        """Count distinct colors using bitwise operations"""
        # Create bitmask for each color
        m0 = 1 << c0
        m1 = 1 << c1
        m2 = 1 << c2
        m3 = 1 << c3
        mask = m0 | m1 | m2 | m3
        return bin(mask).count('1')
    
    def solve_with_k(k, time_limit):
        start_time = time.time()
        
        # Precompute bit masks for colors 0..k-1
        color_masks = [1 << i for i in range(k)]
        
        # Random restarts
        for restart in range(200):
            if time.time() - start_time > time_limit:
                return None
            
            # Initialize with random colors
            colors = [random.randint(0, k - 1) for _ in range(N)]
            
            # Calculate initial violations using bitwise operations
            is_bad = [False] * M
            bad_count = 0
            for idx, ap in enumerate(aps):
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                mask = color_masks[c0] | color_masks[c1] | color_masks[c2] | color_masks[c3]
                if bin(mask).count('1') < 3:
                    is_bad[idx] = True
                    bad_count += 1
            
            # Local Search
            max_steps = 8000
            for step in range(max_steps):
                if time.time() - start_time > time_limit:
                    return None
                if bad_count == 0:
                    return colors
                
                # Pick a random bad AP
                found_bad_ap = False
                for _ in range(15):
                    ap_idx = random.randint(0, M - 1)
                    if is_bad[ap_idx]:
                        ap = aps[ap_idx]
                        # Pick a random position in this AP
                        pos = random.choice(ap)
                        found_bad_ap = True
                        break
                
                if not found_bad_ap:
                    pos = random.randint(0, N - 1)
                
                current_val = colors[pos]
                
                # Try to change to a better value using bitwise operations
                best_val = current_val
                best_delta = 0  # Change in bad_count
                
                for new_val in range(k):
                    if new_val == current_val:
                        continue
                    
                    delta = 0
                    for ap_idx in pos_to_aps[pos]:
                        ap = aps[ap_idx]
                        # Get colors, replacing pos with new_val
                        c0 = colors[ap[0]] if ap[0] != pos else new_val
                        c1 = colors[ap[1]] if ap[1] != pos else new_val
                        c2 = colors[ap[2]] if ap[2] != pos else new_val
                        c3 = colors[ap[3]] if ap[3] != pos else new_val
                        
                        # Use bitwise operations to count distinct colors
                        mask = color_masks[c0] | color_masks[c1] | color_masks[c2] | color_masks[c3]
                        was_bad = is_bad[ap_idx]
                        is_bad_new = bin(mask).count('1') < 3
                        
                        if was_bad and not is_bad_new:
                            delta -= 1
                        elif not was_bad and is_bad_new:
                            delta += 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_val = new_val
                
                # If no improvement, pick random different value
                if best_delta >= 0:
                    best_val = random.randint(0, k - 1)
                    if best_val == current_val:
                        best_val = (best_val + 1) % k
                    
                    # Calculate delta for this random choice
                    delta = 0
                    for ap_idx in pos_to_aps[pos]:
                        ap = aps[ap_idx]
                        c0 = colors[ap[0]] if ap[0] != pos else best_val
                        c1 = colors[ap[1]] if ap[1] != pos else best_val
                        c2 = colors[ap[2]] if ap[2] != pos else best_val
                        c3 = colors[ap[3]] if ap[3] != pos else best_val
                        
                        mask = color_masks[c0] | color_masks[c1] | color_masks[c2] | color_masks[c3]
                        was_bad = is_bad[ap_idx]
                        is_bad_new = bin(mask).count('1') < 3
                        
                        if was_bad and not is_bad_new:
                            delta -= 1
                        elif not was_bad and is_bad_new:
                            delta += 1
                    best_delta = delta
                
                # Apply change
                colors[pos] = best_val
                bad_count += best_delta
                
                # Update is_bad array using bitwise operations
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    mask = color_masks[c0] | color_masks[c1] | color_masks[c2] | color_masks[c3]
                    is_bad[ap_idx] = bin(mask).count('1') < 3
        
        return None
    
    # Try k=3 first, then 4, 5, 6
    global_deadline = time.time() + 45.0
    time_per_k = 10.0
    
    for k in [3, 4, 5, 6]:
        if time.time() > global_deadline:
            break
        sol = solve_with_k(k, time_per_k)
        if sol is not None:
            return sol
    
    # Fallback
    return list(range(N))

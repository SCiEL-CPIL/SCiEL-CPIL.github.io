import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 128
    
    # Precompute all 4-term APs
    aps4 = []
    for d in range(1, (N - 3) // 3 + 1):
        for i in range(0, N - 3 * d):
            aps4.append((i, i + d, i + 2 * d, i + 3 * d))
    
    num_aps = len(aps4)
    
    # For each position, which APs does it participate in?
    pos_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps4):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    # Greedy initial coloring
    def greedy():
        colors = []
        for x in range(1, N + 1):
            pos = x - 1
            for color in range(128):
                ok = True
                for d in range(1, (x - 1) // 3 + 1):
                    if pos - 3 * d < 0:
                        break
                    c0 = colors[pos - 3 * d]
                    c1 = colors[pos - 2 * d]
                    c2 = colors[pos - d]
                    if len({c0, c1, c2, color}) < 3:
                        ok = False
                        break
                if ok:
                    colors.append(color)
                    break
        return colors
    
    colors = greedy()
    
    # Incremental violation tracking
    violation_count = [0] * num_aps
    
    def compute_ap_violation(ap_idx, colors):
        ap = aps4[ap_idx]
        return 1 if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3 else 0
    
    # Initialize violation counts
    total_violations = 0
    for idx in range(num_aps):
        violation_count[idx] = compute_ap_violation(idx, colors)
        total_violations += violation_count[idx]
    
    # Local search to minimize distinct colors
    best_colors = list(colors)
    best_num_colors = len(set(colors))
    
    # Try to eliminate colors
    for outer_iter in range(100):
        if time.time() > deadline:
            break
        
        color_counts = {}
        for c in colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        if len(color_counts) < best_num_colors:
            best_colors = list(colors)
            best_num_colors = len(color_counts)
        
        if not color_counts:
            break
        
        # Try to eliminate least frequent color
        target_color = min(color_counts.keys(), key=lambda c: color_counts[c])
        positions = [i for i in range(N) if colors[i] == target_color]
        random.shuffle(positions)
        
        success = True
        for pos in positions:
            if time.time() > deadline:
                success = False
                break
            
            found = False
            other_colors = [c for c in range(128) if c != target_color]
            random.shuffle(other_colors)
            
            for new_color in other_colors:
                # Check if recoloring pos to new_color is valid
                valid = True
                for ap_idx in pos_aps[pos]:
                    ap = aps4[ap_idx]
                    # Compute new violation for this AP
                    temp = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
                    for j, p in enumerate(ap):
                        if p == pos:
                            temp[j] = new_color
                    if len(set(temp)) < 3:
                        valid = False
                        break
                
                if valid:
                    colors[pos] = new_color
                    # Update violation counts
                    for ap_idx in pos_aps[pos]:
                        old_v = violation_count[ap_idx]
                        new_v = compute_ap_violation(ap_idx, colors)
                        violation_count[ap_idx] = new_v
                        total_violations += (new_v - old_v)
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if not success:
            # Try a different approach: random recoloring
            for _ in range(100):
                if time.time() > deadline:
                    break
                pos = random.randint(0, N - 1)
                new_color = random.randint(0, 127)
                if new_color == colors[pos]:
                    continue
                
                # Check validity
                valid = True
                for ap_idx in pos_aps[pos]:
                    ap = aps4[ap_idx]
                    temp = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
                    for j, p in enumerate(ap):
                        if p == pos:
                            temp[j] = new_color
                    if len(set(temp)) < 3:
                        valid = False
                        break
                
                if valid:
                    colors[pos] = new_color
                    for ap_idx in pos_aps[pos]:
                        old_v = violation_count[ap_idx]
                        new_v = compute_ap_violation(ap_idx, colors)
                        violation_count[ap_idx] = new_v
                        total_violations += (new_v - old_v)
    
    # Final check
    if total_violations > 0:
        colors = greedy()
    
    return colors

import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 128
    
    # Precompute all 4-APs
    aps = []
    aps_by_index = {i: [] for i in range(n)}
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            ap = (i1, i2, i3, i4)
            aps.append(ap)
            aps_by_index[i1].append(ap)
            aps_by_index[i2].append(ap)
            aps_by_index[i3].append(ap)
            aps_by_index[i4].append(ap)
            
    def is_valid(colors):
        for ap in aps:
            i1, i2, i3, i4 = ap
            if len({colors[i1], colors[i2], colors[i3], colors[i4]}) < 3:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))

    # Find initial valid coloring
    initial_colors = None
    
    # Try to find a valid coloring with small K using random + local search
    for k in range(5, 20):
        if time.time() > deadline - 20: break
        found = False
        for attempt in range(20):
            if time.time() > deadline - 20: break
            colors = [random.randint(0, k - 1) for _ in range(n)]
            # Local search to fix violations
            for step in range(10000):
                if time.time() > deadline - 20: break
                # Find a random violation
                viol = None
                for _ in range(5):
                    ap = random.choice(aps)
                    i1, i2, i3, i4 = ap
                    if len({colors[i1], colors[i2], colors[i3], colors[i4]}) < 3:
                        viol = ap
                        break
                if not viol:
                    if is_valid(colors):
                        initial_colors = colors
                        found = True
                        break
                # Change a random pos in viol
                pos = random.choice(viol)
                colors[pos] = random.randint(0, k - 1)
            if found:
                break
        if found:
            break
            
    if not initial_colors:
        # Fallback to 128 colors
        initial_colors = list(range(128))

    current_colors = initial_colors
    current_num_colors = count_colors(current_colors)
    
    # Minimize colors
    while current_num_colors > 3:
        if time.time() > deadline - 5:
            break
            
        used_colors = list(set(current_colors))
        freq = {c: current_colors.count(c) for c in used_colors}
        sorted_colors = sorted(used_colors, key=lambda c: freq[c])
        
        success = False
        for c_to_remove in sorted_colors:
            if time.time() > deadline - 5:
                break
                
            old_colors = current_colors.copy()
            positions = [i for i, c in enumerate(current_colors) if c == c_to_remove]
            random.shuffle(positions)
            
            domain = [c for c in used_colors if c != c_to_remove]
            
            def backtrack_reassign(idx, colors_copy):
                if idx == len(positions):
                    return True
                pos = positions[idx]
                for new_c in domain:
                    colors_copy[pos] = new_c
                    valid = True
                    for ap in aps_by_index[pos]:
                        i1, i2, i3, i4 = ap
                        if len({colors_copy[i1], colors_copy[i2], colors_copy[i3], colors_copy[i4]}) < 3:
                            valid = False
                            break
                    if valid:
                        if backtrack_reassign(idx + 1, colors_copy):
                            return True
                    colors_copy[pos] = old_colors[pos]
                return False

            if backtrack_reassign(0, current_colors):
                current_num_colors -= 1
                success = True
                break
            else:
                current_colors = old_colors
                
        if not success:
            break
            
    return current_colors

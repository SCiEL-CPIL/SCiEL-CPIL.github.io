import time
import random

def construct():
    deadline = time.time() + 85
    
    def count_colors(arr):
        return len(set(arr))
    
    def get_aps_involving(pos, n):
        """Get all 4-term APs that include position pos (0-indexed)"""
        aps = []
        for a in range(n):
            for d in range(1, (n - a - 3) // 3 + 1):
                positions = [a, a+d, a+2*d, a+3*d]
                if pos in positions and max(positions) < n:
                    aps.append(tuple(positions))
        return aps
    
    def check_ap(arr, ap):
        return len({arr[ap[0]], arr[ap[1]], arr[ap[2]], arr[ap[3]]}) >= 3
    
    def check_all_aps(arr, n=128):
        for a in range(n):
            for d in range(1, (n - a - 3) // 3 + 1):
                e = a + 3 * d
                if e >= n:
                    break
                if len({arr[a], arr[a+d], arr[a+2*d], arr[e]}) < 3:
                    return False
        return True
    
    # Start with greedy
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            valid = True
            for d in range(1, (x-1)//3 + 1):
                idx = x - 1 - 3*d
                if idx < 0:
                    continue
                if len({color} | {colors[x-1-d], colors[x-1-2*d], colors[idx]}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            for d in range(1, (x-1)//2 + 1):
                idx1 = x - 1 - d
                idx2 = x - 1 - 2*d
                if idx2 < 0:
                    continue
                if colors[idx1] == color and colors[idx2] == color:
                    valid = False
                    break
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            colors.append(0)
    
    best_colors = list(colors)
    best_count = count_colors(best_colors)
    
    # Precompute APs for each position
    n = 128
    aps_by_pos = [set() for _ in range(n)]
    all_aps = []
    for a in range(n):
        for d in range(1, (n - a - 3) // 3 + 1):
            ap = (a, a+d, a+2*d, a+3*d)
            if ap[3] < n:
                all_aps.append(ap)
                for p in ap:
                    aps_by_pos[p].add(ap)
    
    def check_ap(arr, ap):
        return len({arr[ap[0]], arr[ap[1]], arr[ap[2]], arr[ap[3]]}) >= 3
    
    # Local search
    for outer_iter in range(50):
        if time.time() > deadline:
            break
        
        current = list(best_colors)
        current_count = best_count
        found_improvement = False
        
        for inner_iter in range(500):
            if time.time() > deadline:
                break
            
            pos = random.randint(0, n-1)
            old_color = current[pos]
            
            used_colors = list(set(current))
            
            for new_color in used_colors:
                if new_color == old_color:
                    continue
                
                current[pos] = new_color
                
                # Check only APs involving pos
                valid = True
                for ap in aps_by_pos[pos]:
                    if not check_ap(current, ap):
                        valid = False
                        break
                
                if valid:
                    new_count = count_colors(current)
                    if new_count < current_count:
                        current_count = new_count
                        found_improvement = True
                        break
                    elif new_count == current_count:
                        if random.random() < 0.05:
                            break
                    else:
                        current[pos] = old_color
                else:
                    current[pos] = old_color
            
            if found_improvement:
                break
        
        if found_improvement:
            best_colors = current
            best_count = current_count
    
    return best_colors

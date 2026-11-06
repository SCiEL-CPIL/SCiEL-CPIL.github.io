import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85
    n = 128
    
    # Precompute all 4-term APs
    all_aps = []
    aps_by_pos = [set() for _ in range(n)]
    for a in range(n):
        for d in range(1, (n - a - 3) // 3 + 1):
            e = a + 3 * d
            if e >= n:
                break
            ap = (a, a+d, a+2*d, e)
            all_aps.append(ap)
            for p in ap:
                aps_by_pos[p].add(ap)
    
    # Convert to lists for faster iteration
    aps_by_pos_list = [list(s) for s in aps_by_pos]
    
    def count_colors(arr):
        return len(set(arr))
    
    def check_all_aps(arr):
        for ap in all_aps:
            if len({arr[ap[0]], arr[ap[1]], arr[ap[2]], arr[ap[3]]}) < 3:
                return False
        return True
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        colors = []
        for x in range(1, n + 1):
            pos = x - 1
            found = False
            # Try existing colors first (sorted by usage to prefer common ones)
            color_order = list(range(128))
            rng.shuffle(color_order)
            for color in color_order:
                valid = True
                # Check all APs ending at pos or containing pos
                for ap in aps_by_pos_list[pos]:
                    if pos not in ap:
                        continue
                    others = [ap[i] for i in range(4) if ap[i] != pos]
                    if any(o >= pos for o in others):
                        continue  # Only check APs where all other positions are already colored
                    existing = {colors[o] for o in others}
                    if len({color} | existing) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    found = True
                    break
            if not found:
                # Assign a new color
                colors.append(len(colors) % 128)
        return colors
    
    def local_search(arr, seed, time_budget):
        rng = random.Random(seed)
        current = list(arr)
        current_count = count_colors(current)
        start_time = time.time()
        
        # First: try to eliminate colors by reassigning
        for elim_iter in range(200):
            if time.time() - start_time > time_budget * 0.4:
                break
            
            used = list(set(current))
            if len(used) <= 2:
                break
            
            # Pick a color to try to eliminate (least used first)
            color_counts = {c: current.count(c) for c in used}
            sorted_colors = sorted(used, key=lambda c: color_counts[c])
            
            for target_color in sorted_colors[:3]:  # Try to eliminate up to 3 least-used colors
                if time.time() - start_time > time_budget * 0.4:
                    break
                
                target_positions = [i for i, c in enumerate(current) if c == target_color]
                if not target_positions:
                    continue
                
                # Try to reassign all target_color positions to other colors
                old_state = list(current)
                success = True
                
                for pos in target_positions:
                    found_new = False
                    for new_color in used:
                        if new_color == target_color:
                            continue
                        current[pos] = new_color
                        valid = True
                        for ap in aps_by_pos_list[pos]:
                            if len({current[ap[0]], current[ap[1]], current[ap[2]], current[ap[3]]}) < 3:
                                valid = False
                                break
                        if valid:
                            found_new = True
                            break
                    if not found_new:
                        current = old_state
                        success = False
                        break
                
                if success:
                    new_count = count_colors(current)
                    if new_count < current_count:
                        current_count = new_count
                        used = list(set(current))
        
        # Now: single-position swaps to reduce color count
        for swap_iter in range(5000):
            if time.time() - start_time > time_budget:
                break
            
            pos = rng.randint(0, n - 1)
            old_color = current[pos]
            used = list(set(current))
            
            if len(used) <= 2:
                break
            
            # Try reassigning pos to a color that's already used
            for new_color in used:
                if new_color == old_color:
                    continue
                current[pos] = new_color
                valid = True
                for ap in aps_by_pos_list[pos]:
                    if len({current[ap[0]], current[ap[1]], current[ap[2]], current[ap[3]]}) < 3:
                        valid = False
                        break
                if valid:
                    new_count = count_colors(current)
                    if new_count < current_count:
                        current_count = new_count
                        break
                    elif new_count == current_count and rng.random() < 0.1:
                        break
                    else:
                        current[pos] = old_color
                else:
                    current[pos] = old_color
        
        return current, current_count
    
    # Generate initial solutions with different seeds
    best_arr = None
    best_count = 128
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        arr = greedy_construct(seed)
        arr, cnt = local_search(arr, seed, 5.0)
        
        if cnt < best_count:
            best_count = cnt
            best_arr = list(arr)
    
    # If we still have room, do more targeted searches
    if time.time() < deadline - 15 and best_arr is not None:
        for extra_seed in range(10):
            if time.time() > deadline - 5:
                break
            
            # Perturb the best solution
            perturbed = list(best_arr)
            rng = random.Random(1000 + extra_seed)
            for _ in range(rng.randint(1, 5)):
                pos = rng.randint(0, n - 1)
                color = rng.randint(0, 127)
                perturbed[pos] = color
            
            arr, cnt = local_search(perturbed, 1000 + extra_seed, 8.0)
            if cnt < best_count:
                best_count = cnt
                best_arr = list(arr)
    
    if best_arr is None:
        # Fallback: simple greedy
        best_arr = greedy_construct(42)
    
    # Final verification and cleanup
    if not check_all_aps(best_arr):
        # Regenerate with more careful greedy
        for seed in range(50):
            arr = greedy_construct(seed)
            if check_all_aps(arr):
                best_arr = arr
                break
    
    return best_arr

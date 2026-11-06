import time
import random

def construct():
    deadline = time.time() + 85
    n = 128
    
    # Precompute all 4-term APs
    all_aps = []
    for a in range(n):
        for d in range(1, (n - a - 3) // 3 + 1):
            e = a + 3 * d
            if e < n:
                all_aps.append((a, a+d, a+2*d, e))
    
    # Precompute APs by position for fast delta evaluation
    aps_by_pos = [[] for _ in range(n)]
    for ap in all_aps:
        for p in ap:
            aps_by_pos[p].append(ap)
    
    def count_colors(arr):
        return len(set(arr))
    
    def is_valid(arr):
        for ap in all_aps:
            if len({arr[ap[0]], arr[ap[1]], arr[ap[2]], arr[ap[3]]}) < 3:
                return False
        return True
    
    def penalty(arr):
        """Count of violated APs (APs with < 3 distinct colors)"""
        pen = 0
        for ap in all_aps:
            if len({arr[ap[0]], arr[ap[1]], arr[ap[2]], arr[ap[3]]}) < 3:
                pen += 1
        return pen
    
    def delta_penalty(arr, pos, old_c, new_c):
        """Compute change in penalty when changing arr[pos] from old_c to new_c"""
        if old_c == new_c:
            return 0
        dp = 0
        for ap in aps_by_pos[pos]:
            # Count distinct colors before and after
            old_set = {arr[ap[0]], arr[ap[1]], arr[ap[2]], arr[ap[3]]}
            new_set = set(old_set)
            new_set.discard(old_c)
            new_set.add(new_c)
            # If the other 3 positions (excluding pos) already have >= 3 colors,
            # then the AP is valid regardless of pos's color
            others = [arr[ap[i]] for i in range(4) if ap[i] != pos]
            other_colors = set(others)
            if len(other_colors) >= 3:
                continue  # AP is valid regardless
            # Both old and new states depend on pos
            old_valid = len(old_set) >= 3
            new_valid = len(new_set) >= 3
            if not old_valid and new_valid:
                dp -= 1
            elif old_valid and not new_valid:
                dp += 1
        return dp
    
    # Greedy initialization: assign colors to minimize immediate violations
    def greedy_init(max_colors):
        arr = [0] * n
        for x in range(n):
            best_c = 0
            best_pen = float('inf')
            for c in range(max_colors):
                arr[x] = c
                # Count violations involving position x
                pen = 0
                for ap in aps_by_pos[x]:
                    if len({arr[ap[0]], arr[ap[1]], arr[ap[2]], arr[ap[3]]}) < 3:
                        pen += 1
                if pen < best_pen:
                    best_pen = pen
                    best_c = c
                if best_pen == 0:
                    break
            arr[x] = best_c
        return arr
    
    # Simulated annealing
    def simulated_annealing(init_arr, max_colors, start_temp=10.0, end_temp=0.01, steps=200000):
        current = list(init_arr)
        current_pen = penalty(current)
        current_colors = count_colors(current)
        
        best = list(current)
        best_pen = current_pen
        best_colors = current_colors
        
        t = start_temp
        cool_rate = (end_temp / start_temp) ** (1.0 / max(steps, 1))
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            pos = random.randint(0, n - 1)
            old_c = current[pos]
            
            # Choose new color: prefer colors already in use to reduce color count
            if random.random() < 0.7:
                used = list(set(current))
                new_c = random.choice(used)
            else:
                new_c = random.randint(0, max_colors - 1)
            
            if new_c == old_c:
                continue
            
            dp = delta_penalty(current, pos, old_c, new_c)
            new_pen = current_pen + dp
            
            # Accept if better, or with probability exp(-dp/temp)
            if dp < 0:
                accept = True
            else:
                if dp == 0:
                    # Accept neutral moves to explore
                    accept = random.random() < 0.3
                else:
                    accept = random.random() < (2.0 ** (-dp / max(t, 0.001)))
            
            if accept:
                current[pos] = new_c
                current_pen = new_pen
                current_colors = count_colors(current)
                
                if current_pen == 0 and current_colors < best_colors:
                    best = list(current)
                    best_pen = 0
                    best_colors = current_colors
                elif current_pen < best_pen:
                    best = list(current)
                    best_pen = current_pen
                    best_colors = current_colors
            
            t *= cool_rate
        
        return best, best_pen, best_colors
    
    # Try different numbers of colors with simulated annealing
    best_result = None
    best_result_colors = 128
    
    # Start with a reasonable upper bound
    init = greedy_init(128)
    init_pen = penalty(init)
    init_colors = count_colors(init)
    if init_pen == 0:
        best_result = list(init)
        best_result_colors = init_colors
    
    # Try simulated annealing with various max_color bounds
    for max_c in [4, 5, 6, 7, 8, 10]:
        if time.time() > deadline - 10:
            break
        
        if max_c >= best_result_colors:
            continue
        
        init_c = greedy_init(max_c)
        init_pen_c = penalty(init_c)
        init_colors_c = count_colors(init_c)
        
        if init_pen_c == 0 and init_colors_c < best_result_colors:
            best_result = list(init_c)
            best_result_colors = init_colors_c
            continue
        
        # Run SA
        sa_best, sa_pen, sa_colors = simulated_annealing(
            init_c, max_c, start_temp=5.0, end_temp=0.005, steps=150000
        )
        
        if sa_pen == 0 and sa_colors < best_result_colors:
            best_result = sa_best
            best_result_colors = sa_colors
        elif sa_pen == 0 and best_result is None:
            best_result = sa_best
            best_result_colors = sa_colors
    
    # If we still have violations, try higher max_colors
    if best_result is None or penalty(best_result) > 0:
        for max_c in [12, 16, 20]:
            if time.time() > deadline - 5:
                break
            
            init_c = greedy_init(max_c)
            init_pen_c = penalty(init_c)
            init_colors_c = count_colors(init_c)
            
            if init_pen_c == 0 and (best_result is None or init_colors_c < best_result_colors):
                best_result = list(init_c)
                best_result_colors = init_colors_c
                continue
            
            sa_best, sa_pen, sa_colors = simulated_annealing(
                init_c, max_c, start_temp=8.0, end_temp=0.01, steps=100000
            )
            
            if sa_pen == 0 and (best_result is None or sa_colors < best_result_colors):
                best_result = sa_best
                best_result_colors = sa_colors
    
    # Final verification and cleanup: ensure labels are in 0..127
    if best_result is None:
        best_result = greedy_init(128)
    
    # Remap colors to be in 0..127 (they should already be, but ensure)
    unique_colors = sorted(set(best_result))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    best_result = [color_map[c] for c in best_result]
    
    # Verify validity
    if not is_valid(best_result):
        # Fallback: use a known-valid pattern (e.g., modular coloring)
        # Use 4 colors based on index mod 4 - this should satisfy the constraint
        # Actually let's use a safer approach
        for c in range(3, 20):
            candidate = [i % c for i in range(n)]
            if is_valid(candidate):
                best_result = candidate
                break
        else:
            # Last resort: all distinct
            best_result = list(range(n))
    
    return best_result

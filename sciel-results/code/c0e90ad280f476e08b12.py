import random
import time

def construct():
    N = 128
    # Precompute affected APs for each index
    affected_aps = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            for idx in ap:
                others = [j for j in ap if j != idx]
                affected_aps[idx].append(others)

    def count_violations(x):
        viol = 0
        for d in range(1, N // 3 + 1):
            for a in range(0, N - 3 * d):
                colors = {x[a], x[a+d], x[a+2*d], x[a+3*d]}
                if len(colors) < 3:
                    viol += 1
        return viol

    def get_delta_violations(x, i, new_color):
        old_color = x[i]
        if old_color == new_color:
            return 0
        
        delta = 0
        for others in affected_aps[i]:
            c1, c2, c3 = x[others[0]], x[others[1]], x[others[2]]
            
            set_before = {old_color, c1, c2, c3}
            valid_before = len(set_before) >= 3
            
            set_after = {new_color, c1, c2, c3}
            valid_after = len(set_after) >= 3
            
            if valid_before and not valid_after:
                delta += 1
            elif not valid_before and valid_after:
                delta -= 1
        return delta

    start_time = time.time()
    time_limit = 55.0

    best_solution = None
    best_distinct = N

    # Try to find a valid solution with K=3, then K=4, etc.
    for K in range(3, 8):
        for restart in range(100):
            if time.time() - start_time > time_limit:
                break
            
            x = [random.randint(0, K - 1) for _ in range(N)]
            viol = count_violations(x)
            
            # Phase 1: Reduce violations to 0
            T = 1.0
            for step in range(20000):
                if time.time() - start_time > time_limit:
                    break
                
                i = random.randint(0, N - 1)
                new_color = random.randint(0, K - 1)
                
                delta = get_delta_violations(x, i, new_color)
                
                if delta < 0:
                    x[i] = new_color
                    viol += delta
                elif delta == 0:
                    if random.random() < 0.5:
                        x[i] = new_color
                else:
                    T *= 0.995
                    if random.random() < pow(2.718, -delta / max(T, 0.01)):
                        x[i] = new_color
                        viol += delta
                
                if viol == 0:
                    break

            if viol == 0:
                # Phase 2: Compress colors
                # Try to eliminate the least frequent colors
                for _ in range(50): # Try to eliminate up to 50 colors? No, just loop while possible
                    if time.time() - start_time > time_limit:
                        break
                        
                    counts = {}
                    for c in x:
                        counts[c] = counts.get(c, 0) + 1
                    
                    if len(counts) <= 3:
                        break
                        
                    # Find least frequent color
                    min_color = min(counts, key=counts.get)
                    min_count = counts[min_color]
                    
                    # Try to recolor all instances of min_color to other colors
                    success = True
                    positions_to_change = [i for i, c in enumerate(x) if c == min_color]
                    random.shuffle(positions_to_change)
                    
                    for i in positions_to_change:
                        # Try to find a new color that doesn't increase violations
                        found = False
                        # Try colors other than min_color
                        other_colors = [c for c in range(K) if c != min_color]
                        random.shuffle(other_colors)
                        
                        for new_c in other_colors:
                            delta = get_delta_violations(x, i, new_c)
                            if delta == 0:
                                x[i] = new_c
                                found = True
                                break
                            elif delta < 0: # Should not happen if valid, but just in case
                                x[i] = new_c
                                found = True
                                break
                        if not found:
                            # Try random color in range K
                            new_c = random.randint(0, K - 1)
                            if new_c != min_color:
                                delta = get_delta_violations(x, i, new_c)
                                if delta == 0:
                                    x[i] = new_c
                                    found = True
                            
                        if not found:
                            success = False
                            break
                    
                    if not success:
                        break # Can't eliminate this color, try next or stop

                distinct = len(set(x))
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_solution = x[:]
                
                if best_distinct == 3:
                    return best_solution

    if best_solution is not None:
        return best_solution
    else:
        return [random.randint(0, 5) for _ in range(N)]

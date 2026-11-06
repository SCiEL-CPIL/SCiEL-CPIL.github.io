import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 128
    
    pos_to_aps = [[] for _ in range(n)]
    all_aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            if a + 3 * d <= n:
                all_aps.append((a, d))
                for k in range(4):
                    p = a + k * d
                    pos_to_aps[p - 1].append((a, d))
    
    def ap_violates(a, d, colors):
        return len({colors[a-1], colors[a+d-1], colors[a+2*d-1], colors[a+3*d-1]}) < 3
    
    def count_violations(colors):
        return sum(1 for (a, d) in all_aps if ap_violates(a, d, colors))
    
    best_colors = None
    best_viol = float('inf')
    
    for nc in [3, 4, 5, 6]:
        if time.time() > deadline - 10:
            break
        
        for trial in range(400):
            if time.time() > deadline - 15:
                break
            
            if trial == 0:
                colors = [i % nc for i in range(n)]
            elif trial == 1:
                colors = [(i * 7) % nc for i in range(n)]
            elif trial == 2:
                colors = [(i * 13) % nc for i in range(n)]
            elif trial == 3:
                colors = [(i * 5) % nc for i in range(n)]
            elif trial == 4:
                colors = [(i * 11) % nc for i in range(n)]
            else:
                colors = [random.randint(0, nc - 1) for _ in range(n)]
            
            current_viol = count_violations(colors)
            if current_viol < best_viol:
                best_viol = current_viol
                best_colors = colors[:]
                if current_viol == 0:
                    return colors[:]
            
            for it in range(15000):
                if time.time() > deadline - 15:
                    break
                
                pos = random.randint(0, n - 1)
                old = colors[pos]
                new = random.randint(0, nc - 1)
                if new == old:
                    continue
                
                affected = pos_to_aps[pos]
                old_local = sum(1 for (a, d) in affected if ap_violates(a, d, colors))
                
                colors[pos] = new
                new_local = sum(1 for (a, d) in affected if ap_violates(a, d, colors))
                
                if new_local <= old_local:
                    if it % 500 == 0:
                        current_viol = count_violations(colors)
                        if current_viol < best_viol:
                            best_viol = current_viol
                            best_colors = colors[:]
                            if current_viol == 0:
                                return colors[:]
                else:
                    colors[pos] = old
            
            current_viol = count_violations(colors)
            if current_viol < best_viol:
                best_viol = current_viol
                best_colors = colors[:]
                if current_viol == 0:
                    return colors[:]
        
        if best_viol == 0:
            return best_colors
    
    return best_colors if best_colors else [i % 4 for i in range(n)]

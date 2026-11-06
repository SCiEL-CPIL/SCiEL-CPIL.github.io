import random
import time
import math

def construct():
    n = 128
    # Precompute all 4-APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map index to list of AP indices it belongs to
    ap_idx_map = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_idx_map[pos].append(idx)

    def count_violations(colors):
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                v += 1
        return v

    def delta_violations(colors, idx, new_color):
        old_color = colors[idx]
        if old_color == new_color:
            return 0
        
        delta = 0
        for ap_idx in ap_idx_map[idx]:
            ap = aps[ap_idx]
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            old_set = {c0, c1, c2, c3}
            old_viol = 1 if len(old_set) < 3 else 0
            
            # Update colors
            if ap[0] == idx: c0 = new_color
            elif ap[1] == idx: c1 = new_color
            elif ap[2] == idx: c2 = new_color
            elif ap[3] == idx: c3 = new_color
            
            new_set = {c0, c1, c2, c3}
            new_viol = 1 if len(new_set) < 3 else 0
            
            delta += (new_viol - old_viol)
            
        return delta

    # Multiple restarts to increase chance of finding 0
    start_time = time.time()
    timeout = 85.0
    best_colors = None
    best_viol = float('inf')
    
    while time.time() - start_time < timeout:
        random.seed(int(time.time() * 1000) % 100000)
        colors = [random.randint(0, 2) for _ in range(n)]
        current_viol = count_violations(colors)
        
        if current_viol < best_viol:
            best_viol = current_viol
            best_colors = colors[:]
            if best_viol == 0:
                break

        T = 5.0
        cooling = 0.9995
        steps = 0
        max_steps = 5000
        
        while steps < max_steps and time.time() - start_time < timeout:
            if current_viol < best_viol:
                best_viol = current_viol
                best_colors = colors[:]
                if best_viol == 0:
                    break
            
            idx = random.randint(0, n - 1)
            old_color = colors[idx]
            new_color = random.randint(0, 2)
            if new_color == old_color:
                continue
                
            d_viol = delta_violations(colors, idx, new_color)
            
            if d_viol < 0:
                colors[idx] = new_color
                current_viol += d_viol
            else:
                if d_viol == 0:
                    prob = 0.5
                else:
                    prob = math.exp(-d_viol / T)
                
                if random.random() < prob:
                    colors[idx] = new_color
                    current_viol += d_viol
            
            T *= cooling
            steps += 1
        
        if best_viol == 0:
            break

    return best_colors

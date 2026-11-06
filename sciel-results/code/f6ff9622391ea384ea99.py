import random
import time

def construct():
    N = 128
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
            
    index_to_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for idx in ap:
            index_to_aps[idx].append(i)

    def is_bad(colors, ap):
        return len(set(colors[j] for j in ap)) < 3

    def calc_cost(colors):
        return sum(1 for ap in aps if is_bad(colors, ap))

    def local_search(k, time_limit):
        start_time = time.time()
        best_cost = float('inf')
        best_colors = None
        
        while time.time() - start_time < time_limit:
            colors = [random.randint(0, k - 1) for _ in range(N)]
            cost = calc_cost(colors)
            
            if cost == 0:
                return colors
            
            temp = 1.0
            min_temp = 0.001
            cooling = 0.9999
            
            while time.time() - start_time < time_limit:
                if cost < best_cost:
                    best_cost = cost
                    best_colors = colors[:]
                    if cost == 0:
                        return colors
                
                i = random.randint(0, N - 1)
                old_color = colors[i]
                if k == 1:
                    continue
                new_color = random.randint(0, k - 1)
                if new_color == old_color:
                    continue
                
                delta = 0
                for ap_idx in index_to_aps[i]:
                    ap = aps[ap_idx]
                    old_dist = len(set(colors[j] for j in ap))
                    colors[i] = new_color
                    new_dist = len(set(colors[j] for j in ap))
                    colors[i] = old_color
                    
                    old_v = 1 if old_dist < 3 else 0
                    new_v = 1 if new_dist < 3 else 0
                    delta += new_v - old_v
                
                if delta < 0 or (random.random() < pow(2.718, -delta / temp)):
                    colors[i] = new_color
                    cost += delta
                
                temp *= cooling
                if temp < min_temp:
                    break
            
            if best_cost == 0:
                return best_colors
                
        return best_colors

    deadline = time.time() + 85
    limit_k3 = min(40, deadline - time.time())
    if limit_k3 <= 0:
        return [random.randint(0, 3) for _ in range(N)]
        
    best_k3 = local_search(3, limit_k3)
    
    deadline = time.time() + 85
    limit_k4 = min(40, deadline - time.time())
    if limit_k4 > 0:
        best_k4 = local_search(4, limit_k4)
    else:
        best_k4 = None

    def is_valid(colors):
        if not colors: return False
        return all(not is_bad(colors, ap) for ap in aps)

    if best_k3 and is_valid(best_k3):
        return best_k3
    if best_k4 and is_valid(best_k4):
        return best_k4
        
    # Fallback
    if best_k4: return best_k4
    if best_k3: return best_k3
    return [random.randint(0, 3) for _ in range(N)]

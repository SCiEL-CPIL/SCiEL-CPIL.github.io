import time
import random
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    k4s = []
    for verts in itertools.combinations(range(n), 4):
        e_indices = [edge_index[(u, v)] for u, v in itertools.combinations(verts, 2)]
        k4s.append(e_indices)
        
    num_k4s = len(k4s)
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4s):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)
            
    def calculate_cost(colors):
        cost = 0
        for e_indices in k4s:
            seen = set()
            for idx in e_indices:
                seen.add(colors[idx])
            if len(seen) < 5:
                cost += 1
        return cost
        
    def get_delta_cost(e_idx, old_c, new_c, colors, k4_indices):
        if old_c == new_c:
            return 0
        delta = 0
        for k4_idx in k4_indices:
            e_indices = k4s[k4_idx]
            old_colors = set()
            new_colors = set()
            for idx in e_indices:
                if idx == e_idx:
                    old_colors.add(old_c)
                    new_colors.add(new_c)
                else:
                    c = colors[idx]
                    old_colors.add(c)
                    new_colors.add(c)
            old_distinct = len(old_colors)
            new_distinct = len(new_colors)
            old_contrib = 1 if old_distinct < 5 else 0
            new_contrib = 1 if new_distinct < 5 else 0
            delta += new_contrib - old_contrib
        return delta

    def run_sa(K, time_limit):
        # Multiple restarts
        restarts = 0
        max_restarts = 5
        time_per_restart = time_limit / max_restarts
        
        best_cost_overall = float('inf')
        best_colors_overall = None
        
        for r in range(max_restarts):
            if time.time() > start_time + time_limit:
                break
                
            colors = [random.randint(0, K - 1) for _ in range(num_edges)]
            current_cost = calculate_cost(colors)
            
            best_colors_local = colors[:]
            best_cost_local = current_cost
            
            temp = 1.0
            cooling = 0.999
            min_temp = 0.01
            end_time = time.time() + time_per_restart
            
            while time.time() < end_time:
                e_idx = random.randint(0, num_edges - 1)
                old_c = colors[e_idx]
                new_c = random.randint(0, K - 1)
                while new_c == old_c:
                    new_c = random.randint(0, K - 1)
                    
                delta = get_delta_cost(e_idx, old_c, new_c, colors, edge_to_k4s[e_idx])
                
                if delta < 0:
                    colors[e_idx] = new_c
                    current_cost += delta
                    if current_cost < best_cost_local:
                        best_cost_local = current_cost
                        best_colors_local = colors[:]
                        if best_cost_local == 0:
                            return best_colors_local, 0
                else:
                    if delta == 0:
                        colors[e_idx] = new_c
                    else:
                        if temp > min_temp:
                            ratio = delta / temp
                            if ratio < 100:
                                if random.random() < math.exp(-ratio):
                                    colors[e_idx] = new_c
                                    current_cost += delta
                temp *= cooling
                if temp < min_temp:
                    temp = min_temp
                    
            if best_cost_local < best_cost_overall:
                best_cost_overall = best_cost_local
                best_colors_overall = best_colors_local
                if best_cost_overall == 0:
                    return best_colors_overall, 0
                    
        return best_colors_overall, best_cost_overall

    # Try K=13
    t_start = time.time()
    colors, cost = run_sa(13, 40)
    if cost == 0:
        return colors
        
    # Try K=14
    t_start = time.time()
    colors, cost = run_sa(14, 40)
    if cost == 0:
        return colors
        
    # Try K=15
    t_start = time.time()
    colors, cost = run_sa(15, 40)
    if cost == 0:
        return colors
        
    # Fallback: Seed's logic or simple 17
    # We can try to apply the seed's matching to the 17-color base
    # But for now, return 17.
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

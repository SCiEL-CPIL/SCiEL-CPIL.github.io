"""EV-routing solver: Beam Search with Dijkstra-like expansion."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import heapq
import math


def _euclidean(coords):
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def plan(instance):
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    dist_mat = _euclidean(coords)
    
    customers = list(range(1, n_customers + 1))
    m = len(customers)
    full_mask = (1 << m) - 1
    cust_to_bit = {c: i for i, c in enumerate(customers)}
    
    # Heuristic
    def heuristic(mask, node):
        if mask == full_mask:
            return dist_mat[node, 0]
        rem = [c for c in customers if not (mask & (1 << cust_to_bit[c]))]
        if not rem:
            return dist_mat[node, 0]
        min_out = min(dist_mat[node, c] for c in rem)
        min_in = min(dist_mat[c, 0] for c in rem)
        return min_out + min_in

    # Beam Search
    # State: (f, g, node, mask, battery, time, load, path)
    beam_width = 200
    max_depth = 300 # Safety
    
    # Initial state
    start = (heuristic(0, 0), 0.0, 0, 0, float(battery_max), 0.0, 0.0, [0])
    
    # We use a list of states, sorted by f
    # To manage beam, we keep the best 'beam_width' states at each "step"?
    # Since steps vary (stations), we just keep a global priority queue and prune.
    
    pq = [start]
    heapq.heapify(pq)
    
    best_sol_dist = float('inf')
    best_sol_path = None
    visited_states = set() # (node, mask) to avoid duplicates? No, battery/time vary.
    
    expansions = 0
    max_expansions = 10000
    
    while pq and expansions < max_expansions:
        f, g, node, mask, battery, time, load, path = heapq.heappop(pq)
        expansions += 1
        
        # Goal check
        if mask == full_mask:
            d_back = dist_mat[node, 0]
            t_back = time + d_back / speed
            bat_back = battery - d_back
            if bat_back >= -1e-9 and t_back <= horizon + 1e-9:
                total_dist = g + d_back
                if total_dist < best_sol_dist:
                    best_sol_dist = total_dist
                    best_sol_path = path + [0]
        
        # Expand
        # To limit branching, we only consider the top K neighbors?
        # Or all feasible neighbors.
        
        for j in range(1, n_nodes):
            if j == 0: continue
            
            is_cust = j in cust_to_bit
            is_stat = j in stations
            
            if is_cust and (mask & (1 << cust_to_bit[j])):
                continue
                
            d = dist_mat[node, j]
            t_arr = time + d / speed
            bat_arr = battery - d
            new_load = load
            
            if bat_arr < -1e-9:
                continue
                
            if is_cust:
                if t_arr < ready[j]: t_arr = ready[j]
                if t_arr > due[j] + 1e-9: continue
                new_load += demands[j]
                if new_load > capacity + 1e-9: continue
                t_dep = t_arr + service[j]
            else:
                t_dep = t_arr
                
            if is_stat:
                # Charge to full
                if bat_arr < battery_max:
                    charge_time = (battery_max - bat_arr) * recharge_rate
                    t_dep += charge_time
                    bat_arr = battery_max
            
            new_mask = mask | (1 << cust_to_bit[j]) if is_cust else mask
            new_g = g + d
            new_f = new_g + heuristic(new_mask, j)
            
            # Pruning: If we've seen this (node, mask) with better g, time, battery?
            # Simple check: If new_f > best_sol_dist, skip.
            if new_f >= best_sol_dist:
                continue
                
            heapq.heappush(pq, (new_f, new_g, j, new_mask, bat_arr, t_dep, new_load, path + [j]))
            
            # Beam pruning: Keep only best 'beam_width' states in pq?
            # This is tricky with a single pq.
            # We can just let it run and rely on max_expansions.
            
    if best_sol_path:
        # Extract customers
        route = [n for n in best_sol_path if n in cust_to_bit]
        return [route]
        
    # Fallback
    return [[c] for c in customers]

# EVOLVE-BLOCK-END

"""Reformulated EV routing: Dijkstra on state space for clustered customers."""
# EVOLVE-BLOCK-START
import numpy as np
import heapq
import math

def plan(instance):
    n_cust = instance["n_customers"]
    coords = np.array(instance["coords"])
    demands = np.array(instance["demands"])
    ready = np.array(instance["ready"])
    due = np.array(instance["due"])
    service = np.array(instance["service"])
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    if n_cust == 0:
        return []
        
    customer_ids = list(range(1, n_cust + 1))
    
    # Estimate K
    total_demand = sum(demands[1:n_cust+1])
    min_vehicles = max(1, math.ceil(total_demand / capacity))
    K = min_vehicles + 1
    K = min(K, n_cust)
    
    # Iterative clustering with capacity check
    while True:
        # Simple K-Means-like clustering
        # Initialize centroids
        centroids = []
        remaining = list(customer_ids)
        
        # Pick first centroid as the one farthest from depot? Or just first?
        # Let's pick the one with the earliest due date to prioritize tight windows
        first_c = min(customer_ids, key=lambda c: due[c])
        centroids.append(first_c)
        remaining.remove(first_c)
        
        for i in range(1, K):
            if not remaining:
                break
            # Pick the customer that maximizes min distance to existing centroids
            best_c = -1
            best_min_dist = -1
            for c in remaining:
                min_d = min(math.sqrt(np.sum((coords[c] - coords[cent])**2)) for cent in centroids)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_c = c
            if best_c != -1:
                centroids.append(best_c)
                remaining.remove(best_c)
                
        # Assign customers to nearest centroid
        clusters = [[] for _ in range(len(centroids))]
        for c in customer_ids:
            dists = [math.sqrt(np.sum((coords[c] - coords[cent])**2)) for cent in centroids]
            min_idx = np.argmin(dists)
            clusters[min_idx].append(c)
            
        # Check capacity constraints
        valid = True
        for cluster in clusters:
            if sum(demands[c] for c in cluster) > capacity:
                valid = False
                break
                
        if valid:
            break
            
        # If not valid, increase K
        K += 1
        if K > n_cust:
            # Fallback: one customer per route
            clusters = [[c] for c in customer_ids]
            break

    routes = []
    for cluster in clusters:
        if not cluster:
            continue
        route = solve_cluster(cluster, coords, demands, ready, due, service, stations, capacity, battery_max, speed, recharge_rate, horizon, n_cust)
        if route:
            routes.append(route)
            
    return routes

def solve_cluster(cluster_ids, coords, demands, ready, due, service, stations, capacity, battery_max, speed, recharge_rate, horizon, n_cust):
    m = len(cluster_ids)
    if m == 0:
        return []
        
    id_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
    full_mask = (1 << m) - 1
    
    # Labels: dict mapping (node, mask) to list of (time, battery, cost, path)
    labels = {}
    
    def is_dominated(node, mask, new_time, new_battery, new_cost):
        if (node, mask) not in labels:
            return False
        for t, b, c, p in labels[(node, mask)]:
            if t <= new_time and b >= new_battery and c <= new_cost:
                return True
        return False
        
    def add_label(node, mask, time, battery, cost, path):
        key = (node, mask)
        if key not in labels:
            labels[key] = []
        
        # Remove labels dominated by the new one
        new_list = []
        for t, b, c, p in labels[key]:
            if not (time <= t and battery >= b and cost <= c):
                new_list.append((t, b, c, p))
        
        # Check if new one is dominated by any remaining
        if is_dominated(node, mask, time, battery, cost):
            return False
            
        new_list.append((time, battery, cost, path))
        labels[key] = new_list
        return True

    start_node = 0
    start_mask = 0
    start_time = 0.0
    start_battery = battery_max
    start_cost = 0.0
    start_path = [0]
    
    add_label(start_node, start_mask, start_time, start_battery, start_cost, start_path)
    
    pq = []
    heapq.heappush(pq, (start_cost, start_node, start_mask, start_time, start_battery, start_path))
    
    best_final_cost = float('inf')
    best_final_path = None
    
    # Pre-compute distances from all relevant nodes to all relevant nodes?
    # Relevant: 0, cluster_ids, stations
    relevant_nodes = [0] + cluster_ids + list(stations)
    unique_relevant = list(set(relevant_nodes))
    n_rel = len(unique_relevant)
    node_to_idx_rel = {n: i for i, n in enumerate(unique_relevant)}
    dist_mat = np.zeros((n_rel, n_rel))
    for i in range(n_rel):
        for j in range(n_rel):
            dx = coords[unique_relevant[i]][0] - coords[unique_relevant[j]][0]
            dy = coords[unique_relevant[i]][1] - coords[unique_relevant[j]][1]
            dist_mat[i][j] = math.sqrt(dx*dx + dy*dy)
            
    def get_dist(n1, n2):
        return dist_mat[node_to_idx_rel[n1]][node_to_idx_rel[n2]]

    while pq:
        cost, node, mask, time, battery, path = heapq.heappop(pq)
        
        # Check if this label is still valid (not dominated by a better one found later)
        # We check if any label in labels[(node, mask)] dominates this one
        key = (node, mask)
        if key in labels:
            dominated = False
            for t, b, c, p in labels[key]:
                if t <= time and b >= battery and c <= cost:
                    # If it's the exact same label, it's fine? 
                    # If it's a different label that dominates, skip.
                    # To avoid skipping the "current" best, we can check identity?
                    # Or just accept that we might process some redundant states.
                    # Given small size, this is okay.
                    dominated = True
                    break
            if dominated:
                continue
                
        if mask == full_mask:
            # Try to return to depot
            d = get_dist(node, 0)
            if battery >= d:
                new_battery = battery - d
                new_time = time + d / speed
                new_cost = cost + d
                if new_time <= horizon:
                    if new_cost < best_final_cost:
                        best_final_cost = new_cost
                        best_final_path = path + [0]
            continue
            
        # Expand neighbors
        next_nodes = []
        for cid in cluster_ids:
            if not (mask & (1 << id_to_idx[cid])):
                next_nodes.append(cid)
        for s in stations:
            next_nodes.append(s)
            
        for next_node in next_nodes:
            d = get_dist(node, next_node)
            if battery < d:
                continue
                
            new_battery = battery - d
            new_time = time + d / speed
            new_cost = cost + d
            
            if next_node in id_to_idx:
                # Customer
                if new_time > due[next_node]:
                    continue
                new_time = max(new_time, ready[next_node])
                new_time += service[next_node]
                new_mask = mask | (1 << id_to_idx[next_node])
                
                # Check capacity
                current_demand = sum(demands[c] for i, c in enumerate(cluster_ids) if mask & (1 << i))
                if current_demand + demands[next_node] > capacity:
                    continue
            else:
                # Station
                recharge_time = (battery_max - new_battery) * recharge_rate
                new_time += recharge_time
                new_battery = battery_max
                new_mask = mask
                
            if is_dominated(next_node, new_mask, new_time, new_battery, new_cost):
                continue
                
            heapq.heappush(pq, (new_cost, next_node, new_mask, new_time, new_battery, path + [next_node]))
            add_label(next_node, new_mask, new_time, new_battery, new_cost, path + [next_node])
            
    if best_final_path:
        return best_final_path[1:-1]
    else:
        return []
# EVOLVE-BLOCK-END

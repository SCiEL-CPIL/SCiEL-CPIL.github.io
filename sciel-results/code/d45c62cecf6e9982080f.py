"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import sqrt

def _build_distance_matrix(coords):
    """Precompute Euclidean distance matrix."""
    coords = np.array(coords)
    x_diff = coords[:, 0, np.newaxis] - coords[np.newaxis, :, 0]
    y_diff = coords[:, 1, np.newaxis] - coords[np.newaxis, :, 1]
    dist_matrix = np.sqrt(x_diff**2 + y_diff**2)
    return dist_matrix

def _simulate_route(route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
    """
    Simulate a route and return (is_valid, total_distance, total_time).
    """
    if not route:
        return True, 0.0, 0.0
        
    cur_node = 0
    cur_time = 0.0
    cur_bat = battery
    total_dist = 0.0
    load = 0.0
    
    for node in route:
        d = dist_matrix[cur_node, node]
        cur_time += d / speed
        cur_bat -= d
        total_dist += d
        
        if cur_bat < -1e-9:
            return False, total_dist, cur_time
        
        if node in stations_set:
            # Recharge
            energy_needed = battery - cur_bat
            if energy_needed > 0:
                cur_time += energy_needed * recharge_rate
                cur_bat = battery
        else:
            # Customer
            if cur_time < ready[node]:
                cur_time = ready[node]
            if cur_time > due[node] + 1e-9:
                return False, total_dist, cur_time
            cur_time += service[node]
            load += demands[node]
            if load > capacity + 1e-9:
                return False, total_dist, cur_time
        
        cur_node = node
    
    # Return to depot
    d = dist_matrix[cur_node, 0]
    cur_time += d / speed
    cur_bat -= d
    total_dist += d
    
    if cur_bat < -1e-9:
        return False, total_dist, cur_time
    if cur_time > horizon + 1e-9:
        return False, total_dist, cur_time
        
    return True, total_dist, cur_time

def _solve_instance(instance, time_limit=120.0):
    start_time = time.time()
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations_list = instance["stations"]
    stations_set = set(stations_list)
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distances
    dist_matrix = _build_distance_matrix(coords)
    
    customers = list(range(1, n_customers + 1))
    
    # --- Phase 1: Clustering ---
    # Group customers by time windows to ensure feasibility
    customers_sorted = sorted(customers, key=lambda c: ready[c])
    
    clusters = []
    current_cluster = []
    current_cluster_max_due = -1
    current_cluster_min_ready = float('inf')
    current_cluster_load = 0
    
    for c in customers_sorted:
        if not current_cluster:
            current_cluster.append(c)
            current_cluster_max_due = due[c]
            current_cluster_min_ready = ready[c]
            current_cluster_load = demands[c]
        else:
            if current_cluster_load + demands[c] <= capacity:
                if len(current_cluster) < 5:
                    current_cluster.append(c)
                    current_cluster_max_due = max(current_cluster_max_due, due[c])
                    current_cluster_min_ready = min(current_cluster_min_ready, ready[c])
                    current_cluster_load += demands[c]
                else:
                    clusters.append(current_cluster)
                    current_cluster = [c]
                    current_cluster_max_due = due[c]
                    current_cluster_min_ready = ready[c]
                    current_cluster_load = demands[c]
            else:
                clusters.append(current_cluster)
                current_cluster = [c]
                current_cluster_max_due = due[c]
                current_cluster_min_ready = ready[c]
                current_cluster_load = demands[c]
                
    if current_cluster:
        clusters.append(current_cluster)
        
    # --- Phase 2: Construct Routes per Cluster ---
    routes = []
    unserved = set(customers)
    
    clusters.sort(key=lambda cl: min(ready[c] for c in cl))
    
    for cluster in clusters:
        current_route = []
        cur_node = 0
        cur_time = 0.0
        cur_bat = battery
        cur_load = 0.0
        
        cluster_sorted = sorted(cluster, key=lambda c: due[c])
        
        for c in cluster_sorted:
            if c not in unserved:
                continue
                
            if cur_load + demands[c] > capacity + 1e-9:
                continue
                
            d = dist_matrix[cur_node, c]
            test_bat = cur_bat - d
            test_time = cur_time + d / speed
            
            if test_bat < -1e-9:
                continue
                
            if test_time < ready[c]:
                test_time = ready[c]
            if test_time > due[c] + 1e-9:
                continue
            test_time += service[c]
            
            cur_time = test_time
            cur_bat = test_bat
            cur_load += demands[c]
            cur_node = c
            current_route.append(c)
            unserved.discard(c)
            
        if current_route:
            d_ret = dist_matrix[cur_node, 0]
            test_bat_ret = cur_bat - d_ret
            test_time_ret = cur_time + d_ret / speed
            
            if test_bat_ret >= -1e-9 and test_time_ret <= horizon + 1e-9:
                routes.append(current_route)
            else:
                best_station = None
                best_station_dist = float('inf')
                for st in stations_list:
                    d_st = dist_matrix[cur_node, st]
                    if cur_bat - d_st >= -1e-9:
                        bat_after_st = battery
                        t_arrive_st = cur_time + d_st / speed
                        t_leave_st = t_arrive_st + (battery - (cur_bat - d_st)) * recharge_rate
                        t_return = t_leave_st + dist_matrix[st, 0] / speed
                        if t_return <= horizon + 1e-9:
                            if d_st < best_station_dist:
                                best_station_dist = d_st
                                best_station = st
                
                if best_station is not None:
                    current_route.append(best_station)
                    routes.append(current_route)
                else:
                    for c in current_route:
                        routes.append([c])
    
    for c in list(unserved):
        routes.append([c])
        unserved.discard(c)
        
    # --- Phase 3: Local Search ---
    # Optimize each route individually with 2-Opt and Or-opt
    for i in range(len(routes)):
        if len(routes[i]) < 2:
            continue
            
        best_route = routes[i]
        best_dist, _, _ = _simulate_route(best_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
        
        improved = True
        while improved and (time.time() - start_time < time_limit * 0.7):
            improved = False
            n = len(best_route)
            
            # 2-Opt
            for k in range(1, n):
                for l in range(k + 1, n):
                    new_route = best_route[:k] + best_route[k:l+1][::-1] + best_route[l+1:]
                    valid, dist, _ = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                    if valid and dist < best_dist - 1e-6:
                        best_route = new_route
                        best_dist = dist
                        improved = True
                        break
                if improved:
                    break
            
            # Or-opt
            if not improved:
                for move_len in [1, 2, 3]:
                    for i_node in range(n - move_len + 1):
                        segment = best_route[i_node:i_node+move_len]
                        rest = best_route[:i_node] + best_route[i_node+move_len:]
                        
                        for j_node in range(len(rest) + 1):
                            new_route = rest[:j_node] + segment + rest[j_node:]
                            valid, dist, _ = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                            if valid and dist < best_dist - 1e-6:
                                best_route = new_route
                                best_dist = dist
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
        routes[i] = best_route
        
    # Inter-route relocate
    for i in range(len(routes)):
        if not routes[i]:
            continue
        for j in range(len(routes)):
            if i == j:
                continue
            if not routes[j]:
                continue
            
            for k in range(len(routes[i])):
                if routes[i][k] in stations_set:
                    continue
                
                c = routes[i][k]
                route_i = routes[i][:k] + routes[i][k+1:]
                best_pos = -1
                best_dist = float('inf')
                
                dist_i, _, _ = _simulate_route(route_i, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                if not dist_i:
                    continue
                dist_j, _, _ = _simulate_route(routes[j], dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                base_dist = dist_i + dist_j
                
                for pos in range(len(routes[j]) + 1):
                    new_route_j = routes[j][:pos] + [c] + routes[j][pos:]
                    valid, d_new_j, _ = _simulate_route(new_route_j, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                    if valid:
                        valid_i, d_i, _ = _simulate_route(route_i, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                        if valid_i:
                            total_d = d_i + d_new_j
                            if total_d < best_dist - 1e-6:
                                best_dist = total_d
                                best_pos = pos
                                
                if best_pos != -1:
                    routes[i] = route_i
                    routes[j] = routes[j][:best_pos] + [c] + routes[j][best_pos:]
                    
    routes = [r for r in routes if r]
    
    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0
    
    try:
        routes = _solve_instance(instance, time_limit=140.0)
        
        coords = instance["coords"]
        demands = instance["demands"]
        ready = instance["ready"]
        due = instance["due"]
        service = instance["service"]
        n_customers = instance["n_customers"]
        stations_set = set(instance["stations"])
        capacity = instance["capacity"]
        battery = instance["battery"]
        speed = instance["speed"]
        recharge_rate = instance["recharge_rate"]
        horizon = instance["horizon"]
        
        dist_matrix = _build_distance_matrix(coords)
        
        final_routes = []
        all_customers = set(range(1, n_customers + 1))
        served_customers = set()
        
        for r in routes:
            valid, _, _ = _simulate_route(r, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
            if valid:
                final_routes.append(r)
                for node in r:
                    if node not in stations_set:
                        served_customers.add(node)
            else:
                for node in r:
                    if node not in stations_set:
                        final_routes.append([node])
                        served_customers.add(node)
        
        missing = all_customers - served_customers
        for c in missing:
            final_routes.append([c])
            
        return final_routes
    except Exception:
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        return [[v] for v in customers]
# EVOLVE-BLOCK-END

"""EV routing with time windows: Optimized Clarke-Wright + efficient local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import heapq

def _build_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2)
            D[i, j] = d
            D[j, i] = d
    return D

def _route_stats_fast(route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set):
    """
    Fast route simulation. Returns (feasible, total_dist, total_time).
    """
    if not route:
        return True, 0.0, 0.0

    total_demand = 0
    for v in route:
        total_demand += demands[v]
    if total_demand > capacity:
        return False, 0.0, 0.0

    t = 0.0
    battery = battery_cap
    total_dist = 0.0
    prev = 0

    for v in route:
        d = D[prev, v]
        t += d / speed
        battery -= d
        
        if battery < -1e-9:
            return False, 0.0, 0.0

        if v in stations_set:
            energy_added = battery_cap - battery
            t += energy_added * recharge_rate
            battery = battery_cap
        else:
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0
            t += service[v]

        total_dist += d
        prev = v

    d = D[prev, 0]
    t += d / speed
    battery -= d
    if battery < -1e-9:
        return False, 0.0, 0.0
    total_dist += d

    if t > horizon + 1e-9:
        return False, 0.0, 0.0

    return True, total_dist, t

def _best_station_insertion(route, D, battery_cap, stations_set, stations_list, demands, ready, due, service, capacity, speed, recharge_rate, horizon):
    """
    Find best position to insert a station. Returns (new_route, success, dist).
    """
    best_new_route = None
    best_dist = float('inf')
    
    for pos in range(len(route) + 1):
        for s in stations_list:
            new_route = route[:pos] + [s] + route[pos:]
            feasible, dist, _ = _route_stats_fast(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
            if feasible and dist < best_dist:
                best_dist = dist
                best_new_route = new_route
                
    if best_new_route is not None:
        return best_new_route, True, best_dist
    return None, False, float('inf')

def _remove_useless_stations(route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set):
    """Remove stations that aren't needed for feasibility."""
    changed = True
    while changed:
        changed = False
        for idx in range(len(route) - 1, -1, -1):
            if route[idx] in stations_set:
                new_route = route[:idx] + route[idx+1:]
                feasible, _, _ = _route_stats_fast(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                if feasible:
                    route = new_route
                    changed = True
                    break
    return route

def _optimize_route(route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set, stations_list):
    """Full route optimization: remove useless stations, add if needed."""
    route = _remove_useless_stations(route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
    
    feasible, dist, _ = _route_stats_fast(route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
    if not feasible:
        new_route, success, new_dist = _best_station_insertion(route, D, battery_cap, stations_set, stations_list, demands, ready, due, service, capacity, speed, recharge_rate, horizon)
        if success:
            route = new_route
            route = _remove_useless_stations(route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
            feasible, dist, _ = _route_stats_fast(route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
    
    return route, dist

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    stations_set = set(stations)
    customers = list(range(1, n_customers + 1))
    
    D = _build_distance_matrix(coords)

    # Pre-compute: for each customer, check if single route is feasible
    single_route_feasible = {}
    single_route_dist = {}
    for c in customers:
        r = [c]
        feasible, dist, _ = _route_stats_fast(r, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
        if feasible:
            single_route_feasible[c] = True
            single_route_dist[c] = dist
        else:
            new_r, success, new_dist = _best_station_insertion(r, D, battery_cap, stations_set, stations, demands, ready, due, service, capacity, speed, recharge_rate, horizon)
            if success:
                single_route_feasible[c] = True
                single_route_dist[c] = new_dist
                # Store the route with station
                single_route_feasible[c] = new_r
            else:
                single_route_feasible[c] = False

    # Phase 1: Savings-Based Construction
    # Start with individual routes
    routes = []
    route_dists = []
    for c in customers:
        if isinstance(single_route_feasible[c], list):
            routes.append(single_route_feasible[c])
            route_dists.append(single_route_dist[c])
        elif single_route_feasible[c]:
            routes.append([c])
            route_dists.append(single_route_dist[c])
        else:
            # Should not happen, but handle gracefully
            routes.append([c])
            route_dists.append(float('inf'))

    # Pre-compute savings for all pairs
    # savings[i][j] = dist(0,i) + dist(0,j) - dist(i,j)
    # But we need to consider the full route distances
    n_routes = len(routes)
    
    # Greedy merge
    merged = True
    while merged and (time.time() < deadline - 30.0):
        merged = False
        best_saving = 0
        best_i = -1
        best_j = -1
        best_new_route = None
        best_new_dist = 0
        
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for j in range(i + 1, len(routes)):
                if not routes[j]:
                    continue
                
                ri = routes[i]
                rj = routes[j]
                
                # Check demand
                total_demand = sum(demands[v] for v in ri) + sum(demands[v] for v in rj)
                if total_demand > capacity:
                    continue
                
                # Try 4 combinations
                candidates = [
                    ri + rj,
                    ri + rj[::-1],
                    ri[::-1] + rj,
                    ri[::-1] + rj[::-1]
                ]
                
                for new_route in candidates:
                    feasible, dist, _ = _route_stats_fast(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                    if not feasible:
                        # Try adding a station
                        new_route2, success, dist2 = _best_station_insertion(new_route, D, battery_cap, stations_set, stations, demands, ready, due, service, capacity, speed, recharge_rate, horizon)
                        if success:
                            new_route = new_route2
                            dist = dist2
                        else:
                            continue
                    
                    saving = route_dists[i] + route_dists[j] - dist
                    if saving > best_saving:
                        best_saving = saving
                        best_i = i
                        best_j = j
                        best_new_route = new_route
                        best_new_dist = dist

        if best_saving > 1e-9 and best_i >= 0:
            new_route = _optimize_route(best_new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set, stations)
            _, new_dist = _route_stats_fast(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
            
            routes[best_i] = new_route
            routes[best_j] = []
            route_dists[best_i] = new_dist
            route_dists[best_j] = 0.0
            merged = True

    # Clean up
    active_routes = []
    active_dists = []
    for i in range(len(routes)):
        if routes[i]:
            active_routes.append(routes[i])
            active_dists.append(route_dists[i])
    routes = active_routes
    route_dists = active_dists

    # Phase 2: Local Search
    improved = True
    while improved and (time.time() < deadline - 10.0):
        improved = False
        
        # Within-route: 2-opt
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            n = len(route)
            if n < 2:
                continue
            
            for i in range(n - 1):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    
                    feasible, new_dist, _ = _route_stats_fast(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                    if not feasible:
                        new_route2, success, new_dist2 = _best_station_insertion(new_route, D, battery_cap, stations_set, stations, demands, ready, due, service, capacity, speed, recharge_rate, horizon)
                        if success:
                            new_route = new_route2
                            new_dist = new_dist2
                        else:
                            continue
                    
                    if new_dist < route_dists[r_idx] - 1e-9:
                        new_route = _optimize_route(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set, stations)
                        feasible, new_dist, _ = _route_stats_fast(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                        if feasible and new_dist < route_dists[r_idx] - 1e-9:
                            routes[r_idx] = new_route
                            route_dists[r_idx] = new_dist
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue

            # Or-opt
            for seg_len in [1, 2, 3]:
                if n < seg_len + 1:
                    continue
                for i in range(n - seg_len + 1):
                    if time.time() > deadline - 10.0:
                        break
                    segment = route[i:i + seg_len]
                    remaining = route[:i] + route[i + seg_len:]
                    if not remaining:
                        continue
                    for j in range(len(remaining) + 1):
                        if j == i or (j > i and j <= i + seg_len):
                            continue
                        new_route = remaining[:j] + segment + remaining[j:]
                        
                        feasible, new_dist, _ = _route_stats_fast(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                        if not feasible:
                            new_route2, success, new_dist2 = _best_station_insertion(new_route, D, battery_cap, stations_set, stations, demands, ready, due, service, capacity, speed, recharge_rate, horizon)
                            if success:
                                new_route = new_route2
                                new_dist = new_dist2
                            else:
                                continue
                        
                        if new_dist < route_dists[r_idx] - 1e-9:
                            new_route = _optimize_route(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set, stations)
                            feasible, new_dist, _ = _route_stats_fast(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                            if feasible and new_dist < route_dists[r_idx] - 1e-9:
                                routes[r_idx] = new_route
                                route_dists[r_idx] = new_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue

        # Inter-route moves
        if not improved and time.time() < deadline - 15.0:
            for src_idx in range(len(routes)):
                if not routes[src_idx]:
                    continue
                route = routes[src_idx]
                for i in range(len(route)):
                    if time.time() > deadline - 15.0:
                        break
                    cust = route[i]
                    if cust in stations_set:
                        continue
                        
                    new_src = route[:i] + route[i + 1:]
                    
                    for dst_idx in range(len(routes)):
                        if dst_idx == src_idx:
                            continue
                        dst_route = routes[dst_idx]
                        
                        dst_demand = sum(demands[v] for v in dst_route)
                        if dst_demand + demands[cust] > capacity:
                            continue
                            
                        for j in range(len(dst_route) + 1):
                            new_dst = dst_route[:j] + [cust] + dst_route[j:]
                            
                            feasible_dst, dist_dst, _ = _route_stats_fast(new_dst, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                            if not feasible_dst:
                                new_dst2, success, dist_dst2 = _best_station_insertion(new_dst, D, battery_cap, stations_set, stations, demands, ready, due, service, capacity, speed, recharge_rate, horizon)
                                if success:
                                    new_dst = new_dst2
                                    dist_dst = dist_dst2
                                else:
                                    continue
                            
                            if new_src:
                                feasible_src, dist_src, _ = _route_stats_fast(new_src, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                                if not feasible_src:
                                    new_src2, success, dist_src2 = _best_station_insertion(new_src, D, battery_cap, stations_set, stations, demands, ready, due, service, capacity, speed, recharge_rate, horizon)
                                    if success:
                                        new_src = new_src2
                                        dist_src = dist_src2
                                    else:
                                        continue
                            else:
                                dist_src = 0.0
                            
                            old_total = route_dists[src_idx] + route_dists[dst_idx]
                            new_total = dist_src + dist_dst
                            
                            if new_total < old_total - 1e-9:
                                new_src_opt, ds = _optimize_route(new_src, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set, stations) if new_src else ([], 0.0)
                                new_dst_opt, dd = _optimize_route(new_dst, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set, stations)
                                
                                fs, ds, _ = _route_stats_fast(new_src_opt, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set) if new_src_opt else (True, 0.0, 0.0)
                                fd, dd, _ = _route_stats_fast(new_dst_opt, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                                
                                if fs and fd and (ds + dd < old_total - 1e-9):
                                    routes[src_idx] = new_src_opt
                                    routes[dst_idx] = new_dst_opt
                                    route_dists[src_idx] = ds
                                    route_dists[dst_idx] = dd
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    # Final cleanup
    routes = [r for r in routes if r]

    # Verify all customers are served
    all_served = []
    for route in routes:
        all_served.extend(route)
    if sorted(all_served) != sorted(customers):
        # Fallback
        valid_routes = []
        for c in customers:
            r = [c]
            feasible, dist, _ = _route_stats_fast(r, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
            if not feasible:
                new_r, success, _ = _best_station_insertion(r, D, battery_cap, stations_set, stations, demands, ready, due, service, capacity, speed, recharge_rate, horizon)
                if success:
                    valid_routes.append(new_r)
                else:
                    valid_routes.append(r)
            else:
                valid_routes.append(r)
        return valid_routes

    return routes

# EVOLVE-BLOCK-END

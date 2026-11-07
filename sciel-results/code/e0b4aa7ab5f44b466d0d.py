"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def _build_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2)
            D[i, j] = d
            D[j, i] = d
    return D

def _get_route_stats(route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set):
    """
    Simulates the route. Returns (feasible, total_dist, total_time, max_battery_deficit_if_any).
    If a customer is visited, we check time windows.
    If a station is visited, we recharge.
    """
    if not route:
        return True, 0.0, 0.0, 0.0

    total_demand = sum(demands[v] for v in route)
    if total_demand > capacity:
        return False, 0.0, 0.0, 0.0

    t = 0.0
    battery = battery_cap
    total_dist = 0.0
    prev = 0

    for v in route:
        d = D[prev, v]
        t += d / speed
        battery -= d
        
        if battery < -1e-9:
            return False, 0.0, 0.0, 0.0

        if v in stations_set:
            # Recharge
            energy_added = battery_cap - battery
            t += energy_added * recharge_rate
            battery = battery_cap
        else:
            # Customer logic
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0, 0.0
            t += service[v]

        total_dist += d
        prev = v

    # Return to depot
    d = D[prev, 0]
    t += d / speed
    battery -= d
    if battery < -1e-9:
        return False, 0.0, 0.0, 0.0
    total_dist += d

    if t > horizon + 1e-9:
        return False, 0.0, 0.0, 0.0

    return True, total_dist, t, battery

def _try_insert_station(route, D, battery_cap, stations_set, stations_list, battery_cap_limit, demands, ready, due, service, capacity, speed, recharge_rate, horizon):
    """
    Tries to find the best position to insert a station to make the route battery-feasible.
    Returns (new_route, success).
    """
    best_new_route = None
    best_dist = float('inf')
    
    for pos in range(len(route) + 1):
        for s in stations_list:
            new_route = route[:pos] + [s] + route[pos:]
            feasible, dist, _, _ = _get_route_stats(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
            if feasible and dist < best_dist:
                best_dist = dist
                best_new_route = new_route
                
    if best_new_route:
        return best_new_route, True
    return None, False

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
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

    # Phase 1: Savings-Based Construction (Clarke-Wright)
    # 1. Start with individual routes for each customer
    routes = [[c] for c in customers]
    route_dists = []
    for r in routes:
        # Check if single customer route is feasible
        feasible, dist, _, _ = _get_route_stats(r, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
        if feasible:
            route_dists.append(dist)
        else:
            new_route, success = _try_insert_station(r, D, battery_cap, stations_set, stations, battery_cap, demands, ready, due, service, capacity, speed, recharge_rate, horizon)
            if success:
                routes[-1] = new_route
                _, dist, _, _ = _get_route_stats(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                route_dists[-1] = dist
            else:
                route_dists[-1] = float('inf')

    # 2. Calculate savings
    # We will perform merges greedily.
    
    merged = True
    while merged and (time.time() < deadline - 5.0):
        merged = False
        best_saving = 0
        best_i = -1
        best_j = -1
        best_direction = 0 # 0: i+j, 1: i+rev(j), 2: rev(i)+j, 3: rev(i)+rev(j)
        
        for i in range(len(routes)):
            if not routes[i]: continue
            for j in range(i + 1, len(routes)):
                if not routes[j]: continue
                
                ri = routes[i]
                rj = routes[j]
                
                # 4 possible combinations
                candidates = [
                    (ri + rj, 0),
                    (ri + rj[::-1], 1),
                    (ri[::-1] + rj, 2),
                    (ri[::-1] + rj[::-1], 3)
                ]
                
                for new_route, dir_idx in candidates:
                    # Quick check: demand
                    if sum(demands[v] for v in new_route) > capacity:
                        continue
                    
                    feasible, dist, _, _ = _get_route_stats(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                    if not feasible:
                        continue
                    
                    # Calculate saving
                    saving = route_dists[i] + route_dists[j] - dist
                    
                    if saving > best_saving:
                        best_saving = saving
                        best_i = i
                        best_j = j
                        best_direction = dir_idx
                        best_new_route = new_route

        if best_saving > 1e-9:
            ri = routes[best_i]
            rj = routes[best_j]
            
            # Reconstruct the specific merged route based on direction
            if best_direction == 0:
                new_route = ri + rj
            elif best_direction == 1:
                new_route = ri + rj[::-1]
            elif best_direction == 2:
                new_route = ri[::-1] + rj
            else:
                new_route = ri[::-1] + rj[::-1]
            
            routes[best_i] = new_route
            routes[best_j] = []
            
            # Update distances
            _, new_dist, _, _ = _get_route_stats(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
            route_dists[best_i] = new_dist
            route_dists[best_j] = 0.0
            
            merged = True

    # Clean up empty routes
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
    while improved and (time.time() < deadline - 3.0):
        improved = False
        
        # 1. Within-route improvements (2-opt and Or-opt)
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            n = len(route)
            if n < 2:
                continue
            
            # 2-opt
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    
                    feasible, new_dist, _, _ = _get_route_stats(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                    
                    if not feasible:
                        continue
                    
                    if new_dist < route_dists[r_idx] - 1e-9:
                        routes[r_idx] = new_route
                        route_dists[r_idx] = new_dist
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue

            # Or-opt (move 1, 2, 3 nodes)
            for seg_len in [1, 2, 3]:
                if n < seg_len + 1:
                    continue
                for i in range(n - seg_len + 1):
                    segment = route[i:i + seg_len]
                    remaining = route[:i] + route[i + seg_len:]
                    if not remaining:
                        continue
                    for j in range(len(remaining) + 1):
                        # Avoid inserting back in the same place
                        if j == i or (j > i and j <= i + seg_len):
                            continue
                        new_route = remaining[:j] + segment + remaining[j:]
                        
                        feasible, new_dist, _, _ = _get_route_stats(new_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
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
            
            # Station optimization: Try removing stations
            for idx, v in enumerate(route):
                if v in stations_set:
                    # Try removing this station at index idx
                    test_route = route[:idx] + route[idx+1:]
                    feasible, new_dist, _, _ = _get_route_stats(test_route, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                    if feasible and new_dist < route_dists[r_idx] - 1e-9:
                        routes[r_idx] = test_route
                        route_dists[r_idx] = new_dist
                        improved = True
                        break
            
            if improved:
                continue

        # 2. Inter-route moves
        if not improved:
            for src_idx in range(len(routes)):
                if not routes[src_idx]:
                    continue
                route = routes[src_idx]
                for i in range(len(route)):
                    cust = route[i]
                    # Only move customers
                    if cust in stations_set:
                        continue
                        
                    new_src = route[:i] + route[i + 1:]
                    
                    for dst_idx in range(len(routes)):
                        if dst_idx == src_idx:
                            continue
                        dst_route = routes[dst_idx]
                        
                        # Check demand capacity of dst
                        dst_demand = sum(demands[v] for v in dst_route)
                        if dst_demand + demands[cust] > capacity:
                            continue
                            
                        for j in range(len(dst_route) + 1):
                            new_dst = dst_route[:j] + [cust] + dst_route[j:]
                            
                            # Check feasibility of both
                            feasible_dst, dist_dst, _, _ = _get_route_stats(new_dst, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                            if not feasible_dst:
                                continue
                            
                            feasible_src, dist_src, _, _ = _get_route_stats(new_src, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set) if new_src else (True, 0.0, 0.0, 0.0)
                            if not feasible_src:
                                continue
                            
                            old_total = route_dists[src_idx] + route_dists[dst_idx]
                            new_total = dist_src + dist_dst
                            
                            if new_total < old_total - 1e-9:
                                routes[src_idx] = new_src
                                routes[dst_idx] = new_dst
                                route_dists[src_idx] = dist_src
                                route_dists[dst_idx] = dist_dst
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        # 3. Inter-route 2-opt (Cross-exchange)
        if not improved and len(routes) > 1:
            for i in range(len(routes)):
                if not routes[i]: continue
                for j in range(i + 1, len(routes)):
                    if not routes[j]: continue
                    ri = routes[i]
                    rj = routes[j]
                    
                    # Try swapping suffix of ri with suffix of rj
                    for p in range(1, len(ri)):
                        for q in range(1, len(rj)):
                            A = ri[:p]
                            B = ri[p:]
                            C = rj[:q]
                            D_part = rj[q:]
                            
                            new_ri = A + D_part
                            new_rj = C + B
                            
                            # Check demands
                            if sum(demands[v] for v in new_ri) > capacity: continue
                            if sum(demands[v] for v in new_rj) > capacity: continue
                            
                            feasible_i, dist_i, _, _ = _get_route_stats(new_ri, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                            if not feasible_i: continue
                            
                            feasible_j, dist_j, _, _ = _get_route_stats(new_rj, D, demands, ready, due, service, capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                            if not feasible_j: continue
                            
                            old_total = route_dists[i] + route_dists[j]
                            new_total = dist_i + dist_j
                            
                            if new_total < old_total - 1e-9:
                                routes[i] = new_ri
                                routes[j] = new_rj
                                route_dists[i] = dist_i
                                route_dists[j] = dist_j
                                improved = True
                                break
                        if improved: break
                    if improved: break
                if improved: break

    # Clean up empty routes
    routes = [r for r in routes if r]

    # Final check
    all_served = []
    for route in routes:
        all_served.extend(route)
    if sorted(all_served) != sorted(customers):
        return [[v] for v in customers]

    return routes

# EVOLVE-BLOCK-END

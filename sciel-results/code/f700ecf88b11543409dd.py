# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math
from itertools import combinations

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _build_dist_matrix(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dij = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            d[i, j] = dij
            d[j, i] = dij
    return d

def _simulate_route(route, instance, dist_mat, max_time=None):
    """
    Simulate a route (list of customer node ids).
    Returns (total_distance, valid, final_time, final_battery) or (inf, False, ...)
    Depot is node 0.
    """
    if max_time is None:
        max_time = instance["horizon"]
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    stations = instance["stations"]
    
    total_dist = 0.0
    total_demand = 0
    battery = battery_cap
    time = 0.0
    curr = 0
    
    # Check capacity
    for c in route:
        total_demand += demands[c]
    if total_demand > capacity:
        return (float('inf'), False, 0, 0)
    
    for c in route:
        # Travel from curr to c
        d = dist_mat[curr, c]
        travel_time = d / speed
        battery -= d
        time += travel_time
        total_dist += d
        
        if battery < -1e-9:
            return (float('inf'), False, 0, 0)
        
        # Arrive at customer
        if time < ready[c]:
            time = ready[c]
        
        if time > due[c] + 1e-9:
            return (float('inf'), False, 0, 0)
        
        time += service[c]
        curr = c
    
    # Return to depot
    d = dist_mat[curr, 0]
    travel_time = d / speed
    battery -= d
    time += travel_time
    total_dist += d
    
    if battery < -1e-9:
        return (float('inf'), False, 0, 0)
    
    if time > max_time + 1e-9:
        return (float('inf'), False, 0, 0)
    
    return (total_dist, True, time, battery)

def _simulate_route_with_stations(route, instance, dist_mat, max_time=None):
    """
    Simulate a route allowing for station visits.
    route is a list of node ids (customers and/or stations).
    Customers must be served in the order given (relative order preserved).
    This is a helper for local search where we might insert stations.
    For now, we assume the route only contains customers, and we check if it's feasible.
    If not feasible, we might need to insert stations, but that's complex.
    Let's stick to checking feasibility of customer-only routes first.
    """
    return _simulate_route(route, instance, dist_mat, max_time)

def _repair_route(route, instance, dist_mat):
    """
    Try to make a route feasible by inserting stations.
    This is a simplified repair: if battery runs out, insert the nearest station before the failing leg.
    Returns (new_route, is_feasible, total_dist)
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    stations = instance["stations"]
    
    # Check capacity
    total_demand = sum(demands[c] for c in route)
    if total_demand > capacity:
        return (route, False, float('inf'))
    
    # Simulate and track battery
    # We'll build a new route with stations inserted if needed
    new_route = []
    battery = battery_cap
    time = 0.0
    curr = 0
    
    # We need to decide where to insert stations.
    # Strategy: Simulate. If battery < 0 before reaching a customer, insert a station.
    # Which station? The one that minimizes detour + recharge time.
    # Since recharge is fast (0.5) and distance is key, pick nearest station to current or next.
    
    # Actually, the problem says "stations may repeat".
    # Let's do a forward simulation.
    # If we are at curr and next is c, and battery < dist(curr, c), we need a station.
    # Pick station s that minimizes dist(curr, s) + dist(s, c) + recharge_time.
    # recharge_time = (battery - dist(curr, s)) * recharge_rate.
    # Note: battery must be >= dist(curr, s).
    
    for c in route:
        # Check if we can reach c
        d_direct = dist_mat[curr, c]
        
        if battery >= d_direct:
            # Go directly
            travel_time = d_direct / speed
            battery -= d_direct
            time += travel_time
            new_route.append(c)
            
            # Service
            if time < ready[c]:
                time = ready[c]
            if time > due[c] + 1e-9:
                return (new_route, False, float('inf'))
            time += service[c]
            curr = c
        else:
            # Need a station
            best_s = None
            best_cost = float('inf')
            
            for s in stations:
                d1 = dist_mat[curr, s]
                if battery >= d1:
                    # Can reach station
                    time_to_s = d1 / speed
                    battery_at_s = battery - d1
                    recharge_time = battery_at_s * recharge_rate
                    time_at_s = time + time_to_s + recharge_time
                    battery_after = battery_cap # Refilled
                    
                    # Check if we can reach c from s
                    d2 = dist_mat[s, c]
                    if battery_after >= d2:
                        time_to_c = d2 / speed
                        time_at_c = time_at_s + time_to_c
                        # Check time window
                        arr_time = max(time_at_c, ready[c])
                        if arr_time <= due[c] + 1e-9:
                            # Cost is total distance
                            cost = d1 + d2
                            if cost < best_cost:
                                best_cost = cost
                                best_s = s
                                best_time_at_c = arr_time + service[c]
                                best_battery_after_c = battery_after - d2
            
            if best_s is None:
                # Try to go to a station even if we can't reach c directly after?
                # Or maybe we need two stations?
                # For simplicity, if no single station works, mark infeasible.
                return (new_route, False, float('inf'))
            
            # Insert station
            d1 = dist_mat[curr, best_s]
            travel_time = d1 / speed
            battery -= d1
            time += travel_time
            new_route.append(best_s)
            
            recharge_time = battery * recharge_rate
            time += recharge_time
            battery = battery_cap
            
            # Now go to c
            d2 = dist_mat[best_s, c]
            travel_time = d2 / speed
            battery -= d2
            time += travel_time
            new_route.append(c)
            
            if time < ready[c]:
                time = ready[c]
            if time > due[c] + 1e-9:
                return (new_route, False, float('inf'))
            time += service[c]
            curr = c
    
    # Return to depot
    d = dist_mat[curr, 0]
    if battery < d:
        # Need station to return
        # Find station s reachable from curr such that we can reach depot from s
        best_s = None
        best_cost = float('inf')
        for s in stations:
            d1 = dist_mat[curr, s]
            if battery >= d1:
                battery_at_s = battery - d1
                recharge_time = battery_at_s * recharge_rate
                time_at_s = time + d1/speed + recharge_time
                battery_after = battery_cap
                d2 = dist_mat[s, 0]
                if battery_after >= d2:
                    time_at_depot = time_at_s + d2/speed
                    if time_at_depot <= instance["horizon"] + 1e-9:
                        cost = d1 + d2
                        if cost < best_cost:
                            best_cost = cost
                            best_s = s
        
        if best_s is None:
            return (new_route, False, float('inf'))
        
        d1 = dist_mat[curr, best_s]
        time += d1/speed
        battery -= d1
        new_route.append(best_s)
        time += battery * recharge_rate
        battery = battery_cap
        
        d2 = dist_mat[best_s, 0]
        time += d2/speed
        battery -= d2
        total_dist = sum(dist_mat[new_route[i], new_route[i+1]] for i in range(len(new_route)-1)) + dist_mat[0, new_route[0]] + dist_mat[new_route[-1], 0]
        # Recalculate total dist properly
        total_dist = dist_mat[0, new_route[0]]
        for i in range(len(new_route)-1):
            total_dist += dist_mat[new_route[i], new_route[i+1]]
        total_dist += dist_mat[new_route[-1], 0]
        
        if time > instance["horizon"] + 1e-9:
            return (new_route, False, float('inf'))
        
        return (new_route, True, total_dist)
    else:
        time += d/speed
        battery -= d
        total_dist = dist_mat[0, new_route[0]]
        for i in range(len(new_route)-1):
            total_dist += dist_mat[new_route[i], new_route[i+1]]
        total_dist += dist_mat[new_route[-1], 0]
        
        if time > instance["horizon"] + 1e-9:
            return (new_route, False, float('inf'))
        
        return (new_route, True, total_dist)

def _partition_customers(customers, instance, dist_mat, k_routes):
    """
    Partition customers into k_routes groups using k-means-like clustering or simple sorting.
    For simplicity, use k-means on coordinates.
    """
    if k_routes >= len(customers):
        return [[c] for c in customers]
    
    # Initialize centers randomly
    random.seed(42)
    centers = [instance["coords"][c] for c in random.sample(customers, k_routes)]
    
    for _ in range(10):
        # Assign
        clusters = [[] for _ in range(k_routes)]
        for c in customers:
            dists = [math.hypot(instance["coords"][c][0] - centers[k][0], instance["coords"][c][1] - centers[k][1]) for k in range(k_routes)]
            best_k = np.argmin(dists)
            clusters[best_k].append(c)
        
        # Update centers
        for k in range(k_routes):
            if clusters[k]:
                xs = np.mean([instance["coords"][c][0] for c in clusters[k]])
                ys = np.mean([instance["coords"][c][1] for c in clusters[k]])
                centers[k] = (xs, ys)
            else:
                # Reassign a random customer
                if len(customers) > 0:
                    c = random.choice(customers)
                    clusters[k].append(c)
                    centers[k] = instance["coords"][c]
    
    return clusters

def _optimize_route(route, instance, dist_mat):
    """
    Optimize the order of customers in a route using 2-opt.
    Stations are not handled here, only customer order.
    Assumes the route is feasible or will be repaired later.
    """
    if len(route) <= 1:
        return route
    
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                # Try reversing segment i..j
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                
                # Check capacity (same)
                # Check feasibility
                _, valid, _, _ = _simulate_route(new_route, instance, dist_mat)
                if valid:
                    d_old, _, _, _ = _simulate_route(route, instance, dist_mat)
                    d_new, _, _, _ = _simulate_route(new_route, instance, dist_mat)
                    if d_new < d_old - 1e-9:
                        route = new_route
                        improved = True
    
    return route

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = instance["stations"]
    dist_mat = _build_dist_matrix(coords)
    
    # Initial solution: 1 route per customer
    routes = [[c] for c in customers]
    
    # Try to merge routes to reduce number of routes and distance
    # We want to minimize total distance. Merging saves the return-to-depot and go-out-from-depot legs.
    # So we should merge as much as possible.
    
    # Strategy:
    # 1. Cluster customers into groups.
    # 2. Optimize order in each group.
    # 3. Repair if infeasible (insert stations).
    # 4. Local search: Move customers between routes, swap, 2-opt.
    
    # Let's start with a reasonable number of routes, e.g., 5-10.
    # Capacity is 100, demand 5-25. Avg 15. So 100/15 ~ 6-7 customers per route.
    # 25 customers / 7 ~ 4 routes.
    
    best_routes = routes
    best_dist = sum(_simulate_route(r, instance, dist_mat)[0] for r in routes)
    
    # Try different numbers of clusters
    for k in range(3, 11):
        if time.time() > deadline:
            break
        
        clusters = _partition_customers(customers, instance, dist_mat, k)
        
        # Optimize order in each cluster
        new_routes = []
        for cluster in clusters:
            if not cluster:
                continue
            # Sort by due time as initial heuristic
            cluster.sort(key=lambda c: instance["due"][c])
            # 2-opt
            opt_route = _optimize_route(cluster, instance, dist_mat)
            new_routes.append(opt_route)
        
        # Repair and validate
        total_dist = 0
        valid = True
        final_routes = []
        for r in new_routes:
            rep_route, is_valid, d = _repair_route(r, instance, dist_mat)
            if not is_valid:
                valid = False
                break
            total_dist += d
            final_routes.append(rep_route)
        
        if valid and total_dist < best_dist:
            best_dist = total_dist
            best_routes = final_routes
    
    # Local Search: Or-opt (move a sequence of 1-3 customers) and 2-opt between routes
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 1. Try moving a customer from one route to another
        for i in range(len(best_routes)):
            if len(best_routes[i]) <= 1:
                continue
            for j in range(len(best_routes[i])):
                c = best_routes[i][j]
                # Remove c
                new_route_i = best_routes[i][:j] + best_routes[i][j+1:]
                
                # Try inserting into other routes
                for k in range(len(best_routes)):
                    if k == i:
                        continue
                    route_k = best_routes[k]
                    for pos in range(len(route_k) + 1):
                        new_route_k = route_k[:pos] + [c] + route_k[pos:]
                        
                        # Check feasibility of both
                        _, valid_i, _, _ = _simulate_route(new_route_i, instance, dist_mat)
                        _, valid_k, _, _ = _simulate_route(new_route_k, instance, dist_mat)
                        
                        if valid_i and valid_k:
                            d_i, _, _, _ = _simulate_route(new_route_i, instance, dist_mat)
                            d_k, _, _, _ = _simulate_route(new_route_k, instance, dist_mat)
                            d_old_i, _, _, _ = _simulate_route(best_routes[i], instance, dist_mat)
                            d_old_k, _, _, _ = _simulate_route(best_routes[k], instance, dist_mat)
                            
                            if d_i + d_k < d_old_i + d_old_k - 1e-9:
                                best_routes[i] = new_route_i
                                best_routes[k] = new_route_k
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
        
        # 2. Try swapping two customers
        for i in range(len(best_routes)):
            for j in range(len(best_routes)):
                if i >= j:
                    continue
                if len(best_routes[i]) == 0 or len(best_routes[j]) == 0:
                    continue
                for a in range(len(best_routes[i])):
                    for b in range(len(best_routes[j])):
                        c1 = best_routes[i][a]
                        c2 = best_routes[j][b]
                        
                        new_route_i = best_routes[i][:]
                        new_route_j = best_routes[j][:]
                        new_route_i[a] = c2
                        new_route_j[b] = c1
                        
                        _, valid_i, _, _ = _simulate_route(new_route_i, instance, dist_mat)
                        _, valid_j, _, _ = _simulate_route(new_route_j, instance, dist_mat)
                        
                        if valid_i and valid_j:
                            d_i, _, _, _ = _simulate_route(new_route_i, instance, dist_mat)
                            d_j, _, _, _ = _simulate_route(new_route_j, instance, dist_mat)
                            d_old_i, _, _, _ = _simulate_route(best_routes[i], instance, dist_mat)
                            d_old_j, _, _, _ = _simulate_route(best_routes[j], instance, dist_mat)
                            
                            if d_i + d_j < d_old_i + d_old_j - 1e-9:
                                best_routes[i] = new_route_i
                                best_routes[j] = new_route_j
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    # Final check: ensure all customers are covered
    served = set()
    for r in best_routes:
        for c in r:
            if c in customers:
                served.add(c)
    
    if served != set(customers):
        # Fallback to 1 customer per route
        best_routes = [[c] for c in customers]
    
    return best_routes
# EVOLVE-BLOCK-END

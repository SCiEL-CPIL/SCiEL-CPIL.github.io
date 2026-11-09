"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt, atan2, pi

def _build_distance_matrix(coords):
    n = len(coords)
    coords = np.array(coords, dtype=float)
    x_diff = coords[:, 0, np.newaxis] - coords[np.newaxis, :, 0]
    y_diff = coords[:, 1, np.newaxis] - coords[np.newaxis, :, 1]
    dist_matrix = np.sqrt(x_diff**2 + y_diff**2)
    return dist_matrix

def _simulate_route(route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
    """Simulate a route. Returns (is_valid, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0
    cur_node = 0
    cur_time = 0.0
    cur_bat = float(battery)
    total_dist = 0.0
    load = 0.0
    stations_arr = stations_set  # for fast lookup
    
    for node in route:
        d = dist_matrix[cur_node, node]
        cur_time += d / speed
        cur_bat -= d
        total_dist += d
        if cur_bat < -1e-9:
            return False, total_dist, cur_time
        if node in stations_arr:
            energy_needed = battery - cur_bat
            if energy_needed > 0:
                cur_time += energy_needed * recharge_rate
                cur_bat = float(battery)
        else:
            if cur_time < ready[node]:
                cur_time = ready[node]
            if cur_time > due[node] + 1e-9:
                return False, total_dist, cur_time
            cur_time += service[node]
            load += demands[node]
            if load > capacity + 1e-9:
                return False, total_dist, cur_time
        cur_node = node
    
    d = dist_matrix[cur_node, 0]
    cur_time += d / speed
    cur_bat -= d
    total_dist += d
    if cur_bat < -1e-9:
        return False, total_dist, cur_time
    if cur_time > horizon + 1e-9:
        return False, total_dist, cur_time
    return True, total_dist, cur_time

def _fast_feasible_insert(route, pos, node, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
    """Check if inserting node at position pos in route is feasible. Returns (valid, new_dist)."""
    new_route = route[:pos] + [node] + route[pos:]
    valid, dist, _ = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
    return valid, dist

def _sweep_construct(instance, dist_matrix, time_limit):
    """Construct initial routes using angle-based sweep with battery-aware insertion."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    customers = list(range(1, n_customers + 1))
    
    # Compute angles from depot
    depot_x, depot_y = coords[0]
    angles = {}
    for c in customers:
        dx = coords[c][0] - depot_x
        dy = coords[c][1] - depot_y
        angles[c] = atan2(dy, dx)
    
    # Sort by angle
    sorted_customers = sorted(customers, key=lambda c: angles[c])
    
    # Determine number of routes based on total demand
    total_demand = sum(demands[c] for c in customers)
    min_routes = max(1, int(np.ceil(total_demand / capacity)))
    
    # Try different numbers of routes
    best_routes = None
    best_total_dist = float('inf')
    
    for n_routes in range(min_routes, min_routes + 4):
        # Split sorted customers into n_routes groups by angle
        group_size = len(sorted_customers) // n_routes
        remainder = len(sorted_customers) % n_routes
        routes = []
        idx = 0
        for r in range(n_routes):
            size = group_size + (1 if r < remainder else 0)
            group = sorted_customers[idx:idx + size]
            idx += size
            routes.append(group)
        
        # Now build each route with battery-aware insertion
        final_routes = []
        for group in routes:
            route = []
            # Sort group by due date as tiebreaker
            group_sorted = sorted(group, key=lambda c: (angles[c], due[c]))
            
            for c in group_sorted:
                # Try to insert at best position
                best_pos = -1
                best_dist = float('inf')
                for pos in range(len(route) + 1):
                    valid, dist = _fast_feasible_insert(route, pos, c, dist_matrix, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                    if valid and dist < best_dist:
                        best_dist = dist
                        best_pos = pos
                
                if best_pos >= 0:
                    route.insert(best_pos, c)
                else:
                    # Can't insert, append at end if feasible
                    valid, dist = _fast_feasible_insert(route, len(route), c, dist_matrix, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                    if valid:
                        route.append(c)
                    else:
                        # Force as single route
                        final_routes.append(route) if route else None
                        route = [c]
            
            if route:
                final_routes.append(route)
        
        # Calculate total distance
        total_dist = 0.0
        all_valid = True
        for r in final_routes:
            valid, dist, _ = _simulate_route(r, dist_matrix, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
            if not valid:
                all_valid = False
                break
            total_dist += dist
        
        if all_valid and total_dist < best_total_dist:
            best_total_dist = total_dist
            best_routes = [list(r) for r in final_routes]
        
        if time.time() > time_limit:
            break
    
    if best_routes is None:
        # Fallback: single customer routes
        best_routes = [[c] for c in customers]
    
    return best_routes

def _improve_routes(routes, dist_matrix, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon, time_limit):
    """Improve routes using 2-opt and inter-route moves."""
    routes = [list(r) for r in routes]
    
    # Intra-route 2-opt
    for idx, route in enumerate(routes):
        if len(route) < 2:
            continue
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    valid, dist, _ = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                    if valid:
                        old_valid, old_dist, _ = _simulate_route(route, dist_matrix, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                        if dist < old_dist - 1e-6:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        routes[idx] = route
    
    # Inter-route moves: try moving a customer from one route to another
    for _iteration in range(3):
        if time.time() > time_limit:
            break
        moved = False
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                if time.time() > time_limit:
                    break
                cust = routes[i][j]
                # Try removing from route i
                route_i_without = routes[i][:j] + routes[i][j+1:]
                valid_i, dist_i, _ = _simulate_route(route_i_without, dist_matrix, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                if not valid_i:
                    continue
                
                # Try inserting into each other route
                for k in range(len(routes)):
                    if k == i:
                        continue
                    route_k = routes[k]
                    best_pos = -1
                    best_total = float('inf')
                    for pos in range(len(route_k) + 1):
                        new_route_k = route_k[:pos] + [cust] + route_k[pos:]
                        valid_k, dist_k, _ = _simulate_route(new_route_k, dist_matrix, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                        if valid_k:
                            total = dist_i + dist_k
                            if total < best_total:
                                best_total = total
                                best_pos = pos
                    
                    if best_pos >= 0:
                        # Calculate current total
                        _, cur_dist_i, _ = _simulate_route(routes[i], dist_matrix, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                        _, cur_dist_k, _ = _simulate_route(routes[k], dist_matrix, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                        if best_total < cur_dist_i + cur_dist_k - 1e-6:
                            routes[i] = route_i_without
                            routes[k] = route_k[:best_pos] + [cust] + route_k[best_pos:]
                            if not routes[i]:
                                routes.pop(i)
                                i -= 1
                                break
                            moved = True
                            break
                if moved:
                    break
            if moved:
                break
        if not moved:
            break
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    return routes

def _solve_instance(instance, time_limit=45.0):
    start_time = time.time()
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    dist_matrix = _build_distance_matrix(coords)
    
    # Phase 1: Sweep construction
    t1 = time.time()
    routes = _sweep_construct(instance, dist_matrix, t1 + time_limit * 0.3)
    
    # Verify all customers served
    served = set()
    for r in routes:
        for node in r:
            if node not in stations:
                served.add(node)
    all_customers = set(range(1, n_customers + 1))
    unserved = all_customers - served
    if unserved:
        for c in unserved:
            routes.append([c])
    
    # Phase 2: Local improvement
    t2 = time.time()
    remaining = time_limit - (t2 - start_time)
    if remaining > 5:
        routes = _improve_routes(routes, dist_matrix, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon, t2 + remaining)
    
    # Final verification and repair
    final_routes = []
    for r in routes:
        valid, dist, _ = _simulate_route(r, dist_matrix, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
        if valid:
            final_routes.append(r)
        else:
            for c in r:
                if c not in stations:
                    final_routes.append([c])
    
    return final_routes

def plan(instance):
    start_time = time.time()
    
    try:
        routes = _solve_instance(instance, time_limit=130.0)
        return routes
    except Exception:
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        return [[v] for v in customers]
# EVOLVE-BLOCK-END

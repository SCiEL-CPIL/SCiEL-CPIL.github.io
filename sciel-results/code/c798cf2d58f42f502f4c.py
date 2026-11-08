# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt
from scipy.spatial.distance import cdist

def _simulate_route(route, coords, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon):
    """
    Simulate a route (list of customer/station node ids, depot implicit at ends).
    Returns (is_valid, total_distance)
    """
    if not route:
        return True, 0.0
    
    cur_node = 0
    cur_time = 0.0
    cur_bat = battery
    total_dist = 0.0
    load = 0.0
    
    for node in route:
        d = sqrt((coords[cur_node, 0] - coords[node, 0])**2 + (coords[cur_node, 1] - coords[node, 1])**2)
        cur_time += d / speed
        cur_bat -= d
        total_dist += d
        
        if cur_bat < -1e-9:
            return False, total_dist
        
        if node in stations:
            # Recharge
            energy = battery - cur_bat
            cur_time += energy * recharge_rate
            cur_bat = battery
        else:
            # Customer
            load += demands[node]
            if load > capacity + 1e-9:
                return False, total_dist
            cur_time = max(cur_time, ready[node])
            if cur_time > due[node] + 1e-9:
                return False, total_dist
            cur_time += service[node]
        
        cur_node = node
    
    # Return to depot
    d = sqrt((coords[cur_node, 0] - coords[0, 0])**2 + (coords[cur_node, 1] - coords[0, 1])**2)
    cur_time += d / speed
    cur_bat -= d
    total_dist += d
    
    if cur_bat < -1e-9:
        return False, total_dist
    if cur_time > horizon + 1e-9:
        return False, total_dist
        
    return True, total_dist

def _solve_instance(instance, time_limit=10.0):
    coords = np.array(instance["coords"])
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"])
    ready = np.array(instance["ready"])
    due = np.array(instance["due"])
    service = np.array(instance["service"])
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    customers = list(range(1, n_customers + 1))
    
    # Precompute distance matrix
    n_nodes = len(coords)
    dist_matrix = cdist(coords, coords)
    
    # Sort customers by due date for a better initial ordering
    customers.sort(key=lambda x: due[x])
    
    routes = []
    route_loads = []
    
    start_time = time.time()
    
    for cust in customers:
        if time.time() - start_time > time_limit * 0.7:
            break
            
        best_cost = float('inf')
        best_route_idx = -1
        best_pos_idx = -1
        best_new_route = None
        best_new_load = 0.0
        
        # Try inserting into existing routes
        for r_idx, route in enumerate(routes):
            current_load = route_loads[r_idx]
            if current_load + demands[cust] > capacity + 1e-9:
                continue
                
            # Try all positions in the route
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                
                # Quick capacity check already done above
                
                is_valid, total_dist = _simulate_route(new_route, coords, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                
                if is_valid:
                    # We want to minimize total distance. 
                    if total_dist < best_cost:
                        best_cost = total_dist
                        best_route_idx = r_idx
                        best_pos_idx = pos
                        best_new_route = new_route
                        best_new_load = current_load + demands[cust]
                        
        # If no insertion into existing route is feasible, start a new route
        if best_route_idx == -1:
            new_route = [cust]
            is_valid, total_dist = _simulate_route(new_route, coords, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
            if is_valid:
                routes.append(new_route)
                route_loads.append(demands[cust])
            else:
                # Fallback: should not happen for feasible instances
                pass
        else:
            routes[best_route_idx] = best_new_route
            route_loads[best_route_idx] = best_new_load
            
    # Local search: Or-opt and 2-opt within routes to improve distance
    if time.time() - start_time < time_limit * 0.9:
        improved = True
        while improved and time.time() - start_time < time_limit * 0.9:
            improved = False
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                if len(route) < 2:
                    continue
                
                # 2-opt
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        is_valid, new_dist = _simulate_route(new_route, coords, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                        if is_valid:
                            _, old_dist = _simulate_route(route, coords, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                            if new_dist < old_dist - 1e-6:
                                routes[r_idx] = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
                    
    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    routes = _solve_instance(instance, time_limit=130.0)
    
    # Check if we are out of time? The heuristic is fast, but just in case.
    # If _solve_instance is too slow, we might need a simpler fallback.
    # For 25 customers, O(N^2) insertions with simulation O(N) is O(N^3) ~ 25^3 = 15625 ops per instance.
    # Very fast.
    
    return routes
# EVOLVE-BLOCK-END

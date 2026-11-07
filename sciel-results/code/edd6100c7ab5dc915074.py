"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt, inf

def _build_dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = sqrt(dx*dx + dy*dy)
            D[i, j] = d
            D[j, i] = d
    return D

def _get_best_station_path(start, end, t_start, b_start, instance, D):
    """
    Try to find a path from start to end, possibly via one charging station,
    given start time and battery.
    Returns (t_end, b_end, total_dist, expanded_nodes) or None if infeasible.
    """
    d_direct = D[start, end]
    speed = instance["speed"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    
    # Option 1: Direct
    if b_start - d_direct >= -1e-9:
        t_end = t_start + d_direct / speed
        b_end = b_start - d_direct
        return t_end, b_end, d_direct, []
        
    # Option 2: Via a station
    stations = instance["stations"]
    best_option = None
    best_metric = inf
    
    for s in stations:
        d_s1 = D[start, s]
        d_s2 = D[s, end]
        
        # Check if we can reach station
        if b_start - d_s1 < -1e-9:
            continue
            
        t_at_s = t_start + d_s1 / speed
        b_at_s = b_start - d_s1
        
        # Charge
        energy_needed = battery_max - b_at_s
        if energy_needed > 0:
            t_charge = energy_needed * recharge_rate
        else:
            t_charge = 0
            
        t_after_charge = t_at_s + t_charge
        b_after_charge = battery_max
        
        # Check if we can reach end from station
        if b_after_charge - d_s2 < -1e-9:
            continue
            
        t_end = t_after_charge + d_s2 / speed
        b_end = b_after_charge - d_s2
        total_dist = d_s1 + d_s2
        
        # Metric: minimize total distance primarily
        metric = total_dist
        
        if metric < best_metric:
            best_metric = metric
            best_option = (t_end, b_end, total_dist, [s])
            
    return best_option

def _simulate_route(route, instance, D):
    """
    Simulate a route of customers.
    Returns (feasible, total_dist, total_time, expanded_route)
    """
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    horizon = instance["horizon"]
    
    t = 0.0
    b = float(battery_max)
    prev = 0
    total_dist = 0.0
    expanded_route = []
    
    for node in route:
        result = _get_best_station_path(prev, node, t, b, instance, D)
        if result is None:
            return False, inf, inf, None
        
        t_next, b_next, d_seg, inserted_stations = result
        
        total_dist += d_seg
        t = t_next
        b = b_next
        
        if node <= n_customers:
            if t < instance["ready"][node]:
                t = instance["ready"][node]
            if t > instance["due"][node] + 1e-9:
                return False, inf, inf, None
            t += instance["service"][node]
            
        for s in inserted_stations:
            expanded_route.append(s)
        expanded_route.append(node)
        
        prev = node
        
    result = _get_best_station_path(prev, 0, t, b, instance, D)
    if result is None:
        return False, inf, inf, None
        
    t_end, b_end, d_seg, inserted_stations = result
    
    total_dist += d_seg
    t = t_end
    
    for s in inserted_stations:
        expanded_route.append(s)
        
    if t > horizon + 1e-9:
        return False, inf, inf, None
        
    return True, total_dist, t, expanded_route

def plan(instance):
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    
    D = _build_dist_matrix(coords)
    
    customers = list(range(1, n_customers + 1))
    customers_sorted = sorted(customers, key=lambda v: instance["due"][v])
    
    routes = []
    
    def _eval_route(cust_route):
        feasible, dist, time, expanded = _simulate_route(cust_route, instance, D)
        if not feasible:
            return inf, False
        return dist, True

    for cust in customers_sorted:
        best_option = None
        best_dist = inf
        
        for r_idx, route in enumerate(routes):
            n = len(route)
            for pos in range(n + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                dist, feasible = _eval_route(new_route)
                if feasible and dist < best_dist:
                    best_dist = dist
                    best_option = (r_idx, new_route)
        
        new_single = [cust]
        dist, feasible = _eval_route(new_single)
        if feasible and dist < best_dist:
            best_dist = dist
            best_option = (-1, new_single)
            
        if best_option is None:
            routes.append([cust])
        elif best_option[0] == -1:
            routes.append(best_option[1])
        else:
            routes[best_option[0]] = best_option[1]
            
    final_routes = []
    for route in routes:
        feasible, dist, time, expanded = _simulate_route(route, instance, D)
        if feasible:
            final_routes.append(expanded)
            
    served = set()
    for route in final_routes:
        for node in route:
            if node <= n_customers:
                served.add(node)
                
    for cust in customers:
        if cust not in served:
            feasible, dist, time, expanded = _simulate_route([cust], instance, D)
            if feasible:
                final_routes.append(expanded)
            else:
                final_routes.append([cust])
                
    return final_routes
# EVOLVE-BLOCK-END

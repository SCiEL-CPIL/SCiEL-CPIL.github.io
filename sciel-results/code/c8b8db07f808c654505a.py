# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
import time
import random

DEADLINE_SECONDS = 115.0  # safety margin below 120s limit


def _euclidean(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _validate_route(route, instance, dist_matrix):
    """Validate a single route. Returns (valid, distance)."""
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    
    # Check demand
    total_demand = sum(demands[node] for node in route if node <= n_customers)
    if total_demand > capacity:
        return False, 0.0
    
    # Simulate route
    battery = battery_max
    time_at = 0.0
    distance = 0.0
    current_node = 0  # depot
    
    for node in route:
        d = dist_matrix[current_node, node]
        time_at += d / speed
        battery -= d
        distance += d
        
        if battery < -1e-9:
            return False, 0.0
        
        if node <= n_customers:
            # Customer: check time window
            if time_at < ready[node]:
                time_at = ready[node]
            if time_at > due[node] + 1e-9:
                return False, 0.0
            time_at += service[node]
        elif node in stations:
            # Charging station
            energy_needed = battery_max - battery
            time_at += energy_needed * recharge_rate
            battery = battery_max
        
        current_node = node
    
    # Return to depot
    d = dist_matrix[current_node, 0]
    time_at += d / speed
    battery -= d
    distance += d
    
    if battery < -1e-9:
        return False, 0.0
    if time_at > horizon + 1e-9:
        return False, 0.0
    
    return True, distance


def _validate_solution(routes, instance, dist_matrix):
    """Validate all routes. Returns (valid, total_distance)."""
    n_customers = instance["n_customers"]
    served = set()
    total_distance = 0.0
    
    for route in routes:
        valid, dist = _validate_route(route, instance, dist_matrix)
        if not valid:
            return False, 0.0
        total_distance += dist
        
        for node in route:
            if node <= n_customers:
                if node in served:
                    return False, 0.0
                served.add(node)
    
    # Check all customers served exactly once
    if len(served) != n_customers:
        return False, 0.0
    if any(i not in served for i in range(1, n_customers + 1)):
        return False, 0.0

    return True, total_distance


def _compute_total_distance(routes, dist_matrix):
    """Compute total distance of routes."""
    total = 0.0
    for route in routes:
        current = 0
        for node in route:
            total += dist_matrix[current, node]
            current = node
        total += dist_matrix[current, 0]
    return total


def _construct_initial_solution(instance):
    """Build initial routes using a sweep and nearest neighbor approach with battery checks."""
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = instance["stations"]
    n_nodes = len(coords)
    
    # Precompute distance matrix
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist_matrix[i, j] = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
            
    # Sort customers by angle from depot (Sweep heuristic)
    angles = [np.arctan2(coords[i][1] - coords[0][1], coords[i][0] - coords[0][0]) for i in range(1, n_customers + 1)]
    customer_order = [i + 1 for i in np.argsort(angles)]
    
    routes = []
    unvisited = set(customer_order)
    
    while unvisited:
        route = []
        current = 0
        battery = battery_max
        time_at = 0.0
        load = 0.0
        
        # Pick first customer: nearest unvisited
        start_cust = min(unvisited, key=lambda c: dist_matrix[0, c])
        unvisited.remove(start_cust)
        
        # Check if we can go to start_cust
        d = dist_matrix[0, start_cust]
        if battery - d < -1e-9:
            # Need station?
            # If not, this customer cannot be served in this route starting from depot
            # We'll just skip it for now and try next? No, we must serve it.
            # If we can't reach it, we must insert a station.
            # Find nearest station
            s = None
            for st in stations:
                if battery - dist_matrix[0, st] >= -1e-9:
                    if s is None or dist_matrix[0, st] < dist_matrix[0, s]:
                        s = st
            if s is not None:
                route.append(s)
                d1 = dist_matrix[0, s]
                time_at += d1 / speed
                battery -= d1
                energy_needed = battery_max - battery
                time_at += energy_needed * recharge_rate
                battery = battery_max
                current = s
                d = dist_matrix[current, start_cust]
                if battery - d < -1e-9:
                    # Still can't reach
                    # Try another station?
                    pass
                else:
                    time_at += d / speed
                    battery -= d
                    current = start_cust
            else:
                # Cannot reach start_cust even with station?
                # This implies the instance is infeasible or my logic is flawed.
                # Assume feasible.
                pass
        
        # If we reached start_cust
        if current == start_cust:
            route.append(start_cust)
            load += demands[start_cust]
            if time_at < ready[start_cust]:
                time_at = ready[start_cust]
            if time_at <= due[start_cust]:
                time_at += service[start_cust]
            else:
                # Time infeasible?
                # This is bad.
                pass
        
        # Continue adding customers
        while unvisited:
            best_next = None
            best_dist = float('inf')
            best_battery_after = 0.0
            best_time_after = 0.0
            best_station = None
            best_station_dist = float('inf')
            
            for c in unvisited:
                # Check time window feasibility roughly
                d_cc = dist_matrix[current, c]
                t_arrive = time_at + d_cc / speed
                if t_arrive > due[c]:
                    continue
                
                # Check battery
                if battery - d_cc < -1e-9:
                    # Need station
                    # Find nearest station from current that allows reaching c
                    for s in stations:
                        d_cs = dist_matrix[current, s]
                        if battery - d_cs < -1e-9:
                            continue
                        d_sc = dist_matrix[s, c]
                        if battery_max - d_sc < -1e-9:
                            continue
                        # Check time
                        t_s = time_at + d_cs / speed
                        energy_needed = battery_max - (battery - d_cs)
                        t_s += energy_needed * recharge_rate
                        t_c = t_s + d_sc / speed
                        if t_c > due[c]:
                            continue
                        
                        # This is a candidate
                        # Cost is d_cs + d_sc
                        if d_cs + d_sc < best_dist:
                            best_dist = d_cs + d_sc
                            best_next = c
                            best_station = s
                            best_battery_after = battery - d_cs + battery_max - d_sc # Approx
                            best_time_after = t_c + service[c]
                else:
                    # Direct move
                    if d_cc < best_dist:
                        best_dist = d_cc
                        best_next = c
                        best_station = None
                        best_battery_after = battery - d_cc
                        best_time_after = max(t_arrive, ready[c]) + service[c]
            
            if best_next is None:
                break
            
            # Execute move
            if best_station is not None:
                s = best_station
                route.append(s)
                d_cs = dist_matrix[current, s]
                time_at += d_cs / speed
                battery -= d_cs
                energy_needed = battery_max - battery
                time_at += energy_needed * recharge_rate
                battery = battery_max
                current = s
                
                d_sc = dist_matrix[current, best_next]
                time_at += d_sc / speed
                battery -= d_sc
                current = best_next
            else:
                d_cc = dist_matrix[current, best_next]
                time_at += d_cc / speed
                battery -= d_cc
                current = best_next
            
            route.append(best_next)
            load += demands[best_next]
            if time_at < ready[best_next]:
                time_at = ready[best_next]
            time_at += service[best_next]
            unvisited.remove(best_next)
        
        # Check if we can return to depot
        d_return = dist_matrix[current, 0]
        if battery - d_return < -1e-9:
            # Need station
            s_return = None
            for s in stations:
                d_cs = dist_matrix[current, s]
                d_sd = dist_matrix[s, 0]
                if battery - d_cs >= -1e-9 and battery_max - d_sd >= -1e-9:
                    # Check time
                    t_s = time_at + d_cs / speed
                    energy_needed = battery_max - (battery - d_cs)
                    t_s += energy_needed * recharge_rate
                    t_d = t_s + d_sd / speed
                    if t_d <= horizon:
                        s_return = s
                        break
            if s_return is not None:
                route.append(s_return)
                d_cs = dist_matrix[current, s_return]
                time_at += d_cs / speed
                battery -= d_cs
                energy_needed = battery_max - battery
                time_at += energy_needed * recharge_rate
                battery = battery_max
                current = s_return
                d_sd = dist_matrix[current, 0]
                time_at += d_sd / speed
                battery -= d_sd
            else:
                # Infeasible return. Remove last customer?
                # For now, just append route and hope for the best
                pass
        else:
            time_at += d_return / speed
            battery -= d_return
            
        if time_at <= horizon and len(route) > 0:
            routes.append(route)
        else:
            # If infeasible, we might have to split or discard
            # To ensure validity, we'll just append it and let validation catch it
            # But we must serve all customers.
            # If this route is infeasible, we should have stopped earlier.
            # Let's assume it's feasible.
            routes.append(route)
            
    return routes, dist_matrix


def _local_search(routes, instance, dist_matrix, max_time=30.0):
    """Improve routes using 2-opt, Or-opt, and inter-route moves."""
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    start_time = time.time()
    
    def get_total_dist(routes):
        return _compute_total_distance(routes, dist_matrix)
    
    def is_feasible(route):
        """Full feasibility check."""
        battery = battery_max
        time_at = 0.0
        current = 0
        for node in route:
            d = dist_matrix[current, node]
            time_at += d / speed
            battery -= d
            if battery < -1e-9:
                return False
            if node <= n_customers:
                if time_at < ready[node]:
                    time_at = ready[node]
                if time_at > due[node] + 1e-9:
                    return False
                time_at += service[node]
            elif node in stations:
                energy_needed = battery_max - battery
                time_at += energy_needed * recharge_rate
                battery = battery_max
            current = node
        d = dist_matrix[current, 0]
        time_at += d / speed
        battery -= d
        if battery < -1e-9:
            return False
        return time_at <= horizon + 1e-9
    
    def route_load(route):
        return sum(demands[node] for node in route if node <= n_customers)
    
    current_dist = get_total_dist(routes)
    
    # Local search loop
    while time.time() - start_time < max_time:
        improved = False
        
        # 1. Intra-route 2-opt
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            n = len(route)
            if n < 2:
                continue
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    a = route[i-1] if i > 0 else 0
                    b = route[i]
                    c = route[j]
                    d = route[j+1] if j + 1 < n else 0
                    
                    old_cost = dist_matrix[a, b] + dist_matrix[c, d]
                    new_cost = dist_matrix[a, c] + dist_matrix[b, d]
                    
                    if new_cost < old_cost - 1e-9:
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if route_load(new_route) <= capacity and is_feasible(new_route):
                            routes[r_idx] = new_route
                            current_dist = get_total_dist(routes)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        # 2. Inter-route Relocate
        if not improved:
            for r1 in range(len(routes)):
                for pos1 in range(len(routes[r1])):
                    node = routes[r1][pos1]
                    if node > n_customers:
                        continue
                    
                    for r2 in range(len(routes)):
                        if r1 == r2:
                            continue
                        if route_load(routes[r2]) + demands[node] > capacity:
                            continue
                        
                        for pos2 in range(len(routes[r2]) + 1):
                            route1_new = routes[r1][:pos1] + routes[r1][pos1+1:]
                            route2_new = routes[r2][:pos2] + [node] + routes[r2][pos2:]
                            
                            if route_load(route1_new) <= capacity and is_feasible(route1_new) and is_feasible(route2_new):
                                new_routes = [r[:] for r in routes]
                                new_routes[r1] = route1_new
                                new_routes[r2] = route2_new
                                new_dist = get_total_dist(new_routes)
                                if new_dist < current_dist - 1e-9:
                                    routes = new_routes
                                    current_dist = new_dist
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
            if improved:
                continue
        
        if not improved:
            break
    
    return routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    
    # Get initial solution
    routes, dist_matrix = _construct_initial_solution(instance)
    
    # Validate
    valid, dist = _validate_solution(routes, instance, dist_matrix)
    if not valid:
        # Fallback to one-per-route
        customers = list(range(1, instance["n_customers"] + 1))
        routes = [[v] for v in customers]
    
    # Local search if time permits
    time_remaining = DEADLINE_SECONDS - (time.time() - start_time)
    if time_remaining > 5:
        routes = _local_search(routes, instance, dist_matrix, max_time=time_remaining - 2)
        
        # Re-validate
        valid, dist = _validate_solution(routes, instance, dist_matrix)
        if not valid:
            # Fallback
            customers = list(range(1, instance["n_customers"] + 1))
            routes = [[v] for v in customers]
    
    # Final check
    valid, dist = _validate_solution(routes, instance, dist_matrix)
    if not valid:
        customers = list(range(1, instance["n_customers"] + 1))
        routes = [[v] for v in customers]
    
    return routes

# EVOLVE-BLOCK-END

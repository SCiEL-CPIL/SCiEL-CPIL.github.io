"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt


def _dist(coords, i, j):
    dx = coords[i, 0] - coords[j, 0]
    dy = coords[i, 1] - coords[j, 1]
    return sqrt(dx * dx + dy * dy)


def _build_distance_matrix(coords):
    n = len(coords)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _dist(coords, i, j)
            dm[i, j] = d
            dm[j, i] = d
    return dm


def _simulate_route(route, dm, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon):
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
        d = dm[cur_node, node]
        cur_time += d / speed
        cur_bat -= d
        total_dist += d
        
        if cur_bat < -1e-9:
            return False, total_dist
        
        if node in stations:
            # At station: recharge
            energy_needed = battery - cur_bat
            cur_time += energy_needed * recharge_rate
            cur_bat = battery
        else:
            # At customer: check time window, add service
            if cur_time < ready[node]:
                cur_time = ready[node]
            if cur_time > due[node] + 1e-9:
                return False, total_dist
            cur_time += service[node]
            load += demands[node]
            if load > capacity + 1e-9:
                return False, total_dist
        
        cur_node = node
    
    # Return to depot
    d = dm[cur_node, 0]
    cur_time += d / speed
    cur_bat -= d
    total_dist += d
    
    if cur_bat < -1e-9:
        return False, total_dist
    if cur_time > horizon + 1e-9:
        return False, total_dist
    
    return True, total_dist


def _two_opt_improve(route, dm, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon):
    """Apply 2-opt improvement to a route, maintaining feasibility."""
    if len(route) <= 2:
        return route
    
    improved = True
    max_iterations = 100
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                # Try reversing segment [i+1, j]
                new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                
                valid, new_dist = _simulate_route(new_route, dm, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                if valid:
                    _, old_dist = _simulate_route(route, dm, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                    if new_dist < old_dist - 1e-9:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    
    return route


def _solve_instance(instance, deadline=None):
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
    
    # Build distance matrix for faster lookups
    dm = _build_distance_matrix(coords)
    
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: due[v])
    
    routes = []
    unserved = set(customers)
    
    # Greedy nearest-neighbor insertion with battery/time-window simulation
    while unserved:
        if deadline is not None and time.time() > deadline:
            break
        
        # Start a new route
        current_route = []
        cur_node = 0
        cur_time = 0.0
        cur_bat = battery
        cur_load = 0.0
        
        while unserved:
            if deadline is not None and time.time() > deadline:
                break
            
            best_cust = None
            best_dist = float('inf')
            
            for cust in unserved:
                d = dm[cur_node, cust]
                if d < best_dist:
                    # Check if we can serve this customer
                    test_time = cur_time + d / speed
                    test_bat = cur_bat - d
                    if test_bat < -1e-9:
                        continue
                    if test_time < ready[cust]:
                        test_time = ready[cust]
                    if test_time > due[cust] + 1e-9:
                        continue
                    test_time += service[cust]
                    if cur_load + demands[cust] > capacity + 1e-9:
                        continue
                    best_cust = cust
                    best_dist = d
            
            if best_cust is None:
                break
            
            # Move to best customer
            d = dm[cur_node, best_cust]
            cur_time += d / speed
            cur_bat -= d
            if cur_time < ready[best_cust]:
                cur_time = ready[best_cust]
            cur_time += service[best_cust]
            cur_load += demands[best_cust]
            cur_node = best_cust
            current_route.append(best_cust)
            unserved.discard(best_cust)
        
        if not current_route:
            if unserved:
                # If we couldn't add any customer, force-serve one to avoid infinite loop
                cust = min(unserved, key=lambda c: due[c])
                unserved.discard(cust)
                routes.append([cust])
            continue
        
        # Check if we can return to depot
        d = dm[cur_node, 0]
        test_time = cur_time + d / speed
        test_bat = cur_bat - d
        if test_bat < -1e-9 or test_time > horizon + 1e-9:
            # Need to insert a station before returning
            # Try inserting the nearest station
            best_station = None
            best_sdist = float('inf')
            for st in stations:
                d_st = dm[cur_node, st]
                if d_st < best_sdist:
                    # Check if we can reach the station
                    if cur_bat - d_st >= -1e-9:
                        best_station = st
                        best_sdist = d_st
            
            if best_station is not None:
                # Go to station
                d = dm[cur_node, best_station]
                cur_time += d / speed
                cur_bat -= d
                # Recharge
                energy_needed = battery - cur_bat
                cur_time += energy_needed * recharge_rate
                cur_bat = battery
                cur_node = best_station
                current_route.append(best_station)
                
                # Now try to return to depot
                d = dm[cur_node, 0]
                test_time = cur_time + d / speed
                test_bat = cur_bat - d
                if test_bat < -1e-9 or test_time > horizon + 1e-9:
                    # Still can't return; this route is invalid
                    # Put back the last customer
                    last = current_route[-1]
                    if last not in stations:
                        unserved.add(last)
                        current_route.pop()
                    elif len(current_route) > 1:
                        # Last was a station, try removing it
                        current_route.pop()
                        cur_node = current_route[-1] if current_route else 0
                    continue
            else:
                # Can't reach any station; put back the last customer
                last = current_route[-1]
                if last not in stations:
                    unserved.add(last)
                    current_route.pop()
                continue
        
        routes.append(current_route)
    
    # If there are still unserved customers (due to timeout), serve them individually
    if unserved:
        for cust in sorted(unserved, key=lambda c: due[c]):
            routes.append([cust])
        unserved.clear()
    
    # Apply 2-opt improvement to each route
    if deadline is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            for idx, route in enumerate(routes):
                if time.time() > deadline:
                    break
                routes[idx] = _two_opt_improve(route, dm, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
    
    # Verify all routes are valid
    for route in routes:
        valid, _ = _simulate_route(route, dm, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
        if not valid:
            # Fallback: one customer per route
            customers = list(range(1, n_customers + 1))
            customers.sort(key=lambda v: due[v])
            return [[v] for v in customers]
    
    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    try:
        routes = _solve_instance(instance, deadline=deadline)
    except Exception:
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        return [[v] for v in customers]
    
    return routes
# EVOLVE-BLOCK-END

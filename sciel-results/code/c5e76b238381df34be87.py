"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def _get_distance_matrix(coords):
    """Precompute Euclidean distance matrix using NumPy."""
    n = len(coords)
    arr = np.array(coords)
    dx = arr[:, 0, None] - arr[None, :, 0]
    dy = arr[:, 1, None] - arr[None, :, 1]
    dist = np.sqrt(dx**2 + dy**2)
    return dist

def _route_distance_fast(dist_matrix, route):
    """Calculate total distance of a route using precomputed distance matrix."""
    if not route:
        return 0.0
    d = dist_matrix[0][route[0]]
    for k in range(len(route) - 1):
        d += dist_matrix[route[k]][route[k + 1]]
    d += dist_matrix[route[-1]][0]
    return d

def _check_route_feasibility_fast(dist_matrix, demands, ready, due, service,
                              battery, recharge_rate, horizon, route,
                              capacity, n_customers, stations):
    """
    Check if a single route is feasible using precomputed distances.
    Returns (feasible, total_distance).
    """
    if not route:
        return True, 0.0

    # Check capacity
    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
            if total_demand > capacity:
                return False, 0.0

    # Simulate
    t = 0.0
    b = battery
    prev = 0
    total_dist = 0.0
    
    for v in route:
        d = dist_matrix[prev][v]
        t += d
        b -= d
        total_dist += d
        if b < -1e-9:
            return False, 0.0
        
        if v <= n_customers:
            # Customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
        else:
            # Station
            # "a station adds (battery - energy) * recharge_rate time and refills"
            # The "energy" here is the energy consumed to reach the station? 
            # Or is it the current battery level?
            # "refills" implies battery becomes full.
            # The time cost is (battery - energy) * recharge_rate.
            # If energy is the amount of energy used to get here, that's just d.
            # But usually charging time depends on how much charge you need to add.
            # If you arrive with b, you need to add (capacity - b).
            # The text says "(battery - energy) * recharge_rate".
            # "battery" likely refers to the max battery capacity (300).
            # "energy" likely refers to the current battery level upon arrival.
            # So time = (300 - b) * 0.5.
            # Let's assume this interpretation.
            time_charging = (battery - b) * recharge_rate
            t += time_charging
            b = battery  # Refill to full
        
        prev = v

    # Return to depot
    d = dist_matrix[prev][0]
    t += d
    b -= d
    total_dist += d
    if b < -1e-9:
        return False, 0.0
    if t > horizon + 1e-9:
        return False, 0.0

    return True, total_dist

def _solve_instance(instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance.get("speed", 1.0)
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance["horizon"]

    # Precompute distance matrix
    dist_matrix = _get_distance_matrix(coords)

    all_customers = list(range(1, n_customers + 1))
    
    # Initial solution: Greedy insertion
    # Sort customers by due time to prioritize urgent ones? 
    # Or nearest neighbor?
    # Let's start with one customer per route, then merge.
    # Or start with empty routes and insert.
    
    routes = []
    unassigned = sorted(all_customers, key=lambda c: due[c])
    
    # Helper to get current demand of a route
    def _route_demand(route):
        return sum(demands[v] for v in route if v <= n_customers)

    for c in unassigned:
        best_route_idx = None
        best_dist_increase = float('inf')
        best_new_route = None
        
        # Try inserting into each existing route
        for i, route in enumerate(routes):
            # Quick capacity check
            if _route_demand(route) + demands[c] > capacity:
                continue
                
            # Try all positions
            best_pos_in_route = None
            best_route_dist = float('inf')
            
            # Optimization: Only check a subset of positions? 
            # With small route sizes, check all.
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                feasible, nd = _check_route_feasibility_fast(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, new_route,
                    capacity, n_customers, stations
                )
                if feasible and nd < best_route_dist:
                    best_route_dist = nd
                    best_pos_in_route = pos
            
            if best_pos_in_route is not None:
                # Calculate increase in total distance
                old_route_dist = _route_distance_fast(dist_matrix, route)
                increase = best_route_dist - old_route_dist
                if increase < best_dist_increase:
                    best_dist_increase = increase
                    best_route_idx = i
                    best_new_route = route[:best_pos_in_route] + [c] + route[best_pos_in_route:]
        
        if best_route_idx is not None:
            routes[best_route_idx] = best_new_route
        else:
            routes.append([c])

    # Local Search: Or-opt / Relocate
    # We try to move a customer from one route to another (or reposition within same route)
    # To be fast, we limit the number of passes.
    
    # Precompute current total distance
    current_total_dist = sum(_route_distance_fast(dist_matrix, r) for r in routes)
    
    max_iterations = 100
    iteration = 0
    improved = True
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Iterate through all customers
        for c in all_customers:
            # Find current route
            curr_route_idx = -1
            for i, r in enumerate(routes):
                if c in r:
                    curr_route_idx = i
                    break
            if curr_route_idx == -1:
                continue
            
            curr_route = routes[curr_route_idx]
            
            # Remove c from current route
            remaining_curr = [v for v in curr_route if v != c]
            
            best_target_idx = None
            best_improvement = 0.0
            best_new_routes_state = None
            
            # Calculate distance of current route without c
            dist_curr_without_c = _route_distance_fast(dist_matrix, remaining_curr) if remaining_curr else 0.0
            
            # Try inserting c into every other route (and current route at different pos)
            for i, route in enumerate(routes):
                if i == curr_route_idx:
                    # Reinserting into same route
                    # We already removed c, so 'route' here is the original. 
                    # We need to try inserting into 'remaining_curr'
                    target_route = remaining_curr
                    old_target_dist = dist_curr_without_c
                else:
                    target_route = route
                    old_target_dist = _route_distance_fast(dist_matrix, route)
                
                # Quick capacity check
                if _route_demand(target_route) + demands[c] > capacity:
                    continue
                
                best_pos = None
                best_target_dist = float('inf')
                
                for pos in range(len(target_route) + 1):
                    new_target_route = target_route[:pos] + [c] + target_route[pos:]
                    feasible, nd = _check_route_feasibility_fast(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_target_route,
                        capacity, n_customers, stations
                    )
                    if feasible and nd < best_target_dist:
                        best_target_dist = nd
                        best_pos = pos
                
                if best_pos is not None:
                    # Calculate total distance change
                    # Old total: dist_curr (with c) + sum(other routes)
                    # New total: dist_curr_without_c + new_target_dist + sum(other routes)
                    # Note: if i == curr_route_idx, "other routes" doesn't include curr.
                    # Let's just compute the delta for the affected routes.
                    
                    if i == curr_route_idx:
                        # Only current route changes
                        # Old dist: _route_distance_fast(dist_matrix, curr_route)
                        # New dist: best_target_dist
                        old_affected = _route_distance_fast(dist_matrix, curr_route)
                        new_affected = best_target_dist
                        delta = new_affected - old_affected
                    else:
                        # Current route loses c, target route gains c
                        old_affected = _route_distance_fast(dist_matrix, curr_route) + old_target_dist
                        new_affected = dist_curr_without_c + best_target_dist
                        delta = new_affected - old_affected
                    
                    if delta < best_improvement:
                        best_improvement = delta
                        best_target_idx = i
                        # Store the new state
                        # We need to construct the new routes list
                        new_routes = routes.copy()
                        
                        # Update current route
                        if i == curr_route_idx:
                            new_routes[curr_route_idx] = target_route[:best_pos] + [c] + target_route[best_pos:]
                        else:
                            # Remove c from current route
                            new_routes[curr_route_idx] = remaining_curr
                            # Add c to target route
                            new_routes[i] = route[:best_pos] + [c] + route[best_pos:]
                        
                        # If a route becomes empty, we can keep it or remove it.
                        # Keeping it is simpler for indexing, but increases overhead.
                        # Let's keep it to avoid index shifting issues in this loop.
                        
                        best_new_routes_state = new_routes
            
            if best_target_idx is not None and best_improvement < -1e-6:
                routes = best_new_routes_state
                improved = True
                # Update total distance (optional, for tracking)
                # current_total_dist += best_improvement
                # Break early? No, continue to find more improvements.
                
    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Solve
    routes = _solve_instance(instance)
    
    # Final Verification
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance["horizon"]
    
    dist_matrix = _get_distance_matrix(coords)
    
    all_ok = True
    for r in routes:
        ok, _ = _check_route_feasibility_fast(
            dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, r,
            capacity, n_customers, stations
        )
        if not ok:
            all_ok = False
            break
            
    if not all_ok:
        # Fallback: One customer per route
        customers = list(range(1, n_customers + 1))
        routes = [[c] for c in customers]
        
    return routes
# EVOLVE-BLOCK-END

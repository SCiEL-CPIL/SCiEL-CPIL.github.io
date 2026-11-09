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

def _two_opt_route(dist_matrix, demands, ready, due, service,
                   battery, recharge_rate, horizon, route,
                   capacity, n_customers, stations):
    """
    Apply 2-opt local search to a single route to minimize distance.
    Only considers swaps that keep the route feasible.
    Returns the improved route.
    """
    if len(route) < 2:
        return route

    # Extract customer indices only (stations are not present in our current solution generation)
    # The route contains only customers in the current implementation
    n = len(route)
    best_route = route.copy()
    improved = True
    max_iter = 50  # Limit iterations to prevent infinite loops or excessive time
    
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment [i..j]
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                
                feasible, nd = _check_route_feasibility_fast(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, new_route,
                    capacity, n_customers, stations
                )
                
                if feasible:
                    old_dist = _route_distance_fast(dist_matrix, best_route)
                    if nd < old_dist - 1e-6:
                        best_route = new_route
                        improved = True
                        # Note: We don't break immediately to allow multiple improvements in one pass,
                        # but we update best_route. The next i, j will use the updated best_route.
                        # This is slightly expensive as we recalc old_dist, but n is small.
                        
    return best_route

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
    
    max_iterations = 50
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
                    if i == curr_route_idx:
                        old_affected = _route_distance_fast(dist_matrix, curr_route)
                        new_affected = best_target_dist
                        delta = new_affected - old_affected
                    else:
                        old_affected = _route_distance_fast(dist_matrix, curr_route) + old_target_dist
                        new_affected = dist_curr_without_c + best_target_dist
                        delta = new_affected - old_affected
                    
                    if delta < best_improvement:
                        best_improvement = delta
                        best_target_idx = i
                        new_routes = routes.copy()
                        
                        if i == curr_route_idx:
                            new_routes[curr_route_idx] = target_route[:best_pos] + [c] + target_route[best_pos:]
                        else:
                            new_routes[curr_route_idx] = remaining_curr
                            new_routes[i] = route[:best_pos] + [c] + route[best_pos:]
                        
                        best_new_routes_state = new_routes
            
            if best_target_idx is not None and best_improvement < -1e-6:
                routes = best_new_routes_state
                improved = True
                
    # Final 2-opt pass on each route to fine-tune ordering
    for i in range(len(routes)):
        routes[i] = _two_opt_route(
            dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, routes[i],
            capacity, n_customers, stations
        )
        # Remove empty routes if any
        if not routes[i]:
            routes.pop(i)
            i -= 1

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

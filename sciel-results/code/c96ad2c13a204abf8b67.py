"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def _get_distance_matrix(coords):
    """Precompute Euclidean distance matrix using NumPy."""
    arr = np.array(coords, dtype=np.float64)
    dx = arr[:, 0, None] - arr[None, :, 0]
    dy = arr[:, 1, None] - arr[None, :, 1]
    dist = np.sqrt(dx**2 + dy**2)
    return dist

def _route_distance_fast(dist_matrix, route):
    """Calculate total distance of a route using precomputed distance matrix."""
    if not route:
        return 0.0
    n = len(route)
    if n == 0:
        return 0.0
    return (dist_matrix[0, route[0]] + 
            np.sum(dist_matrix[route[:-1], route[1:]]) + 
            dist_matrix[route[-1], 0])

def _check_route_feasibility(dist_matrix, demands, ready, due, service,
                              battery, recharge_rate, horizon, route,
                              capacity, n_customers, stations):
    """
    Check if a single route is feasible.
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
            b = battery
        
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

def _fast_feasibility_check(dist_matrix, demands, ready, due, service,
                              battery, recharge_rate, horizon, route,
                              capacity, n_customers, stations):
    """
    Fast feasibility check that returns (feasible, total_distance) without full simulation
    when possible. Uses early termination.
    """
    if not route:
        return True, 0.0

    # Quick capacity check
    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
            if total_demand > capacity:
                return False, 0.0

    # Full simulation (needed for time windows and battery)
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
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
        else:
            time_charging = (battery - b) * recharge_rate
            t += time_charging
            b = battery
        
        prev = v

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
                   capacity, n_customers, stations, deadline=None):
    """
    Apply 2-opt local search to a single route.
    Uses incremental distance calculation for speed.
    """
    if len(route) < 2:
        return route

    n = len(route)
    best_route = list(route)
    current_dist = _route_distance_fast(dist_matrix, best_route)
    
    # Precompute feasibility of current route
    feasible, _ = _fast_feasibility_check(
        dist_matrix, demands, ready, due, service,
        battery, recharge_rate, horizon, best_route,
        capacity, n_customers, stations
    )
    if not feasible:
        return route  # Can't improve an infeasible route with 2-opt alone

    improved = True
    max_iter = 30
    
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        if deadline and time.time() > deadline:
            break
        
        for i in range(n - 1):
            if deadline and time.time() > deadline:
                break
            for j in range(i + 1, n):
                if deadline and time.time() > deadline:
                    break
                
                # Calculate distance change for reversing segment [i..j]
                # Old edges: (i-1 -> i) and (j -> j+1)
                # New edges: (i-1 -> j) and (i -> j+1)
                # (with depot as 0 at boundaries)
                
                if i == 0:
                    old1 = dist_matrix[0][route[i]]
                    new1 = dist_matrix[0][route[j]]
                else:
                    old1 = dist_matrix[route[i-1]][route[i]]
                    new1 = dist_matrix[route[i-1]][route[j]]
                
                if j == n - 1:
                    old2 = dist_matrix[route[j]][0]
                    new2 = dist_matrix[route[i]][0]
                else:
                    old2 = dist_matrix[route[j]][route[j+1]]
                    new2 = dist_matrix[route[i]][route[j+1]]
                
                delta_d = (new1 + new2) - (old1 + old2)
                
                # Quick rejection: if distance doesn't improve, skip
                if delta_d >= -1e-6:
                    continue
                
                # Check if reversal improves distance
                new_dist = current_dist + delta_d
                if new_dist >= current_dist - 1e-6:
                    continue
                
                # Create new route and check feasibility
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                
                feasible, nd = _fast_feasibility_check(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, new_route,
                    capacity, n_customers, stations
                )
                
                if feasible and nd < current_dist - 1e-6:
                    best_route = new_route
                    current_dist = nd
                    improved = True

    return best_route

def _or_opt_route(dist_matrix, demands, ready, due, service,
                  battery, recharge_rate, horizon, route,
                  capacity, n_customers, stations, deadline=None):
    """
    Apply Or-opt moves (relocate 1-3 consecutive nodes) to a single route.
    """
    if len(route) < 2:
        return route

    n = len(route)
    best_route = list(route)
    current_dist = _route_distance_fast(dist_matrix, best_route)
    
    feasible, _ = _fast_feasibility_check(
        dist_matrix, demands, ready, due, service,
        battery, recharge_rate, horizon, best_route,
        capacity, n_customers, stations
    )
    if not feasible:
        return route

    improved = True
    max_iter = 20
    
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        if deadline and time.time() > deadline:
            break
        
        # Try relocating segments of length 1, 2, 3
        for seg_len in [1, 2, 3]:
            if deadline and time.time() > deadline:
                break
            
            for i in range(n - seg_len + 1):
                if deadline and time.time() > deadline:
                    break
                
                segment = best_route[i:i+seg_len]
                
                # Remove segment
                remaining = best_route[:i] + best_route[i+seg_len:]
                if not remaining:
                    continue
                
                # Try inserting at each position in remaining
                for pos in range(len(remaining) + 1):
                    if pos <= i and pos >= i - seg_len + 1:
                        # Would put it back roughly where it was, skip
                        if pos == i:
                            continue
                    
                    new_route = remaining[:pos] + segment + remaining[pos:]
                    
                    # Quick distance check
                    new_dist = _route_distance_fast(dist_matrix, new_route)
                    if new_dist >= current_dist - 1e-6:
                        continue
                    
                    feasible, nd = _fast_feasibility_check(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_route,
                        capacity, n_customers, stations
                    )
                    
                    if feasible and nd < current_dist - 1e-6:
                        best_route = new_route
                        current_dist = nd
                        improved = True
                        # Restart segment search with new route
                        break
                if improved:
                    break
            if improved:
                break

    return best_route

def _relocate_between_routes(routes, dist_matrix, demands, ready, due, service,
                              battery, recharge_rate, horizon, n_customers, stations,
                              deadline=None):
    """
    Try to relocate customers between routes to improve total distance.
    """
    n_routes = len(routes)
    if n_routes < 2:
        return routes, False

    current_total = sum(_route_distance_fast(dist_matrix, r) for r in routes)
    improved = False
    
    # Build customer-to-route mapping
    cust_to_route = {}
    for i, r in enumerate(routes):
        for c in r:
            if c <= n_customers:
                cust_to_route[c] = i

    for c in list(cust_to_route.keys()):
        if deadline and time.time() > deadline:
            break
        
        src_idx = cust_to_route[c]
        src_route = routes[src_idx]
        
        # Remove c from source
        src_without = [v for v in src_route if v != c]
        
        # Try inserting into each other route
        best_target = -1
        best_delta = 0.0
        best_new_routes = None
        
        for i, route in enumerate(routes):
            if i == src_idx:
                continue
            
            # Quick capacity check
            target_demand = sum(demands[v] for v in route if v <= n_customers)
            if target_demand + demands[c] > 100:  # capacity
                continue
            
            # Try all positions
            best_pos = -1
            best_target_dist = float('inf')
            
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                feasible, nd = _fast_feasibility_check(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, new_route,
                    n_customers, n_customers, stations
                )
                if feasible and nd < best_target_dist:
                    best_target_dist = nd
                    best_pos = pos
            
            if best_pos >= 0:
                src_dist_new = _route_distance_fast(dist_matrix, src_without) if src_without else 0.0
                old_total = _route_distance_fast(dist_matrix, src_route) + _route_distance_fast(dist_matrix, route)
                new_total = src_dist_new + best_target_dist
                delta = new_total - old_total
                
                if delta < best_delta - 1e-6:
                    best_delta = delta
                    new_routes = [list(r) for r in routes]
                    new_routes[src_idx] = src_without
                    new_routes[i] = route[:best_pos] + [c] + route[best_pos:]
                    best_new_routes = new_routes
                    best_target = i
        
        if best_target >= 0 and best_delta < -1e-6:
            routes = best_new_routes
            improved = True
            # Update cust_to_route
            cust_to_route = {}
            for i, r in enumerate(routes):
                for cc in r:
                    if cc <= n_customers:
                        cust_to_route[cc] = i
    
    return routes, improved

def _solve_instance(instance, deadline=None):
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
    
    # Initial solution: Greedy insertion sorted by due time
    routes = []
    unassigned = sorted(all_customers, key=lambda c: due[c])
    
    def _route_demand(route):
        return sum(demands[v] for v in route if v <= n_customers)

    for c in unassigned:
        best_route_idx = None
        best_dist_increase = float('inf')
        best_new_route = None
        
        for i, route in enumerate(routes):
            if _route_demand(route) + demands[c] > capacity:
                continue
                
            best_pos_in_route = None
            best_route_dist = float('inf')
            
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                feasible, nd = _fast_feasibility_check(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, new_route,
                    capacity, n_customers, stations
                )
                if feasible and nd < best_route_dist:
                    best_route_dist = nd
                    best_pos_in_route = pos
            
            if best_pos_in_route is not None:
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

    # Remove empty routes
    routes = [r for r in routes if r]

    # Local search phase 1: Or-opt within each route
    for i in range(len(routes)):
        if deadline and time.time() > deadline:
            break
        routes[i] = _or_opt_route(
            dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, routes[i],
            capacity, n_customers, stations, deadline
        )

    # Local search phase 2: Relocate between routes
    max_relocate_passes = 10
    for _ in range(max_relocate_passes):
        if deadline and time.time() > deadline:
            break
        routes, improved = _relocate_between_routes(
            routes, dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, n_customers, stations, deadline
        )
        if not improved:
            break

    # Local search phase 3: 2-opt within each route
    for i in range(len(routes)):
        if deadline and time.time() > deadline:
            break
        routes[i] = _two_opt_route(
            dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, routes[i],
            capacity, n_customers, stations, deadline
        )

    # Final: Or-opt again after 2-opt
    for i in range(len(routes)):
        if deadline and time.time() > deadline:
            break
        routes[i] = _or_opt_route(
            dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, routes[i],
            capacity, n_customers, stations, deadline
        )

    # Remove any empty routes
    routes = [r for r in routes if r]

    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    try:
        routes = _solve_instance(instance, deadline)
    except Exception:
        # Fallback: one customer per route
        n_customers = instance["n_customers"]
        routes = [[c] for c in range(1, n_customers + 1)]
        return routes

    # Final verification
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
        ok, _ = _check_route_feasibility(
            dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, r,
            capacity, n_customers, stations
        )
        if not ok:
            all_ok = False
            break
    
    # Verify all customers served exactly once
    served = []
    for r in routes:
        for v in r:
            if v <= n_customers:
                served.append(v)
    if sorted(served) != list(range(1, n_customers + 1)):
        all_ok = False
    
    if not all_ok:
        # Fallback: one customer per route
        customers = list(range(1, n_customers + 1))
        routes = [[c] for c in customers]
    
    return routes
# EVOLVE-BLOCK-END

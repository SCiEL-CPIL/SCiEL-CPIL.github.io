import numpy as np
import time
from itertools import combinations

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
    # Using advanced indexing for speed
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
            # Charging time depends on current battery level
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
    
    # Initial solution: Greedy insertion
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
                feasible, nd = _check_route_feasibility(
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
    
    if not routes:
        return []

    current_total_dist = sum(_route_distance_fast(dist_matrix, r) for r in routes)
    best_routes = [list(r) for r in routes]
    best_total_dist = current_total_dist

    # Local Search
    # We will alternate between intra-route and inter-route moves.
    
    def _two_opt_route(route):
        """Apply 2-opt to a single route."""
        if len(route) < 2:
            return route
        n = len(route)
        best_route = list(route)
        current_dist = _route_distance_fast(dist_matrix, best_route)
        feasible, _ = _check_route_feasibility(
            dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, best_route,
            capacity, n_customers, stations
        )
        if not feasible:
            return route

        improved = True
        max_iter = 50
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
                    # Calculate delta
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
                    if delta_d >= -1e-6:
                        continue
                    
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    feasible, nd = _check_route_feasibility(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_route,
                        capacity, n_customers, stations
                    )
                    if feasible and nd < current_dist - 1e-6:
                        best_route = new_route
                        current_dist = nd
                        improved = True
        return best_route

    def _or_opt_route(route):
        """Apply Or-opt to a single route."""
        if len(route) < 2:
            return route
        n = len(route)
        best_route = list(route)
        current_dist = _route_distance_fast(dist_matrix, best_route)
        feasible, _ = _check_route_feasibility(
            dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, best_route,
            capacity, n_customers, stations
        )
        if not feasible:
            return route

        improved = True
        max_iter = 50
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if deadline and time.time() > deadline:
                break
            for seg_len in [1, 2, 3]:
                if deadline and time.time() > deadline:
                    break
                for i in range(n - seg_len + 1):
                    if deadline and time.time() > deadline:
                        break
                    segment = best_route[i:i+seg_len]
                    remaining = best_route[:i] + best_route[i+seg_len:]
                    if not remaining:
                        continue
                    for pos in range(len(remaining) + 1):
                        if pos == i:
                            continue
                        new_route = remaining[:pos] + segment + remaining[pos:]
                        new_dist = _route_distance_fast(dist_matrix, new_route)
                        if new_dist >= current_dist - 1e-6:
                            continue
                        feasible, nd = _check_route_feasibility(
                            dist_matrix, demands, ready, due, service,
                            battery, recharge_rate, horizon, new_route,
                            capacity, n_customers, stations
                        )
                        if feasible and nd < current_dist - 1e-6:
                            best_route = new_route
                            current_dist = nd
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best_route

    def _inter_route_moves(routes):
        """Apply inter-route moves: Relocate, Swap, CROSS."""
        nonlocal current_total_dist, best_routes, best_total_dist
        n_routes = len(routes)
        if n_routes < 2:
            return routes, False

        improved = False
        
        # 1. Relocate
        for i in range(n_routes):
            for j in range(n_routes):
                if i == j:
                    continue
                src_route = routes[i]
                tgt_route = routes[j]
                
                # Try relocating each customer from src to tgt
                for k in range(len(src_route)):
                    if deadline and time.time() > deadline:
                        return routes, improved
                    c = src_route[k]
                    if c > n_customers: # Skip stations for now, handle separately if needed
                        continue
                    
                    # Capacity check
                    if _route_demand(tgt_route) + demands[c] > capacity:
                        continue
                    
                    # Remove from src
                    src_new = src_route[:k] + src_route[k+1:]
                    
                    # Try inserting into tgt at all positions
                    best_pos = -1
                    best_delta = float('inf')
                    
                    for pos in range(len(tgt_route) + 1):
                        tgt_new = tgt_route[:pos] + [c] + tgt_route[pos:]
                        
                        # Check feasibility of both new routes
                        feas_src, dist_src = _check_route_feasibility(
                            dist_matrix, demands, ready, due, service,
                            battery, recharge_rate, horizon, src_new,
                            capacity, n_customers, stations
                        )
                        if not feas_src:
                            continue
                            
                        feas_tgt, dist_tgt = _check_route_feasibility(
                            dist_matrix, demands, ready, due, service,
                            battery, recharge_rate, horizon, tgt_new,
                            capacity, n_customers, stations
                        )
                        if not feas_tgt:
                            continue
                        
                        new_total = dist_src + dist_tgt
                        old_total = _route_distance_fast(dist_matrix, src_route) + _route_distance_fast(dist_matrix, tgt_route)
                        delta = new_total - old_total
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = pos
                            best_src_new = src_new
                            best_tgt_new = tgt_new
                            
                    if best_pos != -1 and best_delta < -1e-6:
                        routes[i] = best_src_new
                        routes[j] = best_tgt_new
                        improved = True
                        # Update global best
                        current_total_dist = sum(_route_distance_fast(dist_matrix, r) for r in routes)
                        if current_total_dist < best_total_dist:
                            best_total_dist = current_total_dist
                            best_routes = [list(r) for r in routes]
                        break # Restart search for this pair
                if improved:
                    break
            if improved:
                break

        if improved:
            return routes, True

        # 2. Swap
        for i in range(n_routes):
            for j in range(i + 1, n_routes):
                if deadline and time.time() > deadline:
                    return routes, improved
                src_route = routes[i]
                tgt_route = routes[j]
                
                for k in range(len(src_route)):
                    if deadline and time.time() > deadline:
                        return routes, improved
                    c1 = src_route[k]
                    if c1 > n_customers:
                        continue
                    for l in range(len(tgt_route)):
                        c2 = tgt_route[l]
                        if c2 > n_customers:
                            continue
                        if c1 == c2:
                            continue
                        
                        # Capacity check
                        if _route_demand(src_route) - demands[c1] + demands[c2] > capacity:
                            continue
                        if _route_demand(tgt_route) - demands[c2] + demands[c1] > capacity:
                            continue
                        
                        src_new = src_route[:k] + [c2] + src_route[k+1:]
                        tgt_new = tgt_route[:l] + [c1] + tgt_route[l+1:]
                        
                        feas_src, dist_src = _check_route_feasibility(
                            dist_matrix, demands, ready, due, service,
                            battery, recharge_rate, horizon, src_new,
                            capacity, n_customers, stations
                        )
                        if not feas_src:
                            continue
                            
                        feas_tgt, dist_tgt = _check_route_feasibility(
                            dist_matrix, demands, ready, due, service,
                            battery, recharge_rate, horizon, tgt_new,
                            capacity, n_customers, stations
                        )
                        if not feas_tgt:
                            continue
                        
                        new_total = dist_src + dist_tgt
                        old_total = _route_distance_fast(dist_matrix, src_route) + _route_distance_fast(dist_matrix, tgt_route)
                        delta = new_total - old_total
                        
                        if delta < -1e-6:
                            routes[i] = src_new
                            routes[j] = tgt_new
                            improved = True
                            current_total_dist = sum(_route_distance_fast(dist_matrix, r) for r in routes)
                            if current_total_dist < best_total_dist:
                                best_total_dist = current_total_dist
                                best_routes = [list(r) for r in routes]
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        return routes, improved

    # Main loop
    max_passes = 100
    for p in range(max_passes):
        if deadline and time.time() > deadline:
            break
            
        # Intra-route
        for i in range(len(routes)):
            if deadline and time.time() > deadline:
                break
            routes[i] = _two_opt_route(routes[i])
            routes[i] = _or_opt_route(routes[i])
            
        # Inter-route
        routes, inter_improved = _inter_route_moves(routes)
        
        # Update current total
        current_total_dist = sum(_route_distance_fast(dist_matrix, r) for r in routes)
        if current_total_dist < best_total_dist:
            best_total_dist = current_total_dist
            best_routes = [list(r) for r in routes]

        if not inter_improved:
            # Check if intra-route improved
            # We don't track intra-route improvement separately in this simplified loop, 
            # but if inter didn't improve, we might still be converging.
            # To be safe, we break if no inter-route move was found, assuming intra-route 
            # has converged or will be caught in the next pass if we continue.
            # However, a stricter check is better.
            pass

    # Return best found
    return best_routes

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

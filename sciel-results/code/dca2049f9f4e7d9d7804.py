"""EV routing with time windows – Clarke-Wright Construction + Enhanced Local Search."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def _build_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        xi, yi = coords[i][0], coords[i][1]
        for j in range(i + 1, n):
            d = math.hypot(xi - coords[j][0], yi - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D

def _simulate(route, D, instance, strict=True):
    """
    Simulate a single route. 
    Returns: (total_distance, valid, end_time, end_battery)
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = instance["stations_set"]
    
    t = 0.0
    batt = battery_max
    load = 0
    dist_total = 0.0
    prev = 0

    for node in route:
        d = D[prev, node]
        dist_total += d
        t += d
        batt -= d
        
        if batt < -1e-9:
            return dist_total, False, t, batt

        if node in stations_set:
            energy_needed = battery_max - batt
            t += energy_needed * recharge_rate
            batt = battery_max
        else:
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return dist_total, False, t, batt
            
            load += demands[node]
            if load > capacity + 1e-9:
                return dist_total, False, t, batt
            
            t += service[node]
        prev = node

    d = D[prev, 0]
    dist_total += d
    t += d
    batt -= d

    if batt < -1e-9:
        return dist_total, False, t, batt
    if t > horizon + 1e-9:
        return dist_total, False, t, batt

    return dist_total, True, t, batt

def _simulate_fast(route, D, instance):
    """
    Fast simulation for local search - returns only distance and validity.
    Uses precomputed arrays for speed.
    """
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = instance["stations_set"]
    
    t = 0.0
    batt = battery_max
    load = 0
    dist_total = 0.0
    prev = 0

    for node in route:
        d = D[prev, node]
        dist_total += d
        t += d
        batt -= d
        
        if batt < 0:
            return dist_total, False

        if node in stations_set:
            energy_needed = battery_max - batt
            t += energy_needed * recharge_rate
            batt = battery_max
        else:
            if t < ready[node]:
                t = ready[node]
            if t > due[node]:
                return dist_total, False
            
            load += demands[node]
            if load > capacity:
                return dist_total, False
            
            t += service[node]
        prev = node

    d = D[prev, 0]
    dist_total += d
    t += d
    batt -= d

    if batt < 0:
        return dist_total, False
    if t > horizon:
        return dist_total, False

    return dist_total, True

def _total_distance(routes, D):
    total = 0.0
    for route in routes:
        if not route:
            continue
        prev = 0
        for node in route:
            total += D[prev, node]
            prev = node
        total += D[prev, 0]
    return total

def _is_valid_solution(routes, instance, D):
    served = set()
    stations_set = instance["stations_set"]
    n_customers = instance["n_customers"]
    
    # Check coverage and uniqueness
    for route in routes:
        for node in route:
            if node in stations_set:
                continue
            if node in served:
                return False
            served.add(node)
            
    if served != set(range(1, n_customers + 1)):
        return False

    # Check constraints
    for route in routes:
        _, valid, _, _ = _simulate(route, D, instance)
        if not valid:
            return False
    return True

def _clarke_wright_savings(instance, D):
    """
    Clarke-Wright savings heuristic to construct initial routes.
    """
    demands = instance["demands"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    n_customers = instance["n_customers"]
    stations_set = instance["stations_set"]

    # Initialize: each customer in its own route
    # route_nodes[route_id] = list of nodes
    # route_load[route_id] = total demand
    # route_endpoints[route_id] = set of end nodes (customers only)
    
    routes = {}
    for i in range(1, n_customers + 1):
        routes[i] = {
            'nodes': [i],
            'load': demands[i],
            'ends': {i}
        }
    
    # Compute savings
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = D[0, i] + D[0, j] - D[i, j]
            if s > 0:
                savings.append((s, i, j))
    savings.sort(reverse=True)

    # Process savings
    for s, i, j in savings:
        # Find which routes i and j belong to
        ri = None
        rj = None
        for rid, rdata in routes.items():
            if i in rdata['ends']:
                ri = rid
            if j in rdata['ends']:
                rj = rid
            if ri is not None and rj is not None:
                break
        
        if ri is None or rj is None or ri == rj:
            continue
            
        # Check capacity
        if routes[ri]['load'] + routes[rj]['load'] > capacity:
            continue

        # Determine merge direction
        # We need i and j to be endpoints of their respective routes
        # Let's find the other endpoints
        other_i = [n for n in routes[ri]['ends'] if n != i]
        other_j = [n for n in routes[rj]['ends'] if n != j]
        
        # Since it's a path, there should be exactly one other endpoint if len>1, 
        # or none if len==1 (but ends is a set, so for single node, ends={node})
        # Wait, for a single node [i], ends={i}. 
        # If we merge, we remove i from ends of ri, and j from ends of rj.
        
        # Let's just try the merge and validate.
        # We need to construct the new node list.
        # Route A: [a_start, ..., i] or [i, ..., a_end]
        # Route B: [b_start, ..., j] or [j, ..., b_end]
        
        # To keep it simple and robust, we'll just concatenate and let the local search fix the order?
        # No, CW requires specific endpoint connection.
        
        # Let's retrieve the actual lists.
        # We need to know if i is at the start or end of routes[ri]['nodes'].
        # This is tricky with just 'ends'. Let's store the full list and check index.
        
        nodes_i = routes[ri]['nodes']
        nodes_j = routes[rj]['nodes']
        
        # Check if i is at start or end of nodes_i
        # And j is at start or end of nodes_j
        
        valid_merge = False
        new_nodes = []
        
        # Case 1: i is last of nodes_i, j is first of nodes_j
        if nodes_i[-1] == i and nodes_j[0] == j:
            new_nodes = nodes_i + nodes_j
            valid_merge = True
        # Case 2: i is first of nodes_i, j is last of nodes_j
        elif nodes_i[0] == i and nodes_j[-1] == j:
            new_nodes = nodes_j + nodes_i
            valid_merge = True
        # Case 3: i is last of nodes_i, j is last of nodes_j
        elif nodes_i[-1] == i and nodes_j[-1] == j:
            new_nodes = nodes_i + nodes_j[::-1]
            valid_merge = True
        # Case 4: i is first of nodes_i, j is first of nodes_j
        elif nodes_i[0] == i and nodes_j[0] == j:
            new_nodes = nodes_i[::-1] + nodes_j
            valid_merge = True
            
        if not valid_merge:
            continue
            
        # Validate the merged route (feasibility)
        # Note: We don't have stations in the initial CW route. 
        # The merged route might be infeasible due to battery/time.
        # We'll try to simulate it. If infeasible, we might still keep it and let _fix_route handle it,
        # or we reject it. Rejecting is safer for CW.
        
        dist, valid, _, _ = _simulate(new_nodes, D, instance)
        if not valid:
            # Try to fix by inserting stations? 
            # For initial construction, let's just reject to maintain quality, 
            # or accept and repair later. 
            # Given the directive "fundamentally shifting... to global structural optimization", 
            # we should probably accept if it's close, but strict validity is key.
            # Let's try a quick repair: insert stations if battery fails.
            repaired = _try_fix_route(new_nodes, D, instance)
            if repaired is not None:
                new_nodes = repaired
                dist, valid, _, _ = _simulate(new_nodes, D, instance)
                
        if valid:
            # Merge
            routes[ri]['nodes'] = new_nodes
            routes[ri]['load'] += routes[rj]['load']
            
            # Update ends
            new_ends = set()
            if len(new_nodes) > 1:
                new_ends.add(new_nodes[0])
                new_ends.add(new_nodes[-1])
            else:
                new_ends.add(new_nodes[0])
            routes[ri]['ends'] = new_ends
            
            del routes[rj]

    # Convert back to list of lists
    final_routes = [r['nodes'] for r in routes.values()]
    return final_routes

def _try_fix_route(route, D, instance, max_iters=10):
    """
    Attempt to fix a route by inserting stations.
    Returns fixed route or None.
    """
    stations = instance["stations"]
    current_route = list(route)
    
    for _ in range(max_iters):
        dist, valid, t, batt = _simulate(current_route, D, instance)
        if valid:
            return current_route
            
        # Find where it fails
        # Simple heuristic: insert station at the point of maximum battery deficit or time violation
        # For simplicity, try inserting each station at each position and pick the best valid one?
        # Too slow. 
        # Let's find the first infeasibility.
        
        t_sim = 0.0
        batt_sim = instance["battery"]
        prev = 0
        fail_idx = -1
        
        for i, node in enumerate(current_route):
            d = D[prev, node]
            t_sim += d
            batt_sim -= d
            if batt_sim < -1e-9:
                fail_idx = i
                break
            if node not in instance["stations_set"]:
                if t_sim < instance["ready"][node]:
                    t_sim = instance["ready"][node]
                if t_sim > instance["due"][node] + 1e-9:
                    fail_idx = i
                    break
                t_sim += instance["service"][node]
            prev = node
            
        if fail_idx == -1:
             # Failed on return leg
             # Try inserting station at the end
             for st in stations:
                 new_route = current_route + [st]
                 dist, valid, _, _ = _simulate(new_route, D, instance)
                 if valid:
                     return new_route
             return None
             
        # Try inserting a station before the failing node
        best_new_route = None
        best_dist = float('inf')
        
        for st in stations:
             # Insert at fail_idx
             new_route = current_route[:fail_idx] + [st] + current_route[fail_idx:]
             dist, valid, _, _ = _simulate(new_route, D, instance)
             if valid and dist < best_dist:
                 best_dist = dist
                 best_new_route = new_route
                 
        if best_new_route:
            current_route = best_new_route
        else:
            # Try inserting after the failing node? Or earlier?
            # Let's just fail if we can't fix it easily.
            return None
            
    return current_route if _simulate(current_route, D, instance)[1] else None

def _2opt(route, D, instance, time_limit):
    """2-opt improvement for a single route with delta evaluation."""
    current_dist, valid, _, _ = _simulate(route, D, instance)
    if not valid:
        return route

    n = len(route)
    if n < 2:
        return route

    improved = True
    while improved:
        if time.time() > time_limit:
            break
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if time.time() > time_limit:
                    return route
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                dist, valid = _simulate_fast(new_route, D, instance)
                if valid and dist < current_dist - 1e-9:
                    route = new_route
                    current_dist = dist
                    improved = True
                    break
            if improved:
                break

    return route

def _or_opt(route, D, instance, time_limit):
    """Or-opt: relocate 1 or 2 consecutive nodes."""
    current_dist, valid, _, _ = _simulate(route, D, instance)
    if not valid:
        return route

    n = len(route)
    if n < 2:
        return route

    improved = True
    while improved:
        if time.time() > time_limit:
            break
        improved = False
        for i in range(n):
            for length in [1, 2]:
                if i + length > n:
                    continue
                segment = route[i:i + length]
                rest = route[:i] + route[i + length:]
                if not rest:
                    continue
                for j in range(len(rest) + 1):
                    if time.time() > time_limit:
                        return route
                    new_route = rest[:j] + segment + rest[j:]
                    dist, valid = _simulate_fast(new_route, D, instance)
                    if valid and dist < current_dist - 1e-9:
                        route = new_route
                        current_dist = dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return route

def _station_opt(route, D, instance, time_limit):
    """
    Specifically try to move or insert stations to improve distance or feasibility.
    Uses battery-aware placement heuristics.
    """
    current_dist, valid, _, _ = _simulate(route, D, instance)
    
    if not valid:
        fixed = _try_fix_route(route, D, instance)
        if fixed is not None:
            route = fixed
            current_dist, valid, _, _ = _simulate(route, D, instance)
            if not valid:
                return route
        else:
            return route

    n = len(route)
    if n == 0:
        return route

    improved = True
    while improved:
        if time.time() > time_limit:
            break
        improved = False
        
        # 1. Try moving existing stations to better positions
        for i in range(n):
            if route[i] not in instance["stations_set"]:
                continue
            st = route[i]
            rest = route[:i] + route[i+1:]
            best_pos = -1
            best_dist = current_dist
            for j in range(len(rest) + 1):
                if time.time() > time_limit:
                    break
                new_route = rest[:j] + [st] + rest[j:]
                dist, valid = _simulate_fast(new_route, D, instance)
                if valid and dist < best_dist - 1e-9:
                    best_dist = dist
                    best_pos = j
            if best_pos >= 0:
                route = rest[:best_pos] + [st] + rest[best_pos:]
                current_dist = best_dist
                improved = True
                break
                
        # 2. Try inserting a new station at the point of maximum battery deficit
        if not improved:
            # Find where battery is lowest
            t = 0.0
            batt = instance["battery"]
            prev = 0
            min_batt = batt
            min_idx = -1
            for idx, node in enumerate(route):
                d = D[prev, node]
                batt -= d
                if batt < min_batt:
                    min_batt = batt
                    min_idx = idx
                if node in instance["stations_set"]:
                    batt = instance["battery"]
                else:
                    if t < instance["ready"][node]:
                        t = instance["ready"][node]
                    t += d + instance["service"][node]
                prev = node
            
            if min_idx >= 0 and min_batt < instance["battery"] * 0.5:
                # Try inserting station near the minimum battery point
                for st in instance["stations"]:
                    if st in route:
                        continue
                    # Try inserting around min_idx
                    for offset in [0, -1, 1, -2, 2]:
                        pos = min_idx + offset
                        if pos < 0 or pos > len(route):
                            continue
                        new_route = route[:pos] + [st] + route[pos:]
                        dist, valid = _simulate_fast(new_route, D, instance)
                        if valid and dist < current_dist - 1e-9:
                            route = new_route
                            current_dist = dist
                            improved = True
                            break
                    if improved:
                        break

    return route

def _cross_route_swap(routes, D, instance, time_limit):
    """Inter-route improvement: swap customers between routes."""
    n_routes = len(routes)
    if n_routes < 2:
        return routes

    improved = True
    while improved:
        if time.time() > time_limit:
            break
        improved = False
        for a in range(n_routes):
            for b in range(a + 1, n_routes):
                for i in range(len(routes[a])):
                    for j in range(len(routes[b])):
                        if time.time() > time_limit:
                            return routes
                        if routes[a][i] in instance["stations_set"] or routes[b][j] in instance["stations_set"]:
                            continue
                            
                        ra = list(routes[a])
                        rb = list(routes[b])
                        ra[i], rb[j] = rb[j], ra[i]
                        da, va = _simulate_fast(ra, D, instance)
                        db, vb = _simulate_fast(rb, D, instance)
                        if va and vb:
                            old_dist = _total_distance([ra, rb], D)
                            new_dist = da + db
                            if new_dist < old_dist - 1e-9:
                                routes[a] = ra
                                routes[b] = rb
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return routes

def _polish_routes(routes, D, instance, time_limit):
    """
    High-precision polish pass with tighter tolerances and more iterations.
    """
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for i in range(len(routes)):
            # Multiple 2-opt passes
            for _ in range(3):
                new_route = _2opt(routes[i], D, instance, time.time() + 0.5)
                if new_route != routes[i]:
                    routes[i] = new_route
                    improved = True
                    break
            
            # Or-opt
            new_route = _or_opt(routes[i], D, instance, time.time() + 0.5)
            if new_route != routes[i]:
                routes[i] = new_route
                improved = True
            
            # Station optimization
            new_route = _station_opt(routes[i], D, instance, time.time() + 0.5)
            if new_route != routes[i]:
                routes[i] = new_route
                improved = True

    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    coords = instance["coords"]
    D = _build_distance_matrix(coords)
    instance["stations_set"] = set(instance["stations"])

    # 1. Construct Initial Solution using Clarke-Wright
    routes = _clarke_wright_savings(instance, D)
    
    # 2. Fix any infeasible routes from construction
    fixed_routes = []
    for r in routes:
        fixed = _try_fix_route(r, D, instance)
        if fixed is not None:
            fixed_routes.append(fixed)
        else:
            for c in r:
                if c not in instance["stations_set"]:
                    fixed_routes.append([c])
    
    best_routes = fixed_routes
    
    # Ensure all customers are covered
    served = set()
    for r in best_routes:
        for n in r:
            if n not in instance["stations_set"]:
                served.add(n)
    missing = set(range(1, instance["n_customers"] + 1)) - served
    for m in missing:
        best_routes.append([m])

    # 3. Local Search
    passes = 0
    max_passes = 80
    
    while time.time() < deadline and passes < max_passes:
        passes += 1
        improved = False
        remaining_time = deadline - time.time()
        current_time_limit = time.time() + remaining_time / (max_passes - passes + 1)
        
        # Intra-route optimizations
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
                
            # 2-opt
            new_route = _2opt(best_routes[i], D, instance, current_time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True
                
            # Or-opt
            new_route = _or_opt(best_routes[i], D, instance, current_time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True
                
            # Station Optimization
            new_route = _station_opt(best_routes[i], D, instance, current_time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True

        # Inter-route optimization
        if passes % 3 == 0:
            if time.time() < deadline:
                new_routes = _cross_route_swap(best_routes, D, instance, current_time_limit)
                if new_routes != best_routes:
                    best_routes = new_routes
                    improved = True

        if not improved:
            break

    # 4. High-precision polish pass
    if time.time() < deadline - 5.0:
        best_routes = _polish_routes(best_routes, D, instance, deadline - 3.0)

    # 5. Final Validation
    if not _is_valid_solution(best_routes, instance, D):
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[c] for c in customers]

    return best_routes
# EVOLVE-BLOCK-END

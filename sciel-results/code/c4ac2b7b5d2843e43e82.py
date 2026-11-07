"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _precompute_dist_matrix(coords):
    """Precompute pairwise distance matrix."""
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix

def _simulate_route(route, instance, dist_matrix):
    """
    Simulate a route and return (feasible, total_distance, battery_end, time_end).
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    total_demand = sum(demands[v] for v in route if v not in stations)
    if total_demand > capacity:
        return False, 0.0, 0.0, 0.0
    
    battery = battery_max
    time = 0.0
    prev = 0
    total_dist = 0.0
    
    for v in route:
        d = dist_matrix[prev][v]
        battery -= d
        if battery < -1e-9:
            return False, 0.0, 0.0, 0.0
        time += d / speed
        total_dist += d
        
        if v not in stations:
            time = max(time, ready[v])
            if time > due[v] + 1e-9:
                return False, 0.0, 0.0, 0.0
            time += service[v]
        else:
            energy = battery_max - battery
            time += energy * recharge_rate
            battery = battery_max
        prev = v
    
    d = dist_matrix[prev][0]
    battery -= d
    if battery < -1e-9:
        return False, 0.0, 0.0, 0.0
    time += d / speed
    total_dist += d
    
    if time > horizon + 1e-9:
        return False, 0.0, 0.0, 0.0
    
    return True, total_dist, battery, time

def _construct_savings_routes(instance, dist_matrix):
    """Construct initial routes using Clarke-Wright Savings."""
    coords = instance["coords"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    
    customers = list(range(1, n_customers + 1))
    
    # Initialize each customer as a separate route
    routes = [[c] for c in customers]
    route_index = {c: i for i, c in enumerate(customers)}
    route_demands = [demands[c] for c in customers]
    
    # Calculate savings for merging routes i and j
    # Savings = dist(0, i) + dist(0, j) - dist(i, j)
    # We need to check if they are at the ends of their routes and if merging is feasible.
    
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            c1 = customers[i]
            c2 = customers[j]
            s = dist_matrix[0][c1] + dist_matrix[0][c2] - dist_matrix[c1][c2]
            savings.append((s, i, j))
    
    savings.sort(reverse=True)
    
    for s, i, j in savings:
        if s <= 0:
            break
        
        # Check if i and j are still in separate routes
        if route_index[customers[i]] == route_index[customers[j]]:
            continue
        
        # Check capacity
        if route_demands[i] + route_demands[j] > capacity:
            continue
        
        # Identify the routes
        r1 = routes[route_index[customers[i]]]
        r2 = routes[route_index[customers[j]]]
        
        # Determine connection points
        # We can connect:
        # 1. End of r1 to Start of r2
        # 2. Start of r1 to End of r2
        # 3. End of r1 to End of r2 (reverse r2)
        # 4. Start of r1 to Start of r2 (reverse r2)
        
        # Since we are merging, we just need to find the best valid merge that maintains feasibility.
        # For simplicity and speed in initial construction, we try merging ends.
        # Let's assume we merge r1 and r2.
        # We need to check feasibility of the new combined route.
        
        # Try merging end of r1 with start of r2
        new_route = r1 + r2
        feas, _, _, _ = _simulate_route(new_route, instance, dist_matrix)
        if feas:
            # Merge
            routes[route_index[customers[i]]] = new_route
            routes[route_index[customers[j]]] = []
            route_demands[route_index[customers[i]]] += route_demands[route_index[customers[j]]]
            # Update indices for remaining customers in r2
            for c in r2:
                route_index[c] = route_index[customers[i]]
            continue
        
        # Try merging end of r1 with end of r2 (reverse r2)
        new_route = r1 + r2[::-1]
        feas, _, _, _ = _simulate_route(new_route, instance, dist_matrix)
        if feas:
            routes[route_index[customers[i]]] = new_route
            routes[route_index[customers[j]]] = []
            route_demands[route_index[customers[i]]] += route_demands[route_index[customers[j]]]
            for c in r2:
                route_index[c] = route_index[customers[i]]
            continue
            
        # Try merging start of r1 with start of r2 (reverse r1)
        new_route = r1[::-1] + r2
        feas, _, _, _ = _simulate_route(new_route, instance, dist_matrix)
        if feas:
            routes[route_index[customers[i]]] = new_route
            routes[route_index[customers[j]]] = []
            route_demands[route_index[customers[i]]] += route_demands[route_index[customers[j]]]
            for c in r1:
                route_index[c] = route_index[customers[i]]
            continue
            
        # Try merging start of r1 with end of r2 (reverse r1)
        new_route = r1[::-1] + r2[::-1]
        feas, _, _, _ = _simulate_route(new_route, instance, dist_matrix)
        if feas:
            routes[route_index[customers[i]]] = new_route
            routes[route_index[customers[j]]] = []
            route_demands[route_index[customers[i]]] += route_demands[route_index[customers[j]]]
            for c in r1:
                route_index[c] = route_index[customers[i]]
            continue

    return [r for r in routes if r]

def _two_opt(route, instance, dist_matrix, max_iter=10):
    """Apply 2-opt within a route."""
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if new_route == route:
                    continue
                feas, dist, _, _ = _simulate_route(new_route, instance, dist_matrix)
                if feas:
                    # Compare distance
                    _, old_dist, _, _ = _simulate_route(route, instance, dist_matrix)
                    if dist < old_dist - 1e-9:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    return route

def _or_opt(route, instance, dist_matrix, max_iter=10):
    """Apply Or-opt moves (move segment of length 1, 2, or 3)."""
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for seg_len in [1, 2, 3]:
            for i in range(len(route) - seg_len + 1):
                segment = route[i:i+seg_len]
                # Remove segment
                remainder = route[:i] + route[i+seg_len:]
                if not remainder:
                    continue
                # Try inserting segment at all positions in remainder
                for j in range(len(remainder) + 1):
                    new_route = remainder[:j] + segment + remainder[j:]
                    if new_route == route:
                        continue
                    feas, dist, _, _ = _simulate_route(new_route, instance, dist_matrix)
                    if feas:
                        _, old_dist, _, _ = _simulate_route(route, instance, dist_matrix)
                        if dist < old_dist - 1e-9:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return route

def _swap_routes(routes, instance, dist_matrix, max_iter=10):
    """Swap customers between routes to improve total distance."""
    coords = instance["coords"]
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for r1 in range(len(routes)):
            for r2 in range(len(routes)):
                if r1 == r2:
                    continue
                for i in range(len(routes[r1])):
                    v1 = routes[r1][i]
                    for j in range(len(routes[r2])):
                        v2 = routes[r2][j]
                        # Try swapping v1 and v2
                        new_r1 = routes[r1][:]
                        new_r2 = routes[r2][:]
                        new_r1[i] = v2
                        new_r2[j] = v1
                        
                        feas1, dist1, _, _ = _simulate_route(new_r1, instance, dist_matrix)
                        feas2, dist2, _, _ = _simulate_route(new_r2, instance, dist_matrix)
                        
                        if feas1 and feas2:
                            old_dist1, _, _, _ = _simulate_route(routes[r1], instance, dist_matrix)
                            old_dist2, _, _, _ = _simulate_route(routes[r2], instance, dist_matrix)
                            if (dist1 + dist2) < (old_dist1 + old_dist2) - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return routes

def plan(instance):
    coords = np.array(instance["coords"], dtype=float)
    dist_matrix = _precompute_dist_matrix(coords)
    
    # Build initial routes
    routes = _construct_savings_routes(instance, dist_matrix)
    
    # Ensure all customers are served
    served = set()
    for r in routes:
        for v in r:
            if v not in set(instance["stations"]):
                served.add(v)
    missing = set(range(1, instance["n_customers"] + 1)) - served
    for v in missing:
        routes.append([v])
    
    # Local search
    for _ in range(3): # Multiple passes
        for i in range(len(routes)):
            routes[i] = _two_opt(routes[i], instance, dist_matrix)
            routes[i] = _or_opt(routes[i], instance, dist_matrix)
        
        routes = _swap_routes(routes, instance, dist_matrix)
        
        # Remove empty routes
        routes = [r for r in routes if r]
        
        # If no routes, break
        if not routes:
            break
    
    # Verify all customers served exactly once
    all_nodes = []
    for r in routes:
        all_nodes.extend(r)
    customer_nodes = [v for v in all_nodes if v not in set(instance["stations"])]
    if sorted(customer_nodes) != list(range(1, instance["n_customers"] + 1)):
        # Fallback: one per route
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]
    
    return routes
# EVOLVE-BLOCK-END

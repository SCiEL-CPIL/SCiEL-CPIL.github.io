"""Sweep Angle Initialization with Iterated Local Search refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def _euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def _check_route(coords, demands, ready, due, service, route, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """Check if a route is feasible. Returns (feasible, total_dist)."""
    if not route:
        return True, 0.0
    
    total_demand = sum(demands[n] for n in route)
    if total_demand > capacity:
        return False, 0.0
    
    time = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0
    
    for node in route:
        d = _euclidean(coords[prev], coords[node])
        total_dist += d
        time += d / speed
        battery -= d
        
        if battery < -1e-9:
            return False, 0.0
        
        if node in stations:
            energy_needed = battery_max - battery
            if energy_needed > 1e-9:
                time += energy_needed * recharge_rate
            battery = battery_max
        else:
            if time < ready[node]:
                time = ready[node]
            if time > due[node] + 1e-9:
                return False, 0.0
            time += service[node]
        
        prev = node
    
    d = _euclidean(coords[prev], coords[0])
    total_dist += d
    time += d / speed
    battery -= d
    
    if battery < -1e-9:
        return False, 0.0
    if time > horizon + 1e-9:
        return False, 0.0
    
    return True, total_dist

def _route_distance(coords, route):
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for node in route:
        total += _euclidean(coords[prev], coords[node])
        prev = node
    total += _euclidean(coords[prev], coords[0])
    return total

def _sweep_partition(coords, customers, n_clusters, depot_idx=0):
    """Partition customers into n_clusters based on sweep angle from depot."""
    if not customers:
        return []
    
    # Calculate angles
    angles = []
    for c in customers:
        dx = coords[c][0] - coords[depot_idx][0]
        dy = coords[c][1] - coords[depot_idx][1]
        angle = math.atan2(dy, dx)
        # Normalize to [0, 2*pi)
        if angle < 0:
            angle += 2 * math.pi
        angles.append((angle, c))
    
    angles.sort()
    
    # Split into n_clusters segments
    # To handle the wrap-around of angles (0 and 2pi are close), we might need to rotate the start point
    # For simplicity, we just split the sorted list into n equal parts by count, 
    # but we should try to minimize the "cut" distance. 
    # A better way: try n different start rotations? 
    # Given N=25, we can just do a simple equal split first, then refine.
    
    n = len(angles)
    cluster_sizes = [n // n_clusters] * n_clusters
    remainder = n % n_clusters
    for i in range(remainder):
        cluster_sizes[i] += 1
        
    clusters = []
    start = 0
    for i in range(n_clusters):
        end = start + cluster_sizes[i]
        cluster_nodes = [c for _, c in angles[start:end]]
        clusters.append(cluster_nodes)
        start = end
        
    return clusters

def _build_route_greedy(coords, demands, ready, due, service, cluster_nodes, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """Build a route using greedy insertion."""
    if not cluster_nodes:
        return []
    
    route = []
    remaining = set(cluster_nodes)
    current_pos = 0
    current_time = 0.0
    current_battery = battery_max
    current_demand = 0
    
    while remaining:
        best_node = None
        best_cost = float('inf')
        
        for node in remaining:
            d = _euclidean(coords[current_pos], coords[node])
            new_time = current_time + d / speed
            new_battery = current_battery - d
            
            if new_battery < -1e-9:
                continue
            
            if node not in stations:
                if new_time < ready[node]:
                    new_time = ready[node]
                if new_time > due[node] + 1e-9:
                    continue
                if current_demand + demands[node] > capacity:
                    continue
            
            # Cost: distance + penalty for waiting? Just distance for now
            cost = d
            if cost < best_cost:
                best_cost = cost
                best_node = node
        
        if best_node is None:
            # Try to find any feasible node
            for node in remaining:
                d = _euclidean(coords[current_pos], coords[node])
                new_time = current_time + d / speed
                new_battery = current_battery - d
                if new_battery >= -1e-9:
                    if node not in stations:
                        if new_time < ready[node]:
                            new_time = ready[node]
                        if new_time <= due[node] + 1e-9 and current_demand + demands[node] <= capacity:
                            best_node = node
                            break
            if best_node is None:
                break
        
        # Move to best_node
        d = _euclidean(coords[current_pos], coords[best_node])
        current_time += d / speed
        current_battery -= d
        route.append(best_node)
        remaining.remove(best_node)
        
        if best_node not in stations:
            if current_time < ready[best_node]:
                current_time = ready[best_node]
            current_time += service[best_node]
            current_demand += demands[best_node]
        else:
            energy_needed = battery_max - current_battery
            if energy_needed > 1e-9:
                current_time += energy_needed * recharge_rate
            current_battery = battery_max
        
        current_pos = best_node
    
    # Check if route is feasible
    feasible, _ = _check_route(coords, demands, ready, due, service, route,
                               capacity, battery_max, speed, recharge_rate, horizon, stations)
    return route if feasible else []

def _local_search(coords, demands, ready, due, service, routes, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """Perform local search with 2-opt and Or-opt."""
    current_routes = [r[:] for r in routes]
    current_dist = sum(_route_distance(coords, r) for r in current_routes)
    
    improved = True
    max_iterations = 1000
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        
        # 2-opt within each route
        for r_idx in range(len(current_routes)):
            route = current_routes[r_idx]
            n = len(route)
            if n < 2:
                continue
            
            found_improvement = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, new_dist = _check_route(coords, demands, ready, due, service, new_route,
                                                      capacity, battery_max, speed, recharge_rate, horizon, stations)
                    if feasible:
                        old_dist = _route_distance(coords, route)
                        if new_dist < old_dist - 1e-9:
                            current_routes[r_idx] = new_route
                            current_dist += new_dist - old_dist
                            improved = True
                            found_improvement = True
                            break
                if found_improvement:
                    break
        
        # Or-opt: move a segment of length 1, 2, or 3 within the same route
        for r_idx in range(len(current_routes)):
            route = current_routes[r_idx]
            n = len(route)
            if n < 2:
                continue
            
            found_improvement = False
            for seg_len in [1, 2, 3]:
                for i in range(n - seg_len + 1):
                    segment = route[i:i+seg_len]
                    rest = route[:i] + route[i+seg_len:]
                    
                    for j in range(len(rest) + 1):
                        new_route = rest[:j] + segment + rest[j:]
                        feasible, new_dist = _check_route(coords, demands, ready, due, service, new_route,
                                                          capacity, battery_max, speed, recharge_rate, horizon, stations)
                        if feasible:
                            old_dist = _route_distance(coords, route)
                            if new_dist < old_dist - 1e-9:
                                current_routes[r_idx] = new_route
                                current_dist += new_dist - old_dist
                                improved = True
                                found_improvement = True
                                break
                    if found_improvement:
                        break
                if found_improvement:
                    break
        
        # Relocate between routes
        for r1 in range(len(current_routes)):
            for r2 in range(len(current_routes)):
                if r1 == r2:
                    continue
                route1 = current_routes[r1]
                route2 = current_routes[r2]
                
                found_improvement = False
                for pos1 in range(len(route1)):
                    node = route1[pos1]
                    source = route1[:pos1] + route1[pos1+1:]
                    
                    for pos2 in range(len(route2) + 1):
                        target = route2[:pos2] + [node] + route2[pos2:]
                        
                        feas1, dist1 = _check_route(coords, demands, ready, due, service, source,
                                                    capacity, battery_max, speed, recharge_rate, horizon, stations)
                        feas2, dist2 = _check_route(coords, demands, ready, due, service, target,
                                                    capacity, battery_max, speed, recharge_rate, horizon, stations)
                        
                        if feas1 and feas2:
                            old_dist = _route_distance(coords, route1) + _route_distance(coords, route2)
                            new_dist = dist1 + dist2
                            if new_dist < old_dist - 1e-9:
                                current_routes[r1] = source
                                current_routes[r2] = target
                                current_dist += new_dist - old_dist
                                improved = True
                                found_improvement = True
                                break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            
            if found_improvement:
                break
        
        # Remove empty routes
        current_routes = [r for r in current_routes if r]
        
    return current_routes

def _perturb(routes, strength=0.2):
    """Perturb the solution by moving a segment or swapping nodes."""
    if not routes:
        return routes
    
    new_routes = [r[:] for r in routes]
    
    # Choose a random perturbation type
    p_type = random.random()
    
    if p_type < 0.5:
        # Or-opt perturbation
        r_idx = random.randint(0, len(new_routes) - 1)
        route = new_routes[r_idx]
        if len(route) < 2:
            return new_routes
        
        seg_len = random.randint(1, min(3, len(route)))
        start = random.randint(0, len(route) - seg_len)
        end = start + seg_len - 1
        
        segment = route[start:end+1]
        new_route = route[:start] + route[end+1:]
        
        target_r_idx = random.randint(0, len(new_routes) - 1)
        target_route = new_routes[target_r_idx]
        insert_pos = random.randint(0, len(target_route))
        new_target = target_route[:insert_pos] + segment + target_route[insert_pos:]
        
        new_routes[r_idx] = new_route
        new_routes[target_r_idx] = new_target
        
    else:
        # Swap perturbation
        if len(new_routes) >= 2:
            r1 = random.randint(0, len(new_routes) - 1)
            r2 = random.randint(0, len(new_routes) - 1)
            if r1 == r2:
                return new_routes
            
            route1 = new_routes[r1]
            route2 = new_routes[r2]
            
            if route1 and route2:
                pos1 = random.randint(0, len(route1) - 1)
                pos2 = random.randint(0, len(route2) - 1)
                
                # Swap nodes
                node1 = route1[pos1]
                node2 = route2[pos2]
                
                route1[pos1] = node2
                route2[pos2] = node1
        
        else:
            # If only one route, do an internal swap
            route = new_routes[0]
            if len(route) >= 2:
                pos1 = random.randint(0, len(route) - 1)
                pos2 = random.randint(0, len(route) - 1)
                route[pos1], route[pos2] = route[pos2], route[pos1]
    
    # Remove empty routes
    new_routes = [r for r in new_routes if r]
    
    return new_routes

def _iterated_local_search(coords, demands, ready, due, service, routes, capacity, battery_max, speed, recharge_rate, horizon, stations, n_iter=100):
    """Iterated Local Search."""
    current_routes = _local_search(coords, demands, ready, due, service, routes, capacity, battery_max, speed, recharge_rate, horizon, stations)
    current_dist = sum(_route_distance(coords, r) for r in current_routes)
    
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist
    
    all_customers = set()
    for r in current_routes:
        for n in r:
            if n not in stations:
                all_customers.add(n)
    
    for _ in range(n_iter):
        perturbed = _perturb(current_routes)
        
        served = set()
        valid = True
        for r in perturbed:
            for n in r:
                if n not in stations:
                    if n in served:
                        valid = False
                        break
                    served.add(n)
            if not valid:
                break
        
        if not valid or served != all_customers:
            continue
            
        feasible = True
        for r in perturbed:
            f, _ = _check_route(coords, demands, ready, due, service, r, capacity, battery_max, speed, recharge_rate, horizon, stations)
            if not f:
                feasible = False
                break
        
        if not feasible:
            continue
            
        new_routes = _local_search(coords, demands, ready, due, service, perturbed, capacity, battery_max, speed, recharge_rate, horizon, stations)
        new_dist = sum(_route_distance(coords, r) for r in new_routes)
        
        if new_dist < current_dist - 1e-9:
            current_routes = new_routes
            current_dist = new_dist
            if current_dist < best_dist:
                best_dist = current_dist
                best_routes = [r[:] for r in current_routes]
        
    return best_routes

def plan(instance):
    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    customers = list(range(1, n_customers + 1))
    
    total_demand = sum(demands[c] for c in customers)
    n_clusters = max(1, int(np.ceil(total_demand / capacity)))
    n_clusters = min(n_clusters, len(customers))
    
    clusters = _sweep_partition(coords, customers, n_clusters)
    
    routes = []
    for cluster in clusters:
        route = _build_route_greedy(coords, demands, ready, due, service, cluster,
                                    capacity, battery_max, speed, recharge_rate, horizon, stations)
        if route:
            routes.append(route)
    
    served = set()
    for r in routes:
        for n in r:
            if n not in stations:
                served.add(n)
    
    unserved = [c for c in customers if c not in served]
    if unserved:
        for c in unserved:
            inserted = False
            for r_idx in range(len(routes)):
                for pos in range(len(routes[r_idx]) + 1):
                    new_route = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
                    feasible, _ = _check_route(coords, demands, ready, due, service, new_route,
                                               capacity, battery_max, speed, recharge_rate, horizon, stations)
                    if feasible:
                        routes[r_idx] = new_route
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                routes.append([c])
    
    routes = _iterated_local_search(coords, demands, ready, due, service, routes,
                                    capacity, battery_max, speed, recharge_rate, horizon, stations,
                                    n_iter=100)
    
    return routes
# EVOLVE-BLOCK-END

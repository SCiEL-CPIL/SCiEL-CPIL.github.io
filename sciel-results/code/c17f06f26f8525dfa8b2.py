"""Optimized EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
import time
import random
from heapq import heappush, heappop

DEADLINE_SECONDS = 140.0  # safety margin below 150s limit


def _build_dist_matrix(coords):
    """Precompute distance matrix using vectorized operations."""
    coords = np.array(coords, dtype=np.float64)
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    return dist_matrix


def _simulate_route(route, dist_matrix, instance, n_customers, stations,
                    battery_max, speed, recharge_rate, horizon, capacity):
    """Simulate a route. Returns (valid, distance, time_at_depot, battery_at_depot)."""
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    
    # Check demand
    total_demand = sum(demands[node] for node in route if node <= n_customers)
    if total_demand > capacity:
        return False, 0.0, 0.0, 0.0
    
    battery = battery_max
    time_at = 0.0
    distance = 0.0
    current_node = 0
    
    for node in route:
        d = dist_matrix[current_node, node]
        time_at += d / speed
        battery -= d
        distance += d
        
        if battery < -1e-9:
            return False, 0.0, 0.0, 0.0
        
        if node <= n_customers:
            time_at = max(time_at, ready[node])
            if time_at > due[node] + 1e-9:
                return False, 0.0, 0.0, 0.0
            time_at += service[node]
        elif node in stations:
            energy_needed = battery_max - battery
            time_at += energy_needed * recharge_rate
            battery = battery_max
        
        current_node = node
    
    d = dist_matrix[current_node, 0]
    time_at += d / speed
    battery -= d
    distance += d
    
    if battery < -1e-9:
        return False, 0.0, 0.0, 0.0
    if time_at > horizon + 1e-9:
        return False, 0.0, 0.0, 0.0
    
    return True, distance, time_at, battery


def _validate_solution(routes, instance):
    """Validate all routes. Returns (valid, total_distance)."""
    n_customers = instance["n_customers"]
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
    
    stations = set(instance["stations"])
    served = set()
    total_distance = 0.0
    
    for route in routes:
        valid, dist, _, _ = _simulate_route(route, dist_matrix, instance, n_customers, stations,
                                             instance["battery"], instance["speed"],
                                             instance["recharge_rate"], instance["horizon"],
                                             instance["capacity"])
        if not valid:
            return False, 0.0
        total_distance += dist
        
        for node in route:
            if node <= n_customers:
                if node in served:
                    return False, 0.0
                served.add(node)
    
    if served != set(range(1, n_customers + 1)):
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


def _check_route_feasibility(route, dist_matrix, instance, n_customers, stations,
                              battery_max, speed, recharge_rate, horizon, capacity):
    """Check if a route is feasible."""
    valid, _, _, _ = _simulate_route(route, dist_matrix, instance, n_customers, stations,
                                      battery_max, speed, recharge_rate, horizon, capacity)
    return valid


def _greedy_construct(instance):
    """Construct initial solution using greedy insertion."""
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
    stations = set(instance["stations"])
    
    dist_matrix = _build_dist_matrix(coords)
    
    # Sort customers by due time
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: due[v])
    
    routes = []
    route_loads = []
    
    for cust in customers:
        best_route_idx = -1
        best_insert_pos = -1
        best_cost = float('inf')
        
        for r_idx, route in enumerate(routes):
            if route_loads[r_idx] + demands[cust] > capacity:
                continue
            
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                
                if _check_route_feasibility(new_route, dist_matrix, instance,
                                             n_customers, stations, battery_max,
                                             speed, recharge_rate, horizon, capacity):
                    old_dist = _compute_total_distance([route], dist_matrix)
                    new_dist = _compute_total_distance([new_route], dist_matrix)
                    
                    cost = new_dist - old_dist
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_insert_pos = pos
        
        if best_route_idx == -1:
            routes.append([cust])
            route_loads.append(demands[cust])
        else:
            routes[best_route_idx].insert(best_insert_pos, cust)
            route_loads[best_route_idx] += demands[cust]
    
    return routes


def _two_opt_route(route, dist_matrix, instance, n_customers, stations,
                   battery_max, speed, recharge_rate, horizon, capacity,
                   max_iters=300):
    """Optimize route order using 2-opt."""
    if len(route) < 2:
        return route
    
    def is_feasible(r):
        return _check_route_feasibility(r, dist_matrix, instance, n_customers, stations,
                                         battery_max, speed, recharge_rate, horizon, capacity)
    
    def route_dist(r):
        return _compute_total_distance([r], dist_matrix)
    
    best_route = route[:]
    best_dist = route_dist(best_route)
    
    for _ in range(max_iters):
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                
                if is_feasible(new_route):
                    new_dist = route_dist(new_route)
                    if new_dist < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
        if not improved:
            break
    
    return best_route


def _local_search(routes, dist_matrix, instance, deadline):
    """Improve solution using intra-route 2-opt and inter-route moves."""
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)
    
    def is_feasible(route):
        return _check_route_feasibility(route, dist_matrix, instance, n_customers, stations,
                                         battery_max, speed, recharge_rate, horizon, capacity)
    
    def route_load(route):
        return sum(demands[node] for node in route if node <= n_customers)
    
    start_time = time.time()
    current_dist = get_total_dist(routes)
    
    # First pass: optimize each route's internal order
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _two_opt_route(
            routes[r_idx], dist_matrix, instance, n_customers, stations,
            battery_max, speed, recharge_rate, horizon, capacity, max_iters=150
        )
    
    current_dist = get_total_dist(routes)
    
    # Second pass: inter-route moves
    max_passes = 30
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try moving customers between routes
        for r1 in range(len(routes)):
            if time.time() > deadline:
                break
            for pos1 in range(len(routes[r1])):
                if time.time() > deadline:
                    break
                cust = routes[r1][pos1]
                if cust > n_customers:
                    continue
                
                for r2 in range(len(routes)):
                    if time.time() > deadline:
                        break
                    if r1 == r2:
                        continue
                    if route_load(routes[r2]) + demands[cust] > capacity:
                        continue
                    
                    for pos2 in range(len(routes[r2]) + 1):
                        if time.time() > deadline:
                            break
                        
                        route1_new = routes[r1][:pos1] + routes[r1][pos1+1:]
                        route2_new = routes[r2][:pos2] + [cust] + routes[r2][pos2:]
                        
                        if is_feasible(route1_new) and is_feasible(route2_new):
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
                break
        
        # Try swapping customers between routes
        if not improved and time.time() < deadline:
            for r1 in range(len(routes)):
                if time.time() > deadline:
                    break
                for pos1 in range(len(routes[r1])):
                    if time.time() > deadline:
                        break
                    cust1 = routes[r1][pos1]
                    if cust1 > n_customers:
                        continue
                    
                    for r2 in range(len(routes)):
                        if time.time() > deadline:
                            break
                        if r1 == r2:
                            continue
                        for pos2 in range(len(routes[r2])):
                            if time.time() > deadline:
                                break
                            cust2 = routes[r2][pos2]
                            if cust2 > n_customers:
                                continue
                            
                            load1 = route_load(routes[r1])
                            load2 = route_load(routes[r2])
                            if load1 - demands[cust1] + demands[cust2] > capacity:
                                continue
                            if load2 - demands[cust2] + demands[cust1] > capacity:
                                continue
                            
                            route1_new = routes[r1][:]
                            route1_new[pos1] = cust2
                            route2_new = routes[r2][:]
                            route2_new[pos2] = cust1
                            
                            if is_feasible(route1_new) and is_feasible(route2_new):
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
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    routes = [r for r in routes if len(r) > 0]
    return routes


def _cluster_based_construct(instance, dist_matrix, n_customers, stations,
                              battery_max, speed, recharge_rate, horizon, capacity,
                              n_clusters=5, deadline=None):
    """Construct solution by clustering customers and optimizing each cluster."""
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    
    # Simple k-means clustering
    customer_coords = coords[1:n_customers+1]
    n_cust = n_customers
    
    # Initialize cluster centers using farthest point sampling
    centers = []
    remaining = list(range(n_cust))
    
    # Pick first center randomly
    first = random.choice(remaining)
    centers.append(customer_coords[first])
    remaining.remove(first)
    
    while len(centers) < n_clusters and remaining:
        # Pick point farthest from existing centers
        max_dist = -1
        max_idx = 0
        for idx, cust_idx in enumerate(remaining):
            min_dist_to_center = min(
                np.sqrt(np.sum((customer_coords[cust_idx] - c)**2))
                for c in centers
            )
            if min_dist_to_center > max_dist:
                max_dist = min_dist_to_center
                max_idx = idx
        centers.append(customer_coords[remaining[max_idx]])
        remaining.pop(max_idx)
    
    # Assign customers to nearest center
    clusters = [[] for _ in range(n_clusters)]
    for cust_idx in range(n_cust):
        min_dist = float('inf')
        best_cluster = 0
        for c_idx, center in enumerate(centers):
            d = np.sqrt(np.sum((customer_coords[cust_idx] - center)**2))
            if d < min_dist:
                min_dist = d
                best_cluster = c_idx
        clusters[best_cluster].append(cust_idx + 1)  # +1 for 1-based indexing
    
    # Build routes from clusters
    routes = []
    for cluster in clusters:
        if not cluster:
            continue
        
        # Sort by due time within cluster
        cluster.sort(key=lambda v: due[v])
        
        # Try to build a feasible route from the cluster
        route = []
        for cust in cluster:
            test_route = route + [cust]
            if _check_route_feasibility(test_route, dist_matrix, instance, n_customers,
                                         stations, battery_max, speed, recharge_rate,
                                         horizon, capacity):
                route = test_route
            else:
                # Start a new route with this customer
                if route:
                    routes.append(route)
                route = [cust]
        
        if route:
            routes.append(route)
    
    return routes


def _simulated_annealing(routes, dist_matrix, instance, deadline, max_iters=5000):
    """Improve solution using simulated annealing."""
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)
    
    def is_feasible(route):
        return _check_route_feasibility(route, dist_matrix, instance, n_customers, stations,
                                         battery_max, speed, recharge_rate, horizon, capacity)
    
    def route_load(route):
        return sum(demands[node] for node in route if node <= n_customers)
    
    current_routes = [r[:] for r in routes]
    current_dist = get_total_dist(current_routes)
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist
    
    start_temp = 10.0
    end_temp = 0.1
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    temp = start_temp
    
    start_time = time.time()
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Generate a random perturbation
        new_routes = [r[:] for r in current_routes]
        
        perturbation_type = random.random()
        
        if perturbation_type < 0.4:
            # Move a customer
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    cust = new_routes[r1][pos1]
                    if cust <= n_customers:
                        new_routes[r1].pop(pos1)
                        
                        # Find a route to insert into
                        valid_targets = [r for r in range(len(new_routes))
                                         if route_load(new_routes[r]) + demands[cust] <= capacity]
                        if valid_targets:
                            r2 = random.choice(valid_targets)
                            if len(new_routes[r2]) > 0:
                                pos2 = random.randint(0, len(new_routes[r2]))
                                new_routes[r2].insert(pos2, cust)
                            else:
                                new_routes[r2].append(cust)
        elif perturbation_type < 0.7:
            # Swap two customers between routes
            if len(new_routes) >= 2:
                r1 = random.randint(0, len(new_routes) - 1)
                r2 = random.randint(0, len(new_routes) - 1)
                if r1 != r2 and len(new_routes[r1]) > 0 and len(new_routes[r2]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    pos2 = random.randint(0, len(new_routes[r2]) - 1)
                    cust1 = new_routes[r1][pos1]
                    cust2 = new_routes[r2][pos2]
                    
                    if cust1 <= n_customers and cust2 <= n_customers:
                        load1 = route_load(new_routes[r1])
                        load2 = route_load(new_routes[r2])
                        if load1 - demands[cust1] + demands[cust2] <= capacity and \
                           load2 - demands[cust2] + demands[cust1] <= capacity:
                            new_routes[r1][pos1] = cust2
                            new_routes[r2][pos2] = cust1
        else:
            # Reverse a segment within a route
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) >= 2:
                    i = random.randint(0, len(new_routes[r1]) - 2)
                    j = random.randint(i + 1, len(new_routes[r1]) - 1)
                    new_routes[r1][i:j+1] = new_routes[r1][i:j+1][::-1]
        
        # Check feasibility
        new_routes = [r for r in new_routes if len(r) > 0]
        
        all_feasible = True
        served = set()
        for route in new_routes:
            if not is_feasible(route):
                all_feasible = False
                break
            for node in route:
                if node <= n_customers:
                    if node in served:
                        all_feasible = False
                        break
                    served.add(node)
            if not all_feasible:
                break
        
        if all_feasible and served == set(range(1, n_customers + 1)):
            new_dist = get_total_dist(new_routes)
            delta = new_dist - current_dist
            
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                current_routes = new_routes
                current_dist = new_dist
                
                if current_dist < best_dist:
                    best_routes = [r[:] for r in current_routes]
                    best_dist = current_dist
        
        temp *= cooling_rate
    
    return best_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Build distance matrix once
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
    
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]
    
    # Get greedy solution
    routes = _greedy_construct(instance)
    
    # Validate
    valid, dist = _validate_solution(routes, instance)
    if not valid:
        customers = list(range(1, n_customers + 1))
        routes = [[v] for v in customers]
    
    # Try cluster-based construction as alternative
    if time.time() < deadline - 10:
        cluster_routes = _cluster_based_construct(
            instance, dist_matrix, n_customers, stations,
            battery_max, speed, recharge_rate, horizon, capacity,
            n_clusters=5, deadline=deadline
        )
        
        valid_c, dist_c = _validate_solution(cluster_routes, instance)
        if valid_c and dist_c < dist:
            routes = cluster_routes
            dist = dist_c
    
    # Local search if time permits
    if time.time() < deadline - 5:
        routes = _local_search(routes, dist_matrix, instance, deadline)
        
        valid, dist = _validate_solution(routes, instance)
        if not valid:
            customers = list(range(1, n_customers + 1))
            routes = [[v] for v in customers]
    
    # Simulated annealing if time permits
    if time.time() < deadline - 3:
        sa_routes = _simulated_annealing(routes, dist_matrix, instance, deadline, max_iters=3000)
        
        valid_sa, dist_sa = _validate_solution(sa_routes, instance)
        if valid_sa and dist_sa < dist:
            routes = sa_routes
            dist = dist_sa
    
    # Final check
    valid, dist = _validate_solution(routes, instance)
    if not valid:
        customers = list(range(1, n_customers + 1))
        routes = [[v] for v in customers]
    
    return routes

# EVOLVE-BLOCK-END

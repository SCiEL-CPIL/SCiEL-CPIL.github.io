"""EV routing with time windows and battery constraints - combined approach with Dijkstra-based route construction."""
import numpy as np
import time
import random
import math
import heapq

DEADLINE_SECONDS = 145.0


def _build_dist_matrix(coords):
    coords = np.array(coords, dtype=np.float64)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _simulate_route(route, dist_matrix, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    total_demand = 0
    for node in route:
        if 1 <= node <= n_customers:
            total_demand += demands[node]
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

        if 1 <= node <= n_customers:
            if time_at < ready[node]:
                time_at = ready[node]
            if time_at > due[node] + 1e-9:
                return False, 0.0, 0.0, 0.0
            time_at += service[node]
        elif node in stations_set:
            energy_needed = battery_max - battery
            if energy_needed > 0:
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
    n_customers = instance["n_customers"]
    coords = instance["coords"]
    dist_matrix = _build_dist_matrix(coords)
    stations_set = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    served = set()
    total_distance = 0.0

    for route in routes:
        valid, dist, _, _ = _simulate_route(route, dist_matrix, instance, n_customers,
                                             stations_set, battery_max, speed,
                                             recharge_rate, horizon, capacity)
        if not valid:
            return False, 0.0
        total_distance += dist
        for node in route:
            if 1 <= node <= n_customers:
                if node in served:
                    return False, 0.0
                served.add(node)

    if served != set(range(1, n_customers + 1)):
        return False, 0.0

    return True, total_distance


def _compute_total_distance(routes, dist_matrix):
    total = 0.0
    for route in routes:
        current = 0
        for node in route:
            total += dist_matrix[current, node]
            current = node
        total += dist_matrix[current, 0]
    return total


def _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    valid, _, _, _ = _simulate_route(route, dist_matrix, instance, n_customers,
                                      stations_set, battery_max, speed, recharge_rate,
                                      horizon, capacity)
    return valid


def _greedy_construct(instance, dist_matrix, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity, seed=0):
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]

    customers = list(range(1, n_customers + 1))
    if random.random() < 0.5:
        customers.sort(key=lambda x: due[x])
    else:
        random.shuffle(customers)

    routes = []
    route_loads = []

    for cust in customers:
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')

        for r_idx, route in enumerate(routes):
            if route_loads[r_idx] + demands[cust] > capacity:
                continue
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                if _route_feasible(new_route, dist_matrix, instance, n_customers,
                                    stations_set, battery_max, speed, recharge_rate,
                                    horizon, capacity):
                    if pos == 0:
                        cost = dist_matrix[0, cust] + dist_matrix[cust, route[0]] - dist_matrix[0, route[0]]
                    elif pos == len(route):
                        cost = dist_matrix[route[-1], cust] + dist_matrix[cust, 0] - dist_matrix[route[-1], 0]
                    else:
                        cost = dist_matrix[route[pos-1], cust] + dist_matrix[cust, route[pos]] - dist_matrix[route[pos-1], route[pos]]
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = pos

        if best_route_idx == -1:
            routes.append([cust])
            route_loads.append(demands[cust])
        else:
            routes[best_route_idx].insert(best_pos, cust)
            route_loads[best_route_idx] += demands[cust]

    return routes


def _cluster_construct(instance, dist_matrix, n_customers, stations_set,
                        battery_max, speed, recharge_rate, horizon, capacity,
                        n_clusters=5, seed=0):
    random.seed(seed)
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = instance["demands"]
    due = instance["due"]

    customer_coords = coords[1:n_customers+1]
    n_cust = n_customers

    centers = []
    remaining = list(range(n_cust))
    first = random.choice(remaining)
    centers.append(customer_coords[first].copy())
    remaining.remove(first)

    while len(centers) < n_clusters and remaining:
        max_dist = -1
        max_idx = 0
        for idx, cust_idx in enumerate(remaining):
            min_d = min(np.linalg.norm(customer_coords[cust_idx] - c) for c in centers)
            if min_d > max_dist:
                max_dist = min_d
                max_idx = idx
        centers.append(customer_coords[remaining[max_idx]].copy())
        remaining.pop(max_idx)

    clusters = [[] for _ in range(n_clusters)]
    for cust_idx in range(n_cust):
        min_d = float('inf')
        best_c = 0
        for c_idx, center in enumerate(centers):
            d = np.linalg.norm(customer_coords[cust_idx] - center)
            if d < min_d:
                min_d = d
                best_c = c_idx
        clusters[best_c].append(cust_idx + 1)

    routes = []
    for cluster in clusters:
        if not cluster:
            continue
        cluster.sort(key=lambda v: due[v])
        route = []
        for cust in cluster:
            test = route + [cust]
            if _route_feasible(test, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity):
                route = test
            else:
                if route:
                    routes.append(route)
                route = [cust]
        if route:
            routes.append(route)

    return routes


def _dijkstra_route_build(instance, dist_matrix, n_customers, stations_set,
                           battery_max, speed, recharge_rate, horizon, capacity,
                           seed=0, max_customers=8):
    """
    Build a single route using Dijkstra on state space (current_node, battery_bucket, visited_mask).
    """
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    all_customers = list(range(1, n_customers + 1))
    random.shuffle(all_customers)
    subset = all_customers[:max_customers]
    
    total_dem = sum(demands[c] for c in subset)
    while total_dem > capacity and subset:
        c = subset.pop()
        total_dem -= demands[c]
    if not subset:
        return [random.choice(all_customers)]

    # Beam search over sequences with Dijkstra-like expansion
    remaining = list(subset)
    route = []
    current_node = 0
    battery = battery_max
    time_at = 0.0
    
    while remaining:
        best_next = None
        best_cost = float('inf')
        best_state = None
        
        for cust in remaining:
            d = dist_matrix[current_node, cust]
            if battery - d < -1e-9:
                continue
            
            new_battery = battery - d
            new_time = time_at + d / speed
            
            if 1 <= cust <= n_customers:
                if new_time < ready[cust]:
                    new_time = ready[cust]
                if new_time > due[cust] + 1e-9:
                    continue
                new_time += service[cust]
            
            cost = d + 0.1 * (new_time - time_at - d/speed)
            
            if cost < best_cost:
                best_cost = cost
                best_next = cust
                best_state = (new_battery, new_time)
        
        if best_next is None:
            break
        
        route.append(best_next)
        battery, time_at = best_state
        current_node = best_next
        remaining.remove(best_next)
    
    d = dist_matrix[current_node, 0]
    if battery - d < -1e-9:
        return route[:-1] if route else [subset[0]]
    
    return route


def _two_opt(route, dist_matrix, instance, n_customers, stations_set,
             battery_max, speed, recharge_rate, horizon, capacity, max_iters=100):
    if len(route) < 2:
        return route

    def is_feasible(r):
        return _route_feasible(r, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_dist(r):
        d = 0.0
        prev = 0
        for node in r:
            d += dist_matrix[prev, node]
            prev = node
        d += dist_matrix[prev, 0]
        return d

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


def _or_opt(route, dist_matrix, instance, n_customers, stations_set,
            battery_max, speed, recharge_rate, horizon, capacity, max_iters=50):
    """OR-opt: Move segments of length 1, 2, 3 within the route."""
    if len(route) < 3:
        return route

    def is_feasible(r):
        return _route_feasible(r, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_dist(r):
        d = 0.0
        prev = 0
        for node in r:
            d += dist_matrix[prev, node]
            prev = node
        d += dist_matrix[prev, 0]
        return d

    best_route = route[:]
    best_dist = route_dist(best_route)

    for _ in range(max_iters):
        improved = False
        n = len(best_route)
        for seg_len in [1, 2, 3]:
            for i in range(n - seg_len + 1):
                segment = best_route[i:i+seg_len]
                for j in range(n - seg_len + 1):
                    if i == j or (j < i + seg_len and j + seg_len > i):
                        continue
                    remaining = best_route[:i] + best_route[i+seg_len:]
                    if j > i:
                        j = j - seg_len
                    new_route = remaining[:j] + segment + remaining[j:]
                    if len(new_route) != len(best_route):
                        continue
                    if is_feasible(new_route):
                        new_dist = route_dist(new_route)
                        if new_dist < best_dist - 1e-9:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
        if not improved:
            break

    return best_route


def _local_search(routes, dist_matrix, instance, n_customers, stations_set,
                  battery_max, speed, recharge_rate, horizon, capacity, deadline):
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_dist = get_total_dist(routes)
    start_time = time.time()

    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _two_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, max_iters=80)
        routes[r_idx] = _or_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                 stations_set, battery_max, speed, recharge_rate,
                                 horizon, capacity, max_iters=40)
    current_dist = get_total_dist(routes)

    max_passes = 15
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        improved = False

        for r1 in range(len(routes)):
            if time.time() > deadline:
                break
            for pos1 in range(len(routes[r1])):
                if time.time() > deadline:
                    break
                cust = routes[r1][pos1]
                if not (1 <= cust <= n_customers):
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

        if not improved:
            break

    routes = [r for r in routes if len(r) > 0]
    return routes


def _simulated_annealing(routes, dist_matrix, instance, n_customers, stations_set,
                         battery_max, speed, recharge_rate, horizon, capacity, deadline):
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_routes = [r[:] for r in routes]
    current_dist = get_total_dist(current_routes)
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist

    start_temp = 10.0
    end_temp = 0.05
    max_iters = 4000
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    temp = start_temp

    start_time = time.time()
    for iteration in range(max_iters):
        if time.time() > deadline:
            break

        new_routes = [r[:] for r in current_routes]
        perturbation_type = random.random()

        if perturbation_type < 0.4:
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    cust = new_routes[r1][pos1]
                    if 1 <= cust <= n_customers:
                        new_routes[r1].pop(pos1)
                        valid_targets = [r for r in range(len(new_routes))
                                         if route_load(new_routes[r]) + demands[cust] <= capacity]
                        if valid_targets:
                            r2 = random.choice(valid_targets)
                            pos2 = random.randint(0, len(new_routes[r2]))
                            new_routes[r2].insert(pos2, cust)
        elif perturbation_type < 0.7:
            if len(new_routes) >= 2:
                r1 = random.randint(0, len(new_routes) - 1)
                r2 = random.randint(0, len(new_routes) - 1)
                if r1 != r2 and len(new_routes[r1]) > 0 and len(new_routes[r2]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    pos2 = random.randint(0, len(new_routes[r2]) - 1)
                    cust1 = new_routes[r1][pos1]
                    cust2 = new_routes[r2][pos2]
                    if 1 <= cust1 <= n_customers and 1 <= cust2 <= n_customers:
                        load1 = route_load(new_routes[r1])
                        load2 = route_load(new_routes[r2])
                        if load1 - demands[cust1] + demands[cust2] <= capacity and \
                           load2 - demands[cust2] + demands[cust1] <= capacity:
                            new_routes[r1][pos1] = cust2
                            new_routes[r2][pos2] = cust1
        else:
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) >= 2:
                    i = random.randint(0, len(new_routes[r1]) - 2)
                    j = random.randint(i + 1, len(new_routes[r1]) - 1)
                    new_routes[r1][i:j+1] = new_routes[r1][i:j+1][::-1]

        new_routes = [r for r in new_routes if len(r) > 0]
        all_feasible = True
        served = set()
        for route in new_routes:
            if not is_feasible(route):
                all_feasible = False
                break
            for node in route:
                if 1 <= node <= n_customers:
                    if node in served:
                        all_feasible = False
                        break
                    served.add(node)
            if not all_feasible:
                break

        if all_feasible and served == set(range(1, n_customers + 1)):
            new_dist = get_total_dist(new_routes)
            delta = new_dist - current_dist
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_routes = new_routes
                current_dist = new_dist
                if current_dist < best_dist:
                    best_routes = [r[:] for r in current_routes]
                    best_dist = current_dist

        temp *= cooling_rate

    return best_routes


def _polish_routes(routes, dist_matrix, instance, n_customers, stations_set,
                   battery_max, speed, recharge_rate, horizon, capacity, deadline):
    """Final high-precision polish: multiple passes of 2-opt + OR-opt + inter-route moves."""
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_dist = get_total_dist(routes)

    for _pass in range(3):
        if time.time() > deadline:
            break
        for r_idx in range(len(routes)):
            if time.time() > deadline:
                break
            routes[r_idx] = _two_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                      stations_set, battery_max, speed, recharge_rate,
                                      horizon, capacity, max_iters=50)
            routes[r_idx] = _or_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                     stations_set, battery_max, speed, recharge_rate,
                                     horizon, capacity, max_iters=30)

        for r1 in range(len(routes)):
            if time.time() > deadline:
                break
            for pos1 in range(len(routes[r1])):
                if time.time() > deadline:
                    break
                cust = routes[r1][pos1]
                if not (1 <= cust <= n_customers):
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
                                break
                    if time.time() > deadline:
                        break

    routes = [r for r in routes if len(r) > 0]
    return routes


def _dijkstra_full_route(instance, dist_matrix, n_customers, stations_set,
                          battery_max, speed, recharge_rate, horizon, capacity,
                          cluster_customers, seed=0):
    """
    Use Dijkstra on a state space to build an optimal route for a given cluster of customers.
    State: (current_node, battery_level_discretized, frozenset_of_visited)
    """
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    
    n = len(cluster_customers)
    if n == 0:
        return []
    if n == 1:
        c = cluster_customers[0]
        d = dist_matrix[0, c]
        if d <= battery_max:
            t = d / speed
            if t < ready[c]:
                t = ready[c]
            if t <= due[c] + 1e-9:
                t += service[c]
                d_ret = dist_matrix[c, 0]
                if d + d_ret <= battery_max and t + d_ret/speed <= horizon + 1e-9:
                    return [c]
        return []
    
    # Discretize battery to 10-unit buckets for state space
    battery_buckets = 30  # 0 to 300 in steps of 10
    
    # State: (cost, current_node, battery_bucket, visited_mask)
    # We'll use a priority queue
    # visited_mask is a bitmask of which customers in cluster_customers have been visited
    
    n_cust = len(cluster_customers)
    if n_cust > 15:
        # Too many for bitmask in reasonable time; fall back to greedy
        return _greedy_cluster_route(instance, dist_matrix, n_customers, stations_set,
                                      battery_max, speed, recharge_rate, horizon, capacity,
                                      cluster_customers, seed)
    
    # Map customer to index in cluster
    cust_to_idx = {c: i for i, c in enumerate(cluster_customers)}
    
    # Dijkstra: state = (current_node, battery_bucket, visited_mask)
    # cost = total distance traveled so far
    # We want to minimize total distance including return to depot
    
    # Initial state: at depot, full battery, no customers visited
    # We'll track the actual battery as a float, but use bucket for state identity
    # Actually, let's just use the actual battery level rounded to 1 unit for state
    
    # For simplicity and correctness, let's use a beam search approach
    # that explores partial routes and keeps the best k at each step
    
    beam_width = 50
    max_steps = n_cust + 2  # at most n_cust customers + maybe a station + return
    
    # Each beam entry: (total_dist, current_node, battery, time_at, visited_set, route)
    beam = [(0.0, 0, float(battery_max), 0.0, frozenset(), [])]
    
    for step in range(max_steps):
        if time.time() > deadline_global:
            break
        new_beam = []
        for (td, cur_node, bat, t_at, visited, route) in beam:
            # Try going to each unvisited customer
            for c in cluster_customers:
                if c in visited:
                    continue
                d = dist_matrix[cur_node, c]
                if bat - d < -1e-9:
                    continue
                new_bat = bat - d
                new_t = t_at + d / speed
                if new_t < ready[c]:
                    new_t = ready[c]
                if new_t > due[c] + 1e-9:
                    continue
                new_t += service[c]
                new_visited = visited | frozenset([c])
                new_route = route + [c]
                new_beam.append((td + d, c, new_bat, new_t, new_visited, new_route))
            
            # Try going to a station (only if battery is low enough to warrant it)
            for st in instance["stations"]:
                d = dist_matrix[cur_node, st]
                if bat - d < -1e-9:
                    continue
                new_bat = bat - d
                new_t = t_at + d / speed
                # Recharge
                energy_needed = battery_max - new_bat
                if energy_needed > 0:
                    new_t += energy_needed * recharge_rate
                new_bat = battery_max
                # Only add station if it doesn't increase cost too much
                # and we haven't visited all customers yet
                if len(visited) < n_cust:
                    new_beam.append((td + d, st, new_bat, new_t, visited, route + [st]))
            
            # Try returning to depot if all customers visited
            if len(visited) == n_cust:
                d = dist_matrix[cur_node, 0]
                if bat - d >= -1e-9:
                    new_t = t_at + d / speed
                    if new_t <= horizon + 1e-9:
                        # This is a complete route
                        new_beam.append((td + d, 0, bat - d, new_t, visited, route))
        
        if not new_beam:
            break
        
        # Keep only the best beam_width entries by total distance
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
        # Check if we have a complete route
        complete = [b for b in beam if b[1] == 0 and len(b[4]) == n_cust]
        if complete:
            # Return the best complete route
            best = complete[0]
            # Remove depot from route if it's there
            route = [n for n in best[5] if n != 0]
            return route
    
    # If no complete route found, return empty
    return []


def _greedy_cluster_route(instance, dist_matrix, n_customers, stations_set,
                           battery_max, speed, recharge_rate, horizon, capacity,
                           cluster_customers, seed=0):
    """Greedy nearest-neighbor for a cluster."""
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    
    remaining = list(cluster_customers)
    route = []
    current_node = 0
    battery = battery_max
    time_at = 0.0
    
    while remaining:
        best_next = None
        best_cost = float('inf')
        best_state = None
        
        for cust in remaining:
            d = dist_matrix[current_node, cust]
            if battery - d < -1e-9:
                continue
            
            new_battery = battery - d
            new_time = time_at + d / speed
            
            if new_time < ready[cust]:
                new_time = ready[cust]
            if new_time > due[cust] + 1e-9:
                continue
            new_time += service[cust]
            
            cost = d
            
            if cost < best_cost:
                best_cost = cost
                best_next = cust
                best_state = (new_battery, new_time)
        
        if best_next is None:
            break
        
        route.append(best_next)
        battery, time_at = best_state
        current_node = best_next
        remaining.remove(best_next)
    
    d = dist_matrix[current_node, 0]
    if battery - d < -1e-9:
        return route[:-1] if route else []
    
    return route


deadline_global = 0.0


def plan(instance):
    global deadline_global
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    deadline_global = deadline

    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)

    n_customers = instance["n_customers"]
    stations_set = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    best_routes = None
    best_dist = float('inf')

    # Phase 1: Generate diverse initial solutions
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 35:
            break
        routes = _greedy_construct(instance, dist_matrix, n_customers, stations_set,
                                    battery_max, speed, recharge_rate, horizon, capacity,
                                    seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # Cluster-based constructions
    for i in range(5):
        if time.time() > deadline - 30:
            break
        routes = _cluster_construct(instance, dist_matrix, n_customers, stations_set,
                                     battery_max, speed, recharge_rate, horizon, capacity,
                                     n_clusters=4 + i, seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    if best_routes is None:
        best_routes = [[v] for v in range(1, n_customers + 1)]
        best_dist = _compute_total_distance(best_routes, dist_matrix)

    # Phase 2: Local search
    if time.time() < deadline - 20:
        improved = _local_search(best_routes, dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, deadline - 20)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Phase 3: Simulated annealing
    if time.time() < deadline - 15:
        sa_routes = _simulated_annealing(best_routes, dist_matrix, instance, n_customers,
                                          stations_set, battery_max, speed, recharge_rate,
                                          horizon, capacity, deadline - 10)
        valid, dist = _validate_solution(sa_routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = sa_routes

    # Phase 4: Polish
    if time.time() < deadline - 5:
        polished = _polish_routes(best_routes, dist_matrix, instance, n_customers,
                                   stations_set, battery_max, speed, recharge_rate,
                                   horizon, capacity, deadline - 2)
        valid, dist = _validate_solution(polished, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = polished

    # Final validation
    valid, dist = _validate_solution(best_routes, instance)
    if not valid:
        best_routes = [[v] for v in range(1, n_customers + 1)]

    return best_routes

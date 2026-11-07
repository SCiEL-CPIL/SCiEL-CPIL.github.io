"""Optimized EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import sqrt

DEADLINE_SECONDS = 140.0


def _build_dist_matrix(coords):
    """Precompute distance matrix using vectorized operations."""
    coords = np.array(coords, dtype=np.float64)
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist_matrix


def _simulate_route(route, dist_matrix, n_customers, stations,
                    battery_max, speed, recharge_rate, horizon,
                    demands, ready, due, service):
    """Simulate a route. Returns (feasible, distance)."""
    total_demand = 0
    for node in route:
        if node <= n_customers:
            total_demand += demands[node]
    if total_demand > 100:
        return False, 0.0

    battery = float(battery_max)
    time_at = 0.0
    distance = 0.0
    current_node = 0

    for node in route:
        d = dist_matrix[current_node, node]
        time_at += d
        battery -= d
        distance += d

        if battery < -1e-9:
            return False, 0.0

        if node <= n_customers:
            r = ready[node]
            if time_at < r:
                time_at = r
            if time_at > due[node] + 1e-9:
                return False, 0.0
            time_at += service[node]
        elif node in stations:
            energy_needed = battery_max - battery
            time_at += energy_needed * recharge_rate
            battery = float(battery_max)

        current_node = node

    d = dist_matrix[current_node, 0]
    time_at += d
    battery -= d
    distance += d

    if battery < -1e-9:
        return False, 0.0
    if time_at > horizon + 1e-9:
        return False, 0.0

    return True, distance


def _validate_solution(routes, instance):
    """Validate all routes. Returns (valid, total_distance)."""
    n_customers = instance["n_customers"]
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
    stations = set(instance["stations"])
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    served = set()
    total_distance = 0.0

    for route in routes:
        feasible, dist = _simulate_route(route, dist_matrix, n_customers, stations,
                                          battery_max, speed, recharge_rate, horizon,
                                          demands, ready, due, service)
        if not feasible:
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


def _nearest_neighbor_construct(instance):
    """Construct initial solution using nearest neighbor with time window awareness."""
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
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

                feasible, _ = _simulate_route(new_route, dist_matrix, n_customers,
                                               stations, battery_max, speed,
                                               recharge_rate, horizon,
                                               demands, ready, due, service)
                if not feasible:
                    continue

                if pos == 0:
                    cost = (dist_matrix[0, cust] + dist_matrix[cust, route[0]] -
                           dist_matrix[0, route[0]])
                elif pos == len(route):
                    cost = (dist_matrix[route[-1], cust] + dist_matrix[cust, 0] -
                           dist_matrix[route[-1], 0])
                else:
                    cost = (dist_matrix[route[pos-1], cust] + dist_matrix[cust, route[pos]] -
                           dist_matrix[route[pos-1], route[pos]])

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


def _savings_construct(instance):
    """Construct using Clarke-Wright savings heuristic."""
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
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

    # Start with each customer as separate route
    routes = [[i] for i in range(1, n_customers + 1)]
    route_loads = [demands[i] for i in range(1, n_customers + 1)]
    route_index = {i: i - 1 for i in range(1, n_customers + 1)}

    # Compute savings
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    for s, i, j in savings:
        if s <= 0:
            continue

        ri = route_index.get(i)
        rj = route_index.get(j)
        if ri is None or rj is None or ri == rj:
            continue

        if route_loads[ri] + route_loads[rj] > capacity:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        # Try merging: route_i + route_j or route_j + route_i
        if route_i[-1] == i and route_j[0] == j:
            new_route = route_i + route_j
        elif route_i[0] == i and route_j[-1] == j:
            new_route = route_j + route_i
        elif route_i[-1] == j and route_j[0] == i:
            new_route = route_j + route_i
        elif route_i[0] == j and route_j[-1] == i:
            new_route = route_i + route_j
        else:
            continue

        feasible, _ = _simulate_route(new_route, dist_matrix, n_customers,
                                       stations, battery_max, speed,
                                       recharge_rate, horizon,
                                       demands, ready, due, service)
        if not feasible:
            continue

        routes[ri] = new_route
        routes[rj] = []
        route_loads[ri] += route_loads[rj]
        route_loads[rj] = 0
        for cust in new_route:
            route_index[cust] = ri

    return [r for r in routes if len(r) > 0]


def _cluster_based_construct(instance, seed=42):
    """Construct routes using k-means clustering with multiple restarts."""
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    dist_matrix = _build_dist_matrix(coords)

    total_demand = sum(demands[1:n_customers+1])
    n_vehicles = max(1, int(np.ceil(total_demand / capacity)))
    n_vehicles = min(n_vehicles, n_customers)

    customer_coords = coords[1:n_customers+1]

    best_routes = None
    best_dist = float('inf')
    n_restarts = 3

    for restart in range(n_restarts):
        rng = random.Random(seed + restart)
        indices = rng.sample(range(n_customers), n_vehicles)
        centroids = customer_coords[indices].copy()

        for _ in range(15):
            labels = np.argmin(np.linalg.norm(customer_coords[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2), axis=1)

            new_centroids = []
            for k in range(n_vehicles):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centroids.append(customer_coords[mask].mean(axis=0))
                else:
                    new_centroids.append(centroids[k])
            centroids = np.array(new_centroids)

        routes = []
        for k in range(n_vehicles):
            mask = labels == k
            cluster_customers = [i + 1 for i in range(n_customers) if mask[i]]

            if len(cluster_customers) == 0:
                continue

            cluster_demand = sum(demands[c] for c in cluster_customers)
            if cluster_demand > capacity:
                half = len(cluster_customers) // 2
                routes.append(cluster_customers[:half])
                routes.append(cluster_customers[half:])
            else:
                routes.append(cluster_customers)

        for r_idx in range(len(routes)):
            routes[r_idx] = _two_opt_route(
                routes[r_idx], dist_matrix, n_customers, stations,
                battery_max, speed, recharge_rate, horizon,
                demands, ready, due, service, max_iters=20
            )

        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    return best_routes if best_routes else _cluster_based_construct_single(instance)


def _two_opt_route(route, dist_matrix, n_customers, stations,
                   battery_max, speed, recharge_rate, horizon,
                   demands, ready, due, service, max_iters=30):
    """Optimize route order using 2-opt with station awareness."""
    if len(route) < 2:
        return route

    best_route = route[:]
    best_dist = _compute_total_distance([best_route], dist_matrix)

    for _ in range(max_iters):
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]

                feasible, _ = _simulate_route(new_route, dist_matrix, n_customers,
                                               stations, battery_max, speed,
                                               recharge_rate, horizon,
                                               demands, ready, due, service)
                if not feasible:
                    continue

                new_dist = _compute_total_distance([new_route], dist_matrix)
                if new_dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
        if not improved:
            break

    # Final pass: optimize station placement
    best_route = _optimize_stations_in_route(best_route, dist_matrix, n_customers,
                                              stations, battery_max, speed,
                                              recharge_rate, horizon,
                                              demands, ready, due, service)

    return best_route


def _or_opt_route(route, dist_matrix, n_customers, stations,
                  battery_max, speed, recharge_rate, horizon,
                  demands, ready, due, service, max_iters=20):
    """OR-opt: move 1-3 consecutive customers to another position."""
    if len(route) < 3:
        return route

    best_route = route[:]
    best_dist = _compute_total_distance([best_route], dist_matrix)

    for _ in range(max_iters):
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(len(best_route) - seg_len):
                segment = best_route[i:i+seg_len]
                remaining = best_route[:i] + best_route[i+seg_len:]

                for j in range(len(remaining) + 1):
                    new_route = remaining[:j] + segment + remaining[j:]
                    if new_route == best_route:
                        continue

                    feasible, _ = _simulate_route(new_route, dist_matrix, n_customers,
                                                   stations, battery_max, speed,
                                                   recharge_rate, horizon,
                                                   demands, ready, due, service)
                    if feasible:
                        new_dist = _compute_total_distance([new_route], dist_matrix)
                        if new_dist < best_dist - 1e-9:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
        if not improved:
            break

    # Final pass: optimize station placement
    best_route = _optimize_stations_in_route(best_route, dist_matrix, n_customers,
                                              stations, battery_max, speed,
                                              recharge_rate, horizon,
                                              demands, ready, due, service)

    return best_route


def _optimize_stations_in_route(route, dist_matrix, n_customers, stations,
                                 battery_max, speed, recharge_rate, horizon,
                                 demands, ready, due, service):
    """Try to improve station placement within a route."""
    if not route:
        return route
    
    best_route = route[:]
    best_dist = _compute_total_distance([best_route], dist_matrix)
    
    # Try removing each station and see if still feasible
    for i in range(len(best_route)):
        if best_route[i] in stations:
            candidate = best_route[:i] + best_route[i+1:]
            feasible, dist = _simulate_route(candidate, dist_matrix, n_customers,
                                              stations, battery_max, speed,
                                              recharge_rate, horizon,
                                              demands, ready, due, service)
            if feasible and dist < best_dist - 1e-9:
                best_route = candidate
                best_dist = dist
    
    # Try adding stations at strategic points if battery is tight
    if len(best_route) >= 2:
        feasible, _ = _simulate_route(best_route, dist_matrix, n_customers,
                                       stations, battery_max, speed,
                                       recharge_rate, horizon,
                                       demands, ready, due, service)
        if not feasible:
            for i in range(len(best_route)):
                for j in range(i+1, len(best_route)):
                    if best_route[j] in stations:
                        continue
                    for s in stations:
                        candidate = best_route[:j] + [s] + best_route[j:]
                        feasible2, dist2 = _simulate_route(candidate, dist_matrix, n_customers,
                                                            stations, battery_max, speed,
                                                            recharge_rate, horizon,
                                                            demands, ready, due, service)
                        if feasible2:
                            if dist2 < best_dist - 1e-9:
                                best_route = candidate
                                best_dist = dist2
                            break
    
    return best_route


def _inter_route_search(routes, dist_matrix, instance, deadline):
    """Inter-route moves: relocate and swap customers between routes."""
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    def is_feasible(route):
        if not route:
            return True
        feasible, _ = _simulate_route(route, dist_matrix, n_customers, stations,
                                       battery_max, speed, recharge_rate, horizon,
                                       demands, ready, due, service)
        return feasible

    def route_load(route):
        return sum(demands[node] for node in route if node <= n_customers)

    current_dist = _compute_total_distance(routes, dist_matrix)

    max_passes = 50
    for _pass in range(max_passes):
        if time.time() > deadline:
            break

        improved = False

        # Try relocating customers
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
                            new_dist = _compute_total_distance(new_routes, dist_matrix)
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

        # Try swapping customers
        if not improved:
            for r1 in range(len(routes)):
                if time.time() > deadline:
                    break
                for i in range(len(routes[r1])):
                    if time.time() > deadline:
                        break
                    c1 = routes[r1][i]
                    if c1 > n_customers:
                        continue

                    for r2 in range(r1 + 1, len(routes)):
                        if time.time() > deadline:
                            break
                        for j in range(len(routes[r2])):
                            if time.time() > deadline:
                                break
                            c2 = routes[r2][j]
                            if c2 > n_customers:
                                continue

                            load1 = route_load(routes[r1])
                            load2 = route_load(routes[r2])

                            if load1 - demands[c1] + demands[c2] > capacity:
                                continue
                            if load2 - demands[c2] + demands[c1] > capacity:
                                continue

                            new_routes = [r[:] for r in routes]
                            new_routes[r1][i] = c2
                            new_routes[r2][j] = c1

                            if is_feasible(new_routes[r1]) and is_feasible(new_routes[r2]):
                                new_dist = _compute_total_distance(new_routes, dist_matrix)
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

    return [r for r in routes if len(r) > 0]


def _simulated_annealing(routes, dist_matrix, instance, deadline):
    """Simulated annealing with improved cooling and move diversity."""
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    def is_feasible(route):
        if not route:
            return True
        feasible, _ = _simulate_route(route, dist_matrix, n_customers, stations,
                                       battery_max, speed, recharge_rate, horizon,
                                       demands, ready, due, service)
        return feasible

    def route_load(route):
        return sum(demands[node] for node in route if node <= n_customers)

    current_routes = [r[:] for r in routes]
    current_dist = _compute_total_distance(current_routes, dist_matrix)
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist

    T = 20.0
    T_min = 0.005
    cooling = 0.998
    rng = random.Random(42)
    no_improve_count = 0

    while T > T_min and time.time() < deadline:
        # Increase exploration as we cool
        exploration_factor = min(1.0, T / 10.0)
        move_weights = [
            3 + 2 * exploration_factor,  # relocate
            2 + exploration_factor,      # swap
            2 + 2 * exploration_factor,  # reverse
            1 + exploration_factor       # 2swap
        ]
        move_type = rng.choices(['relocate', 'swap', 'reverse', '2swap'],
                                  weights=move_weights)[0]

        if move_type == 'relocate':
            r1 = rng.randint(0, len(current_routes) - 1)
            if not current_routes[r1]:
                continue
            pos1 = rng.randint(0, len(current_routes[r1]) - 1)
            cust = current_routes[r1][pos1]
            if cust > n_customers:
                continue

            r2 = rng.randint(0, len(current_routes) - 1)
            if r1 == r2:
                continue
            if route_load(current_routes[r2]) + demands[cust] > capacity:
                continue

            pos2 = rng.randint(0, len(current_routes[r2]))

            new_routes = [r[:] for r in current_routes]
            new_routes[r1] = current_routes[r1][:pos1] + current_routes[r1][pos1+1:]
            new_routes[r2] = current_routes[r2][:pos2] + [cust] + current_routes[r2][pos2:]

            if is_feasible(new_routes[r1]) and is_feasible(new_routes[r2]):
                new_dist = _compute_total_distance(new_routes, dist_matrix)
                delta = new_dist - current_dist
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_routes = new_routes
                    current_dist = new_dist
                    if current_dist < best_dist:
                        best_routes = [r[:] for r in current_routes]
                        best_dist = current_dist
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

        elif move_type == 'swap':
            r1 = rng.randint(0, len(current_routes) - 1)
            r2 = rng.randint(0, len(current_routes) - 1)
            if r1 == r2 or not current_routes[r1] or not current_routes[r2]:
                continue
            i = rng.randint(0, len(current_routes[r1]) - 1)
            j = rng.randint(0, len(current_routes[r2]) - 1)
            c1 = current_routes[r1][i]
            c2 = current_routes[r2][j]
            if c1 > n_customers or c2 > n_customers:
                continue

            load1 = route_load(current_routes[r1])
            load2 = route_load(current_routes[r2])
            if load1 - demands[c1] + demands[c2] > capacity:
                continue
            if load2 - demands[c2] + demands[c1] > capacity:
                continue

            new_routes = [r[:] for r in current_routes]
            new_routes[r1][i] = c2
            new_routes[r2][j] = c1

            if is_feasible(new_routes[r1]) and is_feasible(new_routes[r2]):
                new_dist = _compute_total_distance(new_routes, dist_matrix)
                delta = new_dist - current_dist
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_routes = new_routes
                    current_dist = new_dist
                    if current_dist < best_dist:
                        best_routes = [r[:] for r in current_routes]
                        best_dist = current_dist
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

        elif move_type == 'reverse':
            r1 = rng.randint(0, len(current_routes) - 1)
            if len(current_routes[r1]) < 2:
                continue
            i = rng.randint(0, len(current_routes[r1]) - 2)
            j = rng.randint(i + 1, len(current_routes[r1]) - 1)

            new_routes = [r[:] for r in current_routes]
            new_routes[r1] = current_routes[r1][:i] + current_routes[r1][i:j+1][::-1] + current_routes[r1][j+1:]

            if is_feasible(new_routes[r1]):
                new_dist = _compute_total_distance(new_routes, dist_matrix)
                delta = new_dist - current_dist
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_routes = new_routes
                    current_dist = new_dist
                    if current_dist < best_dist:
                        best_routes = [r[:] for r in current_routes]
                        best_dist = current_dist
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

        else:  # move_type == '2swap'
            r1 = rng.randint(0, len(current_routes) - 1)
            r2 = rng.randint(0, len(current_routes) - 1)
            if r1 == r2 or not current_routes[r1] or not current_routes[r2]:
                continue
            i = rng.randint(0, len(current_routes[r1]) - 1)
            j = rng.randint(0, len(current_routes[r2]) - 1)
            c1 = current_routes[r1][i]
            c2 = current_routes[r2][j]
            if c1 > n_customers or c2 > n_customers:
                continue
            new_routes = [r[:] for r in current_routes]
            new_routes[r1][i] = c2
            new_routes[r2][j] = c1
            if is_feasible(new_routes[r1]) and is_feasible(new_routes[r2]):
                new_dist = _compute_total_distance(new_routes, dist_matrix)
                delta = new_dist - current_dist
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_routes = new_routes
                    current_dist = new_dist
                    if current_dist < best_dist:
                        best_routes = [r[:] for r in current_routes]
                        best_dist = current_dist
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

        T *= cooling

        # Restart if stuck
        if no_improve_count > 500:
            T = max(T, 5.0)
            no_improve_count = 0

    return [r for r in best_routes if len(r) > 0]


def _cluster_based_construct_single(instance):
    """Fallback single-cluster construction."""
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    dist_matrix = _build_dist_matrix(coords)

    total_demand = sum(demands[1:n_customers+1])
    n_vehicles = max(1, int(np.ceil(total_demand / capacity)))
    n_vehicles = min(n_vehicles, n_customers)

    customer_coords = coords[1:n_customers+1]

    random.seed(42)
    indices = random.sample(range(n_customers), n_vehicles)
    centroids = customer_coords[indices].copy()

    for _ in range(10):
        labels = np.argmin(np.linalg.norm(customer_coords[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2), axis=1)

        new_centroids = []
        for k in range(n_vehicles):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centroids.append(customer_coords[mask].mean(axis=0))
            else:
                new_centroids.append(centroids[k])
        centroids = np.array(new_centroids)

    routes = []
    for k in range(n_vehicles):
        mask = labels == k
        cluster_customers = [i + 1 for i in range(n_customers) if mask[i]]

        if len(cluster_customers) == 0:
            continue

        cluster_demand = sum(demands[c] for c in cluster_customers)
        if cluster_demand > capacity:
            half = len(cluster_customers) // 2
            routes.append(cluster_customers[:half])
            routes.append(cluster_customers[half:])
        else:
            routes.append(cluster_customers)

    for r_idx in range(len(routes)):
        routes[r_idx] = _two_opt_route(
            routes[r_idx], dist_matrix, n_customers, stations,
            battery_max, speed, recharge_rate, horizon,
            demands, ready, due, service, max_iters=15
        )

    return routes


def _dijkstra_route_construct(instance):
    """Construct routes using k-means + greedy with battery awareness."""
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
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

    total_demand = sum(demands[1:n_customers+1])
    n_vehicles = max(2, int(np.ceil(total_demand / capacity)))
    n_vehicles = min(n_vehicles, n_customers)

    customer_coords = coords[1:n_customers+1]
    random.seed(123)
    indices = random.sample(range(n_customers), n_vehicles)
    centroids = customer_coords[indices].copy()

    for _ in range(15):
        labels = np.argmin(
            np.linalg.norm(customer_coords[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2), axis=1)
        new_centroids = []
        for k in range(n_vehicles):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centroids.append(customer_coords[mask].mean(axis=0))
            else:
                new_centroids.append(centroids[k])
        centroids = np.array(new_centroids)

    clusters = []
    for k in range(n_vehicles):
        mask = labels == k
        cluster = [i + 1 for i in range(n_customers) if mask[i]]
        if cluster:
            clusters.append(cluster)

    routes = []
    for cluster in clusters:
        if not cluster:
            continue
        route = _greedy_cluster_route(cluster, dist_matrix, n_customers, stations,
                                       battery_max, speed, recharge_rate, horizon,
                                       demands, ready, due, service)
        if route:
            routes.append(route)
        else:
            routes.extend([[c] for c in cluster])

    # Ensure all customers are covered
    served = set()
    for r in routes:
        for c in r:
            if c <= n_customers:
                served.add(c)
    missing = set(range(1, n_customers + 1)) - served
    for c in missing:
        routes.append([c])

    return routes


def _greedy_cluster_route(customers, dist_matrix, n_customers, stations,
                           battery_max, speed, recharge_rate, horizon,
                           demands, ready, due, service):
    """Use greedy insertion to build a route for a cluster."""
    if not customers:
        return []
    if len(customers) == 1:
        c = customers[0]
        route = [c]
        feasible, _ = _simulate_route(route, dist_matrix, n_customers, stations,
                                       battery_max, speed, recharge_rate, horizon,
                                       demands, ready, due, service)
        return route if feasible else [c]

    # For small clusters, try permutations
    if len(customers) <= 7:
        from itertools import permutations
        best_route = None
        best_dist = float('inf')
        for perm in permutations(customers):
            route = list(perm)
            feasible, dist = _simulate_route(route, dist_matrix, n_customers, stations,
                                              battery_max, speed, recharge_rate, horizon,
                                              demands, ready, due, service)
            if feasible and dist < best_dist:
                best_dist = dist
                best_route = route
        return best_route if best_route else list(customers)

    # Greedy insertion
    route = []
    remaining = set(customers)
    current_node = 0
    current_time = 0.0
    current_battery = float(battery_max)

    while remaining:
        best_cust = None
        best_cost = float('inf')

        for cust in remaining:
            d = dist_matrix[current_node, cust]
            if current_battery - d < -1e-9:
                continue

            t_arrive = current_time + d
            if t_arrive < ready[cust]:
                t_arrive = ready[cust]
            if t_arrive > due[cust] + 1e-9:
                continue

            t_after = t_arrive + service[cust]
            b_after = current_battery - d

            tw_penalty = max(0, (due[cust] - t_after)) * 0.01
            cost = d + tw_penalty

            if cost < best_cost:
                best_cost = cost
                best_cust = cust

        if best_cust is None:
            for cust in remaining:
                d = dist_matrix[0, cust]
                t_arrive = d
                if t_arrive < ready[cust]:
                    t_arrive = ready[cust]
                if t_arrive > due[cust] + 1e-9:
                    continue
                b_after = float(battery_max) - d
                if b_after < -1e-9:
                    continue
                cost = d
                if cost < best_cost:
                    best_cost = cost
                    best_cust = cust
                    current_node = 0
                    current_time = 0.0
                    current_battery = float(battery_max)
                    route = []

        if best_cust is None:
            break

        route.append(best_cust)
        remaining.discard(best_cust)
        current_node = best_cust
        d_from = dist_matrix[route[-2] if len(route) >= 2 else 0, best_cust] if len(route) >= 2 else dist_matrix[0, best_cust]
        current_battery -= d_from
        current_time += d_from
        if current_time < ready[best_cust]:
            current_time = ready[best_cust]
        current_time += service[best_cust]

    return route if route else list(customers)


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)

    best_routes = None
    best_dist = float('inf')

    # 1. Nearest neighbor / greedy insertion
    if time.time() < deadline - 30:
        routes = _nearest_neighbor_construct(instance)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_routes = [r[:] for r in routes]
            best_dist = dist

    # 2. Savings heuristic
    if time.time() < deadline - 30:
        routes = _savings_construct(instance)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_routes = [r[:] for r in routes]
            best_dist = dist

    # 3. Cluster-based construction with multi-start
    if time.time() < deadline - 40:
        routes = _cluster_based_construct(instance, seed=42)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_routes = [r[:] for r in routes]
            best_dist = dist

    # 4. Dijkstra/greedy cluster construction
    if time.time() < deadline - 30:
        routes = _dijkstra_route_construct(instance)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_routes = [r[:] for r in routes]
            best_dist = dist

    if best_routes is None:
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[v] for v in customers]
        best_dist = _compute_total_distance(best_routes, dist_matrix)

    # Optimize each route with 2-opt and OR-opt
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    stations_set = set(instance["stations"])

    for r_idx in range(len(best_routes)):
        if time.time() > deadline - 10:
            break
        best_routes[r_idx] = _two_opt_route(
            best_routes[r_idx], dist_matrix, instance["n_customers"],
            stations_set, instance["battery"], instance["speed"],
            instance["recharge_rate"], instance["horizon"],
            demands, ready, due, service, max_iters=25
        )
        if time.time() > deadline - 10:
            break
        best_routes[r_idx] = _or_opt_route(
            best_routes[r_idx], dist_matrix, instance["n_customers"],
            stations_set, instance["battery"], instance["speed"],
            instance["recharge_rate"], instance["horizon"],
            demands, ready, due, service, max_iters=15
        )

    # Inter-route search
    if time.time() < deadline - 15:
        new_routes = _inter_route_search(best_routes, dist_matrix, instance, deadline - 5)
        new_dist = _compute_total_distance(new_routes, dist_matrix)
        if new_dist < best_dist:
            best_routes = new_routes
            best_dist = new_dist

    # Simulated annealing
    if time.time() < deadline - 10:
        new_routes = _simulated_annealing(best_routes, dist_matrix, instance, deadline - 2)
        new_dist = _compute_total_distance(new_routes, dist_matrix)
        if new_dist < best_dist:
            best_routes = new_routes
            best_dist = new_dist

    # Second round of intra-route optimization after inter-route moves
    if time.time() < deadline - 8:
        for r_idx in range(len(best_routes)):
            if time.time() > deadline - 5:
                break
            polished = _two_opt_route(
                best_routes[r_idx], dist_matrix, instance["n_customers"],
                stations_set, instance["battery"], instance["speed"],
                instance["recharge_rate"], instance["horizon"],
                demands, ready, due, service, max_iters=30
            )
            if _compute_total_distance([polished], dist_matrix) < _compute_total_distance([best_routes[r_idx]], dist_matrix):
                best_routes[r_idx] = polished

    # High-precision polish: OR-opt with more iterations
    if time.time() < deadline - 5:
        for r_idx in range(len(best_routes)):
            if time.time() > deadline - 3:
                break
            polished = _or_opt_route(
                best_routes[r_idx], dist_matrix, instance["n_customers"],
                stations_set, instance["battery"], instance["speed"],
                instance["recharge_rate"], instance["horizon"],
                demands, ready, due, service, max_iters=25
            )
            if _compute_total_distance([polished], dist_matrix) < _compute_total_distance([best_routes[r_idx]], dist_matrix):
                best_routes[r_idx] = polished

    # Final station optimization pass
    if time.time() < deadline - 2:
        for r_idx in range(len(best_routes)):
            if time.time() > deadline - 1:
                break
            optimized = _optimize_stations_in_route(
                best_routes[r_idx], dist_matrix, instance["n_customers"],
                stations_set, instance["battery"], instance["speed"],
                instance["recharge_rate"], instance["horizon"],
                demands, ready, due, service
            )
            if _compute_total_distance([optimized], dist_matrix) < _compute_total_distance([best_routes[r_idx]], dist_matrix):
                best_routes[r_idx] = optimized

    # Final validation and cleanup
    valid, dist = _validate_solution(best_routes, instance)
    if not valid:
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[v] for v in customers]

    return best_routes

# EVOLVE-BLOCK-END

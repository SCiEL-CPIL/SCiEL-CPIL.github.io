"""EV routing with time windows and battery constraints - TDVRP Tabu Search with regret insertion."""
import numpy as np
import time
import random
import math

DEADLINE_SECONDS = 140.0


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


def _route_distance(route, dist_matrix):
    total = 0.0
    prev = 0
    for node in route:
        total += dist_matrix[prev, node]
        prev = node
    total += dist_matrix[prev, 0]
    return total


def _is_route_feasible_fast(route, dist_matrix, instance, n_customers, stations_set,
                             battery_max, speed, recharge_rate, horizon, capacity,
                             route_load=None):
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    if route_load is None:
        total_demand = 0
        for node in route:
            if 1 <= node <= n_customers:
                total_demand += demands[node]
        if total_demand > capacity:
            return False
    elif route_load > capacity:
        return False

    battery = battery_max
    time_at = 0.0
    current_node = 0

    for node in route:
        d = dist_matrix[current_node, node]
        time_at += d / speed
        battery -= d

        if battery < -1e-9:
            return False

        if 1 <= node <= n_customers:
            if time_at < ready[node]:
                time_at = ready[node]
            if time_at > due[node] + 1e-9:
                return False
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

    if battery < -1e-9:
        return False
    if time_at > horizon + 1e-9:
        return False

    return True


def _regret_insertion_construct(instance, dist_matrix, n_customers, stations_set,
                                  battery_max, speed, recharge_rate, horizon, capacity,
                                  seed=0):
    """Construct routes using regret-based insertion."""
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    # Start with each customer as a separate route
    routes = [[i] for i in range(1, n_customers + 1)]
    route_loads = [demands[i] for i in range(1, n_customers + 1)]

    # Compute regret for each pair
    customers = list(range(1, n_customers + 1))
    n = len(customers)

    # For each customer, compute the best and second-best insertion cost
    regret_list = []
    for i in range(1, n_customers + 1):
        best_cost = float('inf')
        second_best_cost = float('inf')
        best_j = -1
        for j in range(1, n_customers + 1):
            if i == j:
                continue
            # Cost of inserting i into j's route (savings approach)
            cost = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
            if cost < best_cost:
                second_best_cost = best_cost
                best_cost = cost
                best_j = j
            elif cost < second_best_cost:
                second_best_cost = cost

        regret = second_best_cost - best_cost
        regret_list.append((regret, i))

    # Sort by regret descending
    regret_list.sort(key=lambda x: x[0], reverse=True)

    cust_to_route = {i: i - 1 for i in range(1, n_customers + 1)}

    for regret, i in regret_list:
        ri = cust_to_route[i]
        route_i = routes[ri]
        if not route_i:
            continue

        # Try to merge i with the best candidate
        best_j = -1
        best_savings = -float('inf')
        for j in range(1, n_customers + 1):
            if i == j:
                continue
            rj = cust_to_route[j]
            if rj == ri:
                continue
            route_j = routes[rj]
            if not route_j:
                continue

            savings = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]

            # Try all merge options
            merged = None
            if route_i[-1] == i and route_j[0] == j:
                merged = route_i + route_j
            elif route_i[0] == i and route_j[-1] == j:
                merged = route_i + route_j[::-1]
            elif route_i[-1] == i and route_j[-1] == j:
                merged = route_i + route_j[::-1]
            elif route_i[0] == i and route_j[0] == j:
                merged = route_i[::-1] + route_j

            if merged is None:
                continue

            if route_loads[ri] + route_loads[rj] > capacity:
                continue

            if _is_route_feasible_fast(merged, dist_matrix, instance, n_customers,
                                        stations_set, battery_max, speed, recharge_rate,
                                        horizon, capacity,
                                        route_load=route_loads[ri] + route_loads[rj]):
                if savings > best_savings:
                    best_savings = savings
                    best_j = j

        if best_j != -1:
            rj = cust_to_route[best_j]
            route_i = routes[ri]
            route_j = routes[rj]

            merged = None
            if route_i[-1] == i and route_j[0] == best_j:
                merged = route_i + route_j
            elif route_i[0] == i and route_j[-1] == best_j:
                merged = route_i + route_j[::-1]
            elif route_i[-1] == i and route_j[-1] == best_j:
                merged = route_i + route_j[::-1]
            elif route_i[0] == i and route_j[0] == best_j:
                merged = route_i[::-1] + route_j

            if merged is not None:
                routes[ri] = merged
                routes[rj] = []
                route_loads[ri] += route_loads[rj]

                for cust in merged:
                    if 1 <= cust <= n_customers:
                        cust_to_route[cust] = ri

    valid_routes = [r for r in routes if r]
    return valid_routes


def _savings_construct(instance, dist_matrix, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity,
                       seed=0):
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]

    routes = [[i] for i in range(1, n_customers + 1)]
    route_loads = [demands[i] for i in range(1, n_customers + 1)]

    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
            if s > 1e-6:
                tw_penalty = abs(ready[i] - ready[j]) * 0.01
                savings.append((s - tw_penalty, i, j))

    savings.sort(key=lambda x: x[0], reverse=True)

    cust_to_route = {i: i - 1 for i in range(1, n_customers + 1)}

    for s, i, j in savings:
        ri = cust_to_route[i]
        rj = cust_to_route[j]

        if ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        if not route_i or not route_j:
            continue

        merged = None
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        elif route_i[0] == i and route_j[-1] == j:
            merged = route_i + route_j[::-1]
        elif route_i[-1] == i and route_j[-1] == j:
            merged = route_i + route_j[::-1]
        elif route_i[0] == i and route_j[0] == j:
            merged = route_i[::-1] + route_j
        else:
            continue

        if merged is None:
            continue

        if route_loads[ri] + route_loads[rj] > capacity:
            continue

        if _is_route_feasible_fast(merged, dist_matrix, instance, n_customers,
                                    stations_set, battery_max, speed, recharge_rate,
                                    horizon, capacity,
                                    route_load=route_loads[ri] + route_loads[rj]):
            routes[ri] = merged
            routes[rj] = []
            route_loads[ri] += route_loads[rj]

            for cust in merged:
                if 1 <= cust <= n_customers:
                    cust_to_route[cust] = ri

    valid_routes = [r for r in routes if r]
    return valid_routes


def _construct_nearest_neighbor(instance, dist_matrix, n_customers, stations_set,
                                  battery_max, speed, recharge_rate, horizon, capacity,
                                  seed=0, strategy="due"):
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    customers = list(range(1, n_customers + 1))
    if strategy == "due":
        customers.sort(key=lambda c: due[c])
    else:
        random.shuffle(customers)

    routes = []
    unserved = set(customers)

    while unserved:
        route = []
        load = 0
        current = 0
        t = 0.0
        battery = battery_max

        while unserved:
            best_cust = None
            best_dist = float('inf')
            for cust in unserved:
                if load + demands[cust] > capacity:
                    continue
                d = dist_matrix[current, cust]
                new_t = t + d / speed
                new_battery = battery - d
                if new_battery < -1e-9:
                    continue
                if new_t < ready[cust]:
                    new_t = ready[cust]
                if new_t > due[cust] + 1e-9:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_cust = cust

            if best_cust is None:
                break

            route.append(best_cust)
            unserved.remove(best_cust)
            load += demands[best_cust]

            d = dist_matrix[current, best_cust]
            t += d / speed
            battery -= d

            if t < ready[best_cust]:
                t = ready[best_cust]
            t += service[best_cust]
            current = best_cust

        if route:
            routes.append(route)

    return routes


def _tabu_search(routes, dist_matrix, instance, n_customers, stations_set,
                 battery_max, speed, recharge_rate, horizon, capacity, deadline,
                 tabu_tenure=7, max_iters=3000):
    """Tabu Search for TDVRP with time-dependent costs."""
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    def get_total_dist(rts):
        return _compute_total_dist(rts, dist_matrix)

    def is_feasible(route):
        return _is_route_feasible_fast(route, dist_matrix, instance, n_customers, stations_set,
                                       battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_routes = [r[:] for r in routes]
    current_dist = get_total_dist(current_routes)
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist

    # Tabu list: (route_idx, pos) -> iteration when it becomes non-tabu
    tabu_list = {}
    current_iter = 0

    start_time = time.time()
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        current_iter = iteration

        # Generate neighborhood
        neighbors = []

        # 1. Relocate moves
        for r1 in range(len(current_routes)):
            if len(current_routes[r1]) == 0:
                continue
            for pos1 in range(len(current_routes[r1])):
                cust = current_routes[r1][pos1]
                if not (1 <= cust <= n_customers):
                    continue
                for r2 in range(len(current_routes)):
                    if r2 == r1:
                        continue
                    if route_load(current_routes[r2]) + demands[cust] > capacity:
                        continue
                    for pos2 in range(len(current_routes[r2]) + 1):
                        # Check if this move is tabu
                        move_key = (r1, pos1, r2, pos2)
                        if move_key in tabu_list and tabu_list[move_key] > current_iter:
                            continue

                        new_r1 = current_routes[r1][:pos1] + current_routes[r1][pos1+1:]
                        new_r2 = current_routes[r2][:pos2] + [cust] + current_routes[r2][pos2:]

                        if is_feasible(new_r1) and is_feasible(new_r2):
                            new_routes = [r[:] for r in current_routes]
                            new_routes[r1] = new_r1
                            new_routes[r2] = new_r2
                            new_routes = [r for r in new_routes if len(r) > 0]

                            # Validate all customers served exactly once
                            served = set()
                            valid = True
                            for route in new_routes:
                                for node in route:
                                    if 1 <= node <= n_customers:
                                        if node in served:
                                            valid = False
                                            break
                                        served.add(node)
                                if not valid:
                                    break
                            if valid and served == set(range(1, n_customers + 1)):
                                new_dist = get_total_dist(new_routes)
                                delta = new_dist - current_dist
                                neighbors.append((delta, move_key, new_routes))

        # 2. Swap moves
        for r1 in range(len(current_routes)):
            if len(current_routes[r1]) == 0:
                continue
            for pos1 in range(len(current_routes[r1])):
                cust1 = current_routes[r1][pos1]
                if not (1 <= cust1 <= n_customers):
                    continue
                for r2 in range(len(current_routes)):
                    if r2 == r1:
                        continue
                    if len(current_routes[r2]) == 0:
                        continue
                    for pos2 in range(len(current_routes[r2])):
                        cust2 = current_routes[r2][pos2]
                        if not (1 <= cust2 <= n_customers):
                            continue
                        if cust1 == cust2:
                            continue

                        load1 = route_load(current_routes[r1])
                        load2 = route_load(current_routes[r2])
                        if load1 - demands[cust1] + demands[cust2] > capacity:
                            continue
                        if load2 - demands[cust2] + demands[cust1] > capacity:
                            continue

                        move_key = (r1, pos1, r2, pos2, 'swap')
                        if move_key in tabu_list and tabu_list[move_key] > current_iter:
                            continue

                        new_r1 = current_routes[r1][:]
                        new_r2 = current_routes[r2][:]
                        new_r1[pos1] = cust2
                        new_r2[pos2] = cust1

                        if is_feasible(new_r1) and is_feasible(new_r2):
                            new_routes = [r[:] for r in current_routes]
                            new_routes[r1] = new_r1
                            new_routes[r2] = new_r2
                            new_routes = [r for r in new_routes if len(r) > 0]

                            served = set()
                            valid = True
                            for route in new_routes:
                                for node in route:
                                    if 1 <= node <= n_customers:
                                        if node in served:
                                            valid = False
                                            break
                                        served.add(node)
                                if not valid:
                                    break
                            if valid and served == set(range(1, n_customers + 1)):
                                new_dist = get_total_dist(new_routes)
                                delta = new_dist - current_dist
                                neighbors.append((delta, move_key, new_routes))

        # 3. 2-opt within a route
        for r in range(len(current_routes)):
            if len(current_routes[r]) < 2:
                continue
            route = current_routes[r]
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if is_feasible(new_route):
                        move_key = (r, i, j, '2opt')
                        if move_key in tabu_list and tabu_list[move_key] > current_iter:
                            continue

                        new_routes = [x[:] for x in current_routes]
                        new_routes[r] = new_route
                        new_routes = [x for x in new_routes if len(x) > 0]

                        served = set()
                        valid = True
                        for route2 in new_routes:
                            for node in route2:
                                if 1 <= node <= n_customers:
                                    if node in served:
                                        valid = False
                                        break
                                    served.add(node)
                            if not valid:
                                break
                        if valid and served == set(range(1, n_customers + 1)):
                            new_dist = get_total_dist(new_routes)
                            delta = new_dist - current_dist
                            neighbors.append((delta, move_key, new_routes))

        if not neighbors:
            break

        # Select best neighbor
        neighbors.sort(key=lambda x: x[0])
        best_delta, best_move, best_new_routes = neighbors[0]

        # Aspiration criterion: accept if it improves best
        if best_delta < best_dist - current_dist or best_delta < 0:
            current_routes = best_new_routes
            current_dist = current_dist + best_delta

            # Update tabu list
            if isinstance(best_move, tuple) and len(best_move) >= 2:
                tabu_list[best_move] = current_iter + tabu_tenure

            if current_dist < best_dist:
                best_dist = current_dist
                best_routes = [r[:] for r in current_routes]
        else:
            # No improvement, try next best
            found = False
            for delta, move, new_rts in neighbors[1:]:
                if delta < 0:
                    current_routes = new_rts
                    current_dist = current_dist + delta
                    if isinstance(move, tuple) and len(move) >= 2:
                        tabu_list[move] = current_iter + tabu_tenure
                    if current_dist < best_dist:
                        best_dist = current_dist
                        best_routes = [r[:] for r in current_routes]
                    found = True
                    break
            if not found:
                # Diversify: restart from best
                if random.random() < 0.1:
                    current_routes = [r[:] for r in best_routes]
                    current_dist = best_dist

        # Clean up old tabu entries
        if current_iter % 50 == 0:
            tabu_list = {k: v for k, v in tabu_list.items() if v > current_iter}

    return best_routes


def _local_search(routes, dist_matrix, instance, n_customers, stations_set,
                  battery_max, speed, recharge_rate, horizon, capacity, deadline):
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_dist(rts, dist_matrix)

    def is_feasible(route):
        return _is_route_feasible_fast(route, dist_matrix, instance, n_customers, stations_set,
                                       battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_dist = get_total_dist(routes)

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for r_idx in range(len(routes)):
            if time.time() > deadline - 5:
                break
            route = routes[r_idx]
            if len(route) < 2:
                continue

            # 2-opt
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if is_feasible(new_route):
                        new_dist = get_total_dist([new_route] + routes[:r_idx] + routes[r_idx+1:])
                        if new_dist < current_dist - 1e-9:
                            routes[r_idx] = new_route
                            current_dist = new_dist
                            improved = True
                            break
                if improved:
                    break

            if improved:
                continue

            # Or-opt
            for k in [1, 2, 3]:
                if len(route) <= k:
                    continue
                for i in range(len(route) - k + 1):
                    segment = route[i:i+k]
                    rest = route[:i] + route[i+k:]
                    for j in range(len(rest) + 1):
                        new_route = rest[:j] + segment + rest[j:]
                        if new_route == route:
                            continue
                        if is_feasible(new_route):
                            new_dist = get_total_dist([new_route] + routes[:r_idx] + routes[r_idx+1:])
                            if new_dist < current_dist - 1e-9:
                                routes[r_idx] = new_route
                                current_dist = new_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    # Inter-route moves
    for _pass in range(10):
        if time.time() > deadline - 2:
            break
        pass_improved = False
        for r1 in range(len(routes)):
            if time.time() > deadline - 2:
                break
            for pos1 in range(len(routes[r1])):
                cust = routes[r1][pos1]
                if not (1 <= cust <= n_customers):
                    continue
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    if route_load(routes[r2]) + demands[cust] > capacity:
                        continue
                    for pos2 in range(len(routes[r2]) + 1):
                        new_r1 = routes[r1][:pos1] + routes[r1][pos1+1:]
                        new_r2 = routes[r2][:pos2] + [cust] + routes[r2][pos2:]

                        if is_feasible(new_r1) and is_feasible(new_r2):
                            old_dist_part = _compute_total_dist([routes[r1], routes[r2]], dist_matrix)
                            new_dist_part = _compute_total_dist([new_r1, new_r2], dist_matrix)
                            if new_dist_part < old_dist_part - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                current_dist = get_total_dist(routes)
                                pass_improved = True
                                break
                    if pass_improved:
                        break
                if pass_improved:
                    break
            if pass_improved:
                break

        if not pass_improved:
            break

    routes = [r for r in routes if len(r) > 0]
    return routes


def _lns_search(routes, dist_matrix, instance, n_customers, stations_set,
                battery_max, speed, recharge_rate, horizon, capacity, deadline):
    """Large Neighborhood Search: destroy and repair."""
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_dist(rts, dist_matrix)

    def is_feasible(route):
        return _is_route_feasible_fast(route, dist_matrix, instance, n_customers, stations_set,
                                       battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_routes = [r[:] for r in routes]
    current_dist = get_total_dist(current_routes)
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist

    start_temp = 3.0
    end_temp = 0.1
    max_iters = 500
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    temp = start_temp

    start_time = time.time()
    for iteration in range(max_iters):
        if time.time() > deadline:
            break

        # Destroy: remove random subset of customers from a random route
        new_routes = [r[:] for r in current_routes]
        r_idx = random.randint(0, len(new_routes) - 1)
        route = new_routes[r_idx]

        if len(route) < 3:
            continue

        # Remove 2-4 customers
        n_remove = random.randint(2, min(4, len(route)))
        removed = random.sample([c for c in route if 1 <= c <= n_customers], n_remove)
        remaining = [c for c in route if c not in removed]

        # Repair: try to re-insert removed customers greedily
        success = True
        for cust in removed:
            best_target = None
            best_target_dist = float('inf')
            for t_idx in range(len(new_routes)):
                if route_load(new_routes[t_idx]) + demands[cust] > capacity:
                    continue
                for pos in range(len(new_routes[t_idx]) + 1):
                    candidate = new_routes[t_idx][:pos] + [cust] + new_routes[t_idx][pos:]
                    if is_feasible(candidate):
                        d = _route_distance(candidate, dist_matrix)
                        if d < best_target_dist:
                            best_target_dist = d
                            best_target = (t_idx, pos)
            if best_target is not None:
                t_idx, pos = best_target
                new_routes[t_idx] = new_routes[t_idx][:pos] + [cust] + new_routes[t_idx][pos:]
            else:
                success = False
                break

        if not success:
            temp *= cooling_rate
            continue

        new_routes[r_idx] = remaining
        new_routes = [r for r in new_routes if len(r) > 0]

        # Validate
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


def _compute_total_dist(routes, dist_matrix):
    total = 0.0
    for route in routes:
        current = 0
        for node in route:
            total += dist_matrix[current, node]
            current = node
        total += dist_matrix[current, 0]
    return total


def plan(instance):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

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

    # Phase 1: Multiple construction heuristics
    # 1a: Regret-based insertion
    for i in range(5):
        if time.time() > deadline - 60:
            break
        routes = _regret_insertion_construct(instance, dist_matrix, n_customers, stations_set,
                                              battery_max, speed, recharge_rate, horizon, capacity,
                                              seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # 1b: Savings-based construction
    for i in range(8):
        if time.time() > deadline - 55:
            break
        routes = _savings_construct(instance, dist_matrix, n_customers, stations_set,
                                    battery_max, speed, recharge_rate, horizon, capacity,
                                    seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # 1c: Nearest neighbor
    for i in range(3):
        if time.time() > deadline - 50:
            break
        routes = _construct_nearest_neighbor(instance, dist_matrix, n_customers, stations_set,
                                              battery_max, speed, recharge_rate, horizon,
                                              capacity, seed=i, strategy="due")
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # Fallback
    if best_routes is None:
        best_routes = [[v] for v in range(1, n_customers + 1)]
        best_dist = _compute_total_dist(best_routes, dist_matrix)

    # Phase 2: Tabu Search (main optimization)
    if time.time() < deadline - 40:
        tabu_routes = _tabu_search(best_routes, dist_matrix, instance, n_customers,
                                   stations_set, battery_max, speed, recharge_rate,
                                   horizon, capacity, deadline - 35,
                                   tabu_tenure=7, max_iters=2000)
        valid, dist = _validate_solution(tabu_routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = tabu_routes

    # Phase 3: Local Search polish
    if time.time() < deadline - 20:
        improved = _local_search(best_routes, dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, deadline - 15)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Phase 4: LNS
    if time.time() < deadline - 25:
        lns_routes = _lns_search(best_routes, dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, deadline - 15)
        valid, dist = _validate_solution(lns_routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = lns_routes

    # Phase 5: Final polish
    if time.time() < deadline - 5:
        improved = _local_search(best_routes, dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, deadline - 2)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Final validation
    valid, dist = _validate_solution(best_routes, instance)
    if not valid:
        best_routes = [[v] for v in range(1, n_customers + 1)]

    return best_routes

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import permutations


def _dist(coords, i, j):
    return math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])


def _simulate_route(route, instance):
    """Simulate a single route. Returns (total_distance, valid, end_time, end_battery)."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]

    depot = 0
    time_t = 0.0
    battery = battery_max
    load = 0
    total_dist = 0.0
    prev = depot

    for node in route:
        d = _dist(coords, prev, node)
        time_t += d / speed
        battery -= d
        total_dist += d

        if battery < -1e-9:
            return total_dist, False, time_t, battery

        if node in stations:
            # Charging station: recharge
            energy = battery_max - battery
            time_t += energy * recharge_rate
            battery = battery_max
        else:
            # Customer node
            if node > n_customers:
                return total_dist, False, time_t, battery
            # Time window
            if time_t < ready[node]:
                time_t = ready[node]
            if time_t > due[node] + 1e-9:
                return total_dist, False, time_t, battery
            # Service
            time_t += service[node]
            # Load
            load += demands[node]
            if load > capacity + 1e-9:
                return total_dist, False, time_t, battery

        prev = node

    # Return to depot
    d = _dist(coords, prev, depot)
    time_t += d / speed
    battery -= d
    total_dist += d

    if battery < -1e-9:
        return total_dist, False, time_t, battery
    if time_t > horizon + 1e-9:
        return total_dist, False, time_t, battery

    return total_dist, True, time_t, battery


def _total_distance(routes, instance):
    coords = instance["coords"]
    depot = 0
    total = 0.0
    for route in routes:
        prev = depot
        for node in route:
            total += _dist(coords, prev, node)
            prev = node
        total += _dist(coords, prev, depot)
    return total


def _is_valid_solution(routes, instance):
    n_customers = instance["n_customers"]
    served = set()
    for route in routes:
        for node in route:
            if node > n_customers:
                return False
            if node in served:
                return False
            served.add(node)
    if len(served) != n_customers:
        return False

    for route in routes:
        _, valid, _, _ = _simulate_route(route, instance)
        if not valid:
            return False
    return True


def _construct_greedy(instance, seed=0):
    """Greedy construction: nearest neighbor with time window feasibility."""
    random.seed(seed)
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]

    depot = 0
    unserved = set(range(1, n_customers + 1))
    routes = []

    while unserved:
        # Start a new route from depot
        route = []
        time_t = 0.0
        battery = battery_max
        load = 0
        prev = depot

        while unserved:
            # Find best next customer
            best_node = None
            best_cost = float('inf')

            for cust in list(unserved):
                d = _dist(coords, prev, cust)
                t_arrive = time_t + d / speed
                b_arrive = battery - d

                if b_arrive < -1e-9:
                    continue
                if t_arrive < ready[cust]:
                    t_arrive = ready[cust]
                if t_arrive > due[cust] + 1e-9:
                    continue
                t_after_service = t_arrive + service[cust]

                # Check if we can return to depot after this
                d_back = _dist(coords, cust, depot)
                t_return = t_after_service + d_back / speed
                b_return = b_arrive - d_back
                if t_return > horizon + 1e-9 or b_return < -1e-9:
                    continue
                if load + demands[cust] > capacity + 1e-9:
                    continue

                # Cost: distance
                cost = d
                if cost < best_cost:
                    best_cost = cost
                    best_node = cust

            if best_node is None:
                break

            unserved.discard(best_node)
            route.append(best_node)
            time_t += _dist(coords, prev, best_node) / speed
            battery -= _dist(coords, prev, best_node)
            if time_t < ready[best_node]:
                time_t = ready[best_node]
            time_t += service[best_node]
            load += demands[best_node]
            prev = best_node

        if route:
            routes.append(route)

    return routes


def _insert_station_if_needed(route, instance):
    """Insert charging stations into route where needed to maintain battery >= 0."""
    coords = instance["coords"]
    stations = instance["stations"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    n_customers = instance["n_customers"]

    depot = 0
    # Check if route is feasible without stations
    _, valid, _, _ = _simulate_route(route, instance)
    if valid:
        return route

    # Try inserting stations at various positions
    # First, check battery feasibility only
    time_t = 0.0
    battery = battery_max
    prev = depot
    new_route = []

    for node in route:
        d = _dist(coords, prev, node)
        battery -= d
        time_t += d / speed

        if battery < -1e-9:
            # Need to insert a station before this node
            # Find nearest station to prev that we can reach
            best_station = None
            best_dist = float('inf')
            for st in stations:
                d_st = _dist(coords, prev, st)
                if battery + d_st >= -1e-9:  # Can reach station
                    if d_st < best_dist:
                        best_dist = d_st
                        best_station = st

            if best_station is not None:
                new_route.append(best_station)
                battery = battery_max
                time_t += best_dist / speed
                energy = battery_max - battery
                time_t += energy * recharge_rate
                battery = battery_max
                prev = best_station
                # Continue to node
                d2 = _dist(coords, prev, node)
                battery -= d2
                time_t += d2 / speed
            else:
                # Cannot fix
                return route

        new_route.append(node)
        if node not in stations:
            if time_t < ready[node]:
                time_t = ready[node]
            time_t += service[node]
        else:
            energy = battery_max - battery
            time_t += energy * recharge_rate
            battery = battery_max
        prev = node

    # Check return
    d = _dist(coords, prev, depot)
    battery -= d
    time_t += d / speed

    if battery < -1e-9:
        # Need station before depot
        best_station = None
        best_dist = float('inf')
        for st in stations:
            d_st = _dist(coords, prev, st)
            if battery + d_st >= -1e-9:
                if d_st < best_dist:
                    best_dist = d_st
                    best_station = st
        if best_station is not None:
            new_route.append(best_station)

    return new_route


def _local_search(routes, instance, max_iters=200):
    """2-opt and swap local search."""
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    best_routes = [list(r) for r in routes]
    best_dist = _total_distance(best_routes, instance)

    for it in range(max_iters):
        improved = False

        # 2-opt within each route
        for ri in range(len(best_routes)):
            route = best_routes[ri]
            for i in range(len(route)):
                for j in range(i + 2, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    # Check validity
                    valid = True
                    _, v, _, _ = _simulate_route(new_route, instance)
                    if not v:
                        continue
                    # Check all customers still served exactly once
                    all_nodes = []
                    for r in best_routes:
                        if r is best_routes[ri]:
                            all_nodes.extend(new_route)
                        else:
                            all_nodes.extend(r)
                    if len(all_nodes) != n_customers or len(set(all_nodes)) != n_customers:
                        continue
                    if any(n > n_customers for n in all_nodes):
                        continue
                    d = _total_distance([new_route if idx == ri else best_routes[idx] for idx in range(len(best_routes))], instance)
                    if d < best_dist - 1e-9:
                        best_routes[ri] = new_route
                        best_dist = d
                        improved = True

        # Swap between routes
        for ri in range(len(best_routes)):
            for rj in range(ri + 1, len(best_routes)):
                for i in range(len(best_routes[ri])):
                    for j in range(len(best_routes[rj])):
                        a = best_routes[ri][i]
                        b = best_routes[rj][j]
                        if a == b:
                            continue
                        new_ri = list(best_routes[ri])
                        new_rj = list(best_routes[rj])
                        new_ri[i] = b
                        new_rj[j] = a
                        # Check validity
                        valid = True
                        _, v1, _, _ = _simulate_route(new_ri, instance)
                        if not v1:
                            valid = False
                        _, v2, _, _ = _simulate_route(new_rj, instance)
                        if not v2:
                            valid = False
                        if not valid:
                            continue
                        d = _total_distance([new_ri if idx == ri else (new_rj if idx == rj else best_routes[idx]) for idx in range(len(best_routes))], instance)
                        if d < best_dist - 1e-9:
                            best_routes[ri] = new_ri
                            best_routes[rj] = new_rj
                            best_dist = d
                            improved = True

        if not improved:
            break

    return best_routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    n_customers = instance["n_customers"]
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Generate multiple initial solutions
    best_routes = None
    best_dist = float('inf')

    # Try different random seeds for greedy construction
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        routes = _construct_greedy(instance, seed=seed)
        # Insert stations if needed
        new_routes = []
        for r in routes:
            new_routes.append(_insert_station_if_needed(r, instance))
        if _is_valid_solution(new_routes, instance):
            d = _total_distance(new_routes, instance)
            if d < best_dist:
                best_dist = d
                best_routes = new_routes

    if best_routes is None:
        # Fallback: one customer per route
        customers = list(range(1, n_customers + 1))
        customers.sort(key=lambda v: due[v])
        return [[v] for v in customers]

    # Local search
    if time.time() < deadline - 10:
        best_routes = _local_search(best_routes, instance, max_iters=100)
        if _is_valid_solution(best_routes, instance):
            best_dist = _total_distance(best_routes, instance)

    # Additional local search passes if time remains
    while time.time() < deadline - 5:
        improved = False
        # Try more local search
        new_routes = _local_search(best_routes, instance, max_iters=50)
        if _is_valid_solution(new_routes, instance):
            d = _total_distance(new_routes, instance)
            if d < best_dist - 1e-9:
                best_dist = d
                best_routes = new_routes
                improved = True
        if not improved:
            break

    return best_routes
# EVOLVE-BLOCK-END

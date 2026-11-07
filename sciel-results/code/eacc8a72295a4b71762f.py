"""EV routing with time windows – improved heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt, inf

def _dist(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def _build_dist(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = _dist(coords[i], coords[j])
    return D

def _simulate(route, instance, D):
    """Return (total_distance, feasible, final_time, final_battery)."""
    coords = instance["coords"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]

    t = 0.0
    batt = battery_max
    load = 0
    total_dist = 0.0
    prev = 0
    served = 0

    for node in route:
        d = D[prev, node]
        total_dist += d
        t += d / speed
        batt -= d
        if batt < -1e-9:
            return total_dist, False, t, batt
        prev = node
        if node == 0:
            batt = battery_max
            continue
        if node in stations:
            # recharge
            energy_needed = battery_max - batt
            t += energy_needed * recharge_rate
            batt = battery_max
            continue
        # customer
        if node > n_customers:
            continue  # shouldn't happen
        if t < ready[node]:
            t = ready[node]
        if t > due[node] + 1e-9:
            return total_dist, False, t, batt
        t += service[node]
        load += demands[node]
        if load > capacity + 1e-9:
            return total_dist, False, t, batt
        served += 1

    # return to depot
    d = D[prev, 0]
    total_dist += d
    t += d / speed
    batt -= d
    if batt < -1e-9:
        return total_dist, False, t, batt
    if t > horizon + 1e-9:
        return total_dist, False, t, batt

    return total_dist, True, t, batt

def _nearest_neighbor_route(unvisited, instance, D, start=0):
    """Build a route using nearest-neighbor with time window feasibility check."""
    coords = instance["coords"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]

    unvisited = list(unvisited)
    route = []
    t = 0.0
    batt = battery_max
    load = 0
    prev = start

    while unvisited:
        best_node = None
        best_dist = inf
        best_t = inf
        best_batt = inf
        for node in unvisited:
            d = D[prev, node]
            new_t = t + d / speed
            new_batt = batt - d
            if new_batt < -1e-9:
                continue
            # check if we can serve this customer
            if node in stations:
                continue
            if new_t < ready[node]:
                new_t = ready[node]
            if new_t > due[node] + 1e-9:
                continue
            new_load = load + demands[node]
            if new_load > capacity + 1e-9:
                continue
            # check return to depot
            d_ret = D[node, 0]
            final_t = new_t + service[node] + d_ret / speed
            final_batt = new_batt - d_ret
            if final_t > horizon + 1e-9 or final_batt < -1e-9:
                continue
            # prefer nearest
            if d < best_dist:
                best_dist = d
                best_node = node
                best_t = new_t
                best_batt = new_batt

        if best_node is None:
            # try adding a station then a customer
            for st in stations:
                d_st = D[prev, st]
                new_t_st = t + d_st / speed
                new_batt_st = batt - d_st
                if new_batt_st < -1e-9:
                    continue
                new_t_st += (battery_max - new_batt_st) * recharge_rate
                new_batt_st = battery_max
                for node in unvisited:
                    d = D[st, node]
                    new_t = new_t_st + d / speed
                    new_batt = new_batt_st - d
                    if new_batt < -1e-9:
                        continue
                    if node in stations:
                        continue
                    if new_t < ready[node]:
                        new_t = ready[node]
                    if new_t > due[node] + 1e-9:
                        continue
                    new_load = load + demands[node]
                    if new_load > capacity + 1e-9:
                        continue
                    d_ret = D[node, 0]
                    final_t = new_t + service[node] + d_ret / speed
                    final_batt = new_batt - d_ret
                    if final_t > horizon + 1e-9 or final_batt < -1e-9:
                        continue
                    if d < best_dist:
                        best_dist = d
                        best_node = node
                        # mark that we need to insert station
                        best_t = new_t
                        best_batt = new_batt
                        # we'll handle station insertion in the route building
                        # For simplicity, just pick the customer for now
                        break
                if best_node is not None:
                    break

        if best_node is None:
            break

        route.append(best_node)
        unvisited.remove(best_node)
        d = D[prev, best_node]
        t += d / speed
        batt -= d
        if best_node in stations:
            t += (battery_max - batt) * recharge_rate
            batt = battery_max
        else:
            if t < ready[best_node]:
                t = ready[best_node]
            t += service[best_node]
            load += demands[best_node]
        prev = best_node

    return route

def _two_opt(route, instance, D, max_iter=50):
    """Simple 2-opt improvement for a single route."""
    coords = instance["coords"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]

    if len(route) < 2:
        return route

    best_route = route[:]
    best_dist, _, _, _ = _simulate(best_route, instance, D)
    if not best_dist >= 0:
        return route

    for _ in range(max_iter):
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                total_dist, feasible, _, _ = _simulate(new_route, instance, D)
                if feasible and total_dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = total_dist
                    improved = True
        if not improved:
            break

    return best_route

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    coords = instance["coords"]
    D = _build_dist(coords)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = instance["stations"]

    # Sort customers by due time (earliest first)
    customers.sort(key=lambda v: instance["due"][v])

    # Greedy insertion: build routes one by one
    unvisited = set(customers)
    routes = []

    while unvisited:
        if time.time() > deadline:
            break

        # Try to build a route starting from the earliest-due unvisited customer
        best_seed = None
        best_seed_due = inf
        for c in unvisited:
            if instance["due"][c] < best_seed_due:
                best_seed_due = instance["due"][c]
                best_seed = c

        if best_seed is None:
            break

        # Build route using nearest neighbor starting with best_seed
        route = [best_seed]
        unvisited.discard(best_seed)

        # Keep adding customers
        while unvisited:
            if time.time() > deadline:
                break

            # Find best customer to add
            best_node = None
            best_dist = inf
            for node in unvisited:
                d = D[route[-1], node]
                # Quick feasibility check
                # Simulate adding this node
                temp_route = route + [node]
                total_dist, feasible, _, _ = _simulate(temp_route, instance, D)
                if feasible and d < best_dist:
                    best_dist = d
                    best_node = node

            if best_node is None:
                break

            route.append(best_node)
            unvisited.discard(best_node)

        # Apply 2-opt
        if time.time() < deadline:
            route = _two_opt(route, instance, D, max_iter=20)

        routes.append(route)

    # If any customers remain (due to time), add them one per route
    if unvisited:
        for c in unvisited:
            routes.append([c])

    # Final verification
    served = set()
    for r in routes:
        for node in r:
            if 1 <= node <= n_customers:
                if node in served:
                    # duplicate – shouldn't happen
                    pass
                served.add(node)

    # Check all served
    if served != set(customers):
        missing = set(customers) - served
        for c in missing:
            routes.append([c])

    return routes
# EVOLVE-BLOCK-END

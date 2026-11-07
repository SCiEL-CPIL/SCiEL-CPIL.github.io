# EVOLVE-BLOCK-START
import math
import time
import numpy as np


def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_dist(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            v = math.hypot(xi - xj, yi - yj)
            d[i, j] = v
            d[j, i] = v
    return d


def _check_route(route, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed=1.0):
    """Return (feasible, total_distance, final_time, final_battery)."""
    if not route:
        return True, 0.0, 0.0, battery_cap
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    depot = 0

    t = 0.0
    bat = battery_cap
    load = 0.0
    total_dist = 0.0
    prev = depot

    for node in route:
        d = dist[prev][node]
        bat -= d
        t += d / speed
        total_dist += d
        if bat < -1e-9:
            return False, total_dist, t, bat
        if node in stations:
            # charging: time added = (battery - energy) * recharge_rate? 
            # Re-read: "a station adds (battery - energy) * recharge_rate time and refills"
            # I think this means: charging time = (battery_cap - bat) * recharge_rate
            # "refills" means battery becomes full.
            charge_time = (battery_cap - bat) * recharge_rate
            t += charge_time
            bat = battery_cap
        else:
            # customer
            load += demands[node]
            if load > capacity + 1e-9:
                return False, total_dist, t, bat
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return False, total_dist, t, bat
            t += service[node]
        prev = node

    # return to depot
    d = dist[prev][depot]
    bat -= d
    t += d / speed
    total_dist += d
    if bat < -1e-9:
        return False, total_dist, t, bat
    if t > horizon + 1e-9:
        return False, total_dist, t, bat
    return True, total_dist, t, bat


def _try_insert(route, pos, cust, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed=1.0):
    """Try inserting cust at position pos in route. Return (feasible, dist, t, bat)."""
    new_route = route[:pos] + [cust] + route[pos:]
    return _check_route(new_route, instance, dist, stations, capacity, battery_cap, recharge_rate, horizon, speed)


def _greedy_insertion(instance):
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    speed = 1.0

    dist = _build_dist(coords)

    # Sort customers by due time (earliest first) for better feasibility
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda c: due[c])

    # Start with a few empty routes
    max_routes = 10
    routes = [[] for _ in range(max_routes)]
    unserved = set(customers)

    # Greedy insertion: for each customer in order, find best (route, position)
    for cust in customers:
        best_score = float('inf')
        best_r = -1
        best_p = -1
        best_dist = 0.0

        for r_idx, route in enumerate(routes):
            # Try each position in this route
            for pos in range(len(route) + 1):
                feasible, rd, rt, rb = _try_insert(route, pos, cust, instance, dist, stations,
                                                    capacity, battery_cap, recharge_rate, horizon, speed)
                if feasible:
                    if rd < best_score:
                        best_score = rd
                        best_r = r_idx
                        best_p = pos
                        best_dist = rd

        if best_r >= 0:
            routes[best_r].insert(best_p, cust)
        else:
            # No existing route can take it; start a new route
            # Try to find an empty route
            placed = False
            for r_idx, route in enumerate(routes):
                if not route:
                    feasible, rd, rt, rb = _try_insert(route, 0, cust, instance, dist, stations,
                                                        capacity, battery_cap, recharge_rate, horizon, speed)
                    if feasible:
                        routes[r_idx] = [cust]
                        placed = True
                        break
            if not placed:
                # Should not happen with 10 routes and 25 customers, but safety
                routes.append([cust])
        unserved.discard(cust)

    # Remove empty routes
    routes = [r for r in routes if r]

    # 2-opt polish on each route
    for r_idx, route in enumerate(routes):
        improved = True
        while improved:
            improved = False
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, rd, rt, rb = _check_route(new_route, instance, dist, stations,
                                                         capacity, battery_cap, recharge_rate, horizon, speed)
                    old_feasible, old_rd, _, _ = _check_route(route, instance, dist, stations,
                                                              capacity, battery_cap, recharge_rate, horizon, speed)
                    if feasible and rd < old_rd - 1e-9:
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
            routes[r_idx] = route

    return routes


def plan(instance):
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    # Compute greedy insertion solution
    routes = _greedy_insertion(instance)

    # Verify feasibility of the result
    coords = instance["coords"]
    dist = _build_dist(coords)
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    all_ok = True
    for route in routes:
        feasible, _, _, _ = _check_route(route, instance, dist, stations, capacity, battery_cap,
                                         recharge_rate, horizon, speed=1.0)
        if not feasible:
            all_ok = False
            break

    if not all_ok:
        # Fallback: one customer per route, sorted by due time
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]

    return routes
# EVOLVE-BLOCK-END

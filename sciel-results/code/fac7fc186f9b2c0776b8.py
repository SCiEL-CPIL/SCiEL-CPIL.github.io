"""EV routing with time windows – constructive heuristic + local search."""
import numpy as np
import math
import heapq
import time

# EVOLVE-BLOCK-START

def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _build_dist(coords):
    n = len(coords)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
    return d

def _simulate_route(route, inst, dist):
    """Return (feasible, total_distance, total_time, violations)."""
    n_customers = inst["n_customers"]
    capacity = inst["capacity"]
    battery_max = inst["battery"]
    speed = inst["speed"]
    recharge_rate = inst["recharge_rate"]
    horizon = inst["horizon"]
    coords = inst["coords"]
    demands = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    stations = set(inst["stations"])

    total_dist = 0.0
    total_time = 0.0
    battery = battery_max
    load = 0
    violations = 0

    prev_node = 0
    cur_time = 0.0

    for node in route:
        d = dist[prev_node][node]
        total_dist += d
        travel_time = d / speed
        battery -= d
        if battery < -1e-9:
            violations += 1
            battery = 0.0
        cur_time += travel_time

        if node in stations:
            # Charging: time = (battery_max - battery) * recharge_rate
            charge_time = (battery_max - battery) * recharge_rate
            cur_time += charge_time
            battery = battery_max
        else:
            # Customer
            if node <= n_customers:
                load += demands[node]
                if load > capacity + 1e-9:
                    violations += 1
                # Time window
                if cur_time < ready[node]:
                    cur_time = ready[node]
                if cur_time > due[node] + 1e-9:
                    violations += 1
                cur_time += service[node]
        prev_node = node

    # Return to depot
    d = dist[prev_node][0]
    total_dist += d
    travel_time = d / speed
    battery -= d
    if battery < -1e-9:
        violations += 1
    cur_time += travel_time
    if cur_time > horizon + 1e-9:
        violations += 1

    return (violations == 0, total_dist, cur_time, violations)

def _is_feasible(route, inst, dist):
    feasible, _, _, _ = _simulate_route(route, inst, dist)
    return feasible

def _route_distance(route, inst, dist):
    _, total_dist, _, _ = _simulate_route(route, inst, dist)
    return total_dist

def _total_distance(routes, inst, dist):
    return sum(_route_distance(r, inst, dist) for r in routes)

def _nearest_neighbor_route(unvisited, inst, dist, start_node=0):
    """Build a route using nearest-neighbor with time-window feasibility check."""
    n_customers = inst["n_customers"]
    coords = inst["coords"]
    demands = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    stations = set(inst["stations"])
    capacity = inst["capacity"]
    battery_max = inst["battery"]
    speed = inst["speed"]
    recharge_rate = inst["recharge_rate"]
    horizon = inst["horizon"]

    route = []
    unvisited_set = set(unvisited)
    cur_node = start_node
    cur_time = 0.0
    battery = battery_max
    load = 0

    while unvisited_set:
        best_node = None
        best_dist = float('inf')
        for node in sorted(unvisited_set):
            d = dist[cur_node][node]
            if d < best_dist:
                # Check if we can reach and serve this node
                new_battery = battery - d
                new_time = cur_time + d / speed
                if new_battery < -1e-9:
                    continue
                if node in stations:
                    new_time += (battery_max - new_battery) * recharge_rate
                    new_battery = battery_max
                else:
                    if new_time < ready[node]:
                        new_time = ready[node]
                    if new_time > due[node] + 1e-9:
                        continue
                    new_time += service[node]
                    if load + demands[node] > capacity + 1e-9:
                        continue
                best_node = node
                best_dist = d
                break  # Nearest feasible

        if best_node is None:
            # Try next nearest regardless of time window (will be infeasible, but we need to assign)
            best_node = min(unvisited_set, key=lambda n: dist[cur_node][n])

        route.append(best_node)
        unvisited_set.remove(best_node)

        # Update state
        d = dist[cur_node][best_node]
        battery -= d
        cur_time += d / speed
        if best_node in stations:
            cur_time += (battery_max - battery) * recharge_rate
            battery = battery_max
        else:
            if cur_time < ready[best_node]:
                cur_time = ready[best_node]
            cur_time += service[best_node]
            load += demands[best_node]
        cur_node = best_node

    # Check return to depot
    return route

def _solve_instance(inst):
    n_customers = inst["n_customers"]
    coords = inst["coords"]
    demands = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    stations = set(inst["stations"])
    capacity = inst["capacity"]
    battery_max = inst["battery"]
    speed = inst["speed"]
    recharge_rate = inst["recharge_rate"]
    horizon = inst["horizon"]

    dist = _build_dist(coords)
    customers = list(range(1, n_customers + 1))

    # Sort customers by due time for initial construction
    customers_sorted = sorted(customers, key=lambda c: due[c])

    # Greedy insertion: build routes by inserting customers one by one
    routes = []
    remaining = list(customers_sorted)

    while remaining:
        # Try to add each remaining customer to an existing route
        inserted = False
        for cust in remaining[:]:
            for r_idx, route in enumerate(routes):
                # Try inserting cust at each position in route
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [cust] + route[pos:]
                    if _is_feasible(new_route, inst, dist):
                        routes[r_idx] = new_route
                        remaining.remove(cust)
                        inserted = True
                        break
                if inserted:
                    break
            if inserted:
                break

        if not inserted:
            # Start a new route with the next customer
            cust = remaining.pop(0)
            routes.append([cust])

    # Local search: 2-opt within each route
    improved = True
    max_iter = 50
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        for r_idx, route in enumerate(routes):
            if len(route) < 3:
                continue
            for i in range(len(route) - 2):
                for j in range(i + 2, len(route)):
                    # Reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if _is_feasible(new_route, inst, dist):
                        old_d = _route_distance(route, inst, dist)
                        new_d = _route_distance(new_route, inst, dist)
                        if new_d < old_d - 1e-9:
                            routes[r_idx] = new_route
                            improved = True
                            break
                if improved:
                    break

    # Try relocating customers between routes
    for _ in range(10):
        improved = False
        for r1_idx in range(len(routes)):
            for r2_idx in range(len(routes)):
                if r1_idx == r2_idx:
                    continue
                for c_idx, cust in enumerate(routes[r1_idx]):
                    # Try removing cust from route r1 and inserting into r2
                    route1_without = routes[r1_idx][:c_idx] + routes[r1_idx][c_idx+1:]
                    if not route1_without:
                        continue
                    if not _is_feasible(route1_without, inst, dist):
                        continue
                    for pos in range(len(routes[r2_idx]) + 1):
                        route2_new = routes[r2_idx][:pos] + [cust] + routes[r2_idx][pos:]
                        if _is_feasible(route2_new, inst, dist):
                            old_total = _route_distance(routes[r1_idx], inst, dist) + _route_distance(routes[r2_idx], inst, dist)
                            new_total = _route_distance(route1_without, inst, dist) + _route_distance(route2_new, inst, dist)
                            if new_total < old_total - 1e-9:
                                routes[r1_idx] = route1_without
                                routes[r2_idx] = route2_new
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

    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    results = []
    for inst in [instance]:
        if time.time() > deadline:
            # Fallback: one customer per route
            customers = list(range(1, inst["n_customers"] + 1))
            results.append([[c] for c in customers])
            continue
        routes = _solve_instance(inst)
        results.append(routes)

    return results

# EVOLVE-BLOCK-END

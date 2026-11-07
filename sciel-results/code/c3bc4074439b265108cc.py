"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations

def _dist(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

def _build_dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = _dist(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D

def _simulate_route(route, D, demands, ready, due, service, capacity, battery_cap,
                    speed, recharge_rate, horizon, stations, coords):
    """
    Simulate a route (list of node ids, depot implicit at start and end).
    Returns (feasible, total_distance, end_time).
    """
    if not route:
        return True, 0.0, 0.0

    t = 0.0
    battery = battery_cap
    load = 0
    total_dist = 0.0
    prev = 0  # depot

    for node in route:
        d = D[prev, node]
        total_dist += d
        t += d / speed
        battery -= d

        if battery < 0:
            return False, total_dist, t

        # If node is a customer
        if node not in stations and node != 0:
            # Check time window
            if t < ready[node]:
                t = ready[node]
            if t > due[node]:
                return False, total_dist, t

            load += demands[node]
            if load > capacity:
                return False, total_dist, t

            t += service[node]
        elif node in stations:
            # Recharge
            energy_needed = battery_cap - battery
            if energy_needed > 0:
                t += energy_needed / recharge_rate
            battery = battery_cap

        prev = node

    # Return to depot
    d = D[prev, 0]
    total_dist += d
    t += d / speed
    battery -= d

    if battery < 0:
        return False, total_dist, t
    if t > horizon:
        return False, total_dist, t

    return True, total_dist, t

def _compute_total_distance(routes, D):
    total = 0.0
    for route in routes:
        if not route:
            continue
        prev = 0
        for node in route:
            total += D[prev, node]
            prev = node
        total += D[prev, 0]
    return total

def _validate_all(routes, instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]

    D = _build_dist_matrix(coords)

    # Check every customer served exactly once
    served = []
    for route in routes:
        for node in route:
            if 1 <= node <= n_customers:
                served.append(node)
    if len(served) != n_customers:
        return False
    if len(set(served)) != n_customers:
        return False

    for route in routes:
        feasible, _, _ = _simulate_route(route, D, demands, ready, due, service,
                                         capacity, battery_cap, speed, recharge_rate,
                                         horizon, stations, coords)
        if not feasible:
            return False
    return True

def _greedy_construct(instance, D, seed=0):
    """
    Greedy construction: sort customers by due time, insert into existing routes
    or create new routes.
    """
    random.seed(seed)
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]

    customers = list(range(1, n_customers + 1))
    # Sort by due time, with some randomization for diversity
    customers.sort(key=lambda v: (due[v], random.random()))

    routes = []
    # Precompute route states for quick feasibility checks
    # For simplicity, we'll simulate when inserting

    for cust in customers:
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')
        best_feasible = False

        # Try inserting into existing routes
        for r_idx, route in enumerate(routes):
            # Try all positions
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                feasible, dist, end_time = _simulate_route(
                    new_route, D, demands, ready, due, service,
                    capacity, battery_cap, speed, recharge_rate,
                    horizon, stations, coords)
                if feasible:
                    # Cost: additional distance + small penalty for end time
                    cost = dist + 0.01 * end_time
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = pos
                        best_feasible = True

        if best_feasible:
            routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
        else:
            # Create new route
            routes.append([cust])

    return routes

def _two_opt_improve(route, D, demands, ready, due, service, capacity,
                     battery_cap, speed, recharge_rate, horizon, stations, coords,
                     max_iter=100):
    """
    2-opt improvement for a single route.
    """
    n = len(route)
    if n < 2:
        return route

    best_route = route[:]
    best_dist = _simulate_route(best_route, D, demands, ready, due, service,
                                 capacity, battery_cap, speed, recharge_rate,
                                 horizon, stations, coords)[1]

    for _ in range(max_iter):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i..j
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                feasible, dist, _ = _simulate_route(
                    new_route, D, demands, ready, due, service,
                    capacity, battery_cap, speed, recharge_rate,
                    horizon, stations, coords)
                if feasible and dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = dist
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best_route

def _inter_route_swap(routes, D, demands, ready, due, service, capacity,
                      battery_cap, speed, recharge_rate, horizon, stations, coords,
                      max_iter=50):
    """
    Try swapping customers between routes to improve total distance.
    """
    n_routes = len(routes)
    if n_routes < 2:
        return routes

    # Compute current total distance
    current_dist = _compute_total_distance(routes, D)
    best_routes = [r[:] for r in routes]
    best_dist = current_dist

    for _ in range(max_iter):
        improved = False
        for r1 in range(n_routes):
            for r2 in range(r1 + 1, n_routes):
                for i in range(len(best_routes[r1])):
                    for j in range(len(best_routes[r2])):
                        c1 = best_routes[r1][i]
                        c2 = best_routes[r2][j]

                        # Swap
                        new_r1 = best_routes[r1][:]
                        new_r2 = best_routes[r2][:]
                        new_r1[i] = c2
                        new_r2[j] = c1

                        # Check feasibility of both routes
                        f1, d1, _ = _simulate_route(
                            new_r1, D, demands, ready, due, service,
                            capacity, battery_cap, speed, recharge_rate,
                            horizon, stations, coords)
                        f2, d2, _ = _simulate_route(
                            new_r2, D, demands, ready, due, service,
                            capacity, battery_cap, speed, recharge_rate,
                            horizon, stations, coords)

                        if f1 and f2:
                            new_total = d1 + d2
                            # Subtract old distances of these two routes
                            old_d1 = _simulate_route(best_routes[r1], D, demands, ready, due, service,
                                                      capacity, battery_cap, speed, recharge_rate,
                                                      horizon, stations, coords)[1]
                            old_d2 = _simulate_route(best_routes[r2], D, demands, ready, due, service,
                                                      capacity, battery_cap, speed, recharge_rate,
                                                      horizon, stations, coords)[1]
                            delta = new_total - (old_d1 + old_d2)
                            if delta < -1e-9:
                                best_routes[r1] = new_r1
                                best_routes[r2] = new_r2
                                best_dist += delta
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

    return best_routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]

    D = _build_dist_matrix(coords)

    best_routes = None
    best_dist = float('inf')

    # Multiple seeds for greedy construction
    for seed in range(20):
        if time.time() > deadline:
            break

        routes = _greedy_construct(instance, D, seed=seed)

        # 2-opt on each route
        for r_idx in range(len(routes)):
            if time.time() > deadline:
                break
            routes[r_idx] = _two_opt_improve(
                routes[r_idx], D, demands, ready, due, service,
                capacity, battery_cap, speed, recharge_rate,
                horizon, stations, coords, max_iter=50)

        # Inter-route swap
        if time.time() < deadline:
            routes = _inter_route_swap(
                routes, D, demands, ready, due, service,
                capacity, battery_cap, speed, recharge_rate,
                horizon, stations, coords, max_iter=30)

        # Validate
        if _validate_all(routes, instance):
            total_dist = _compute_total_distance(routes, D)
            if total_dist < best_dist:
                best_dist = total_dist
                best_routes = [r[:] for r in routes]

    if best_routes is None:
        # Fallback: one customer per route
        customers = list(range(1, n_customers + 1))
        customers.sort(key=lambda v: due[v])
        best_routes = [[v] for v in customers]

    return best_routes
# EVOLVE-BLOCK-END

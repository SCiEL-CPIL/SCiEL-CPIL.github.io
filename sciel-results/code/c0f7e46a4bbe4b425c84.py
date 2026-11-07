"""EV routing with time windows – improved construction + fast local search."""
import time
import math
import random
import numpy as np


def _build_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate(route, D, instance):
    """
    Simulate a single route. Returns (total_distance, valid, final_time, final_battery, load).
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = instance["stations_set"]

    t = 0.0
    batt = battery_max
    load = 0
    dist_total = 0.0
    prev = 0

    for node in route:
        d = D[prev, node]
        dist_total += d
        t += d
        batt -= d
        if batt < -1e-9:
            return dist_total, False, t, batt, load
        if node in stations_set:
            recharge_time = (battery_max - batt) * recharge_rate
            t += recharge_time
            batt = battery_max
        else:
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return dist_total, False, t, batt, load
            load += demands[node]
            if load > capacity + 1e-9:
                return dist_total, False, t, batt, load
            t += service[node]
        prev = node

    d = D[prev, 0]
    dist_total += d
    t += d
    batt -= d
    if batt < -1e-9:
        return dist_total, False, t, batt, load
    if t > horizon + 1e-9:
        return dist_total, False, t, batt, load

    return dist_total, True, t, batt, load


def _total_distance(routes, D):
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


def _is_valid_solution(routes, instance, D):
    served = set()
    stations_set = instance["stations_set"]
    for route in routes:
        dist, valid, _, _, _ = _simulate(route, D, instance)
        if not valid:
            return False
        for node in route:
            if node in stations_set:
                continue
            if node in served:
                return False
            served.add(node)
    n_customers = instance["n_customers"]
    expected = set(range(1, n_customers + 1))
    return served == expected


def _construct_greedy(instance, D, seed=42):
    """
    Greedy construction: start new route, pick nearest feasible customer.
    """
    rng = random.Random(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    horizon = instance["horizon"]
    stations_set = instance["stations_set"]
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    rng.shuffle(customers)

    routes = []
    remaining = set(customers)

    while remaining:
        route = []
        t = 0.0
        batt = battery_max
        load = 0
        prev = 0

        while remaining:
            best_node = None
            best_dist = float('inf')
            for c in remaining:
                d = D[prev, c]
                new_batt = batt - d
                if new_batt < -1e-9:
                    continue
                new_t = t + d
                if new_t < ready[c]:
                    new_t = ready[c]
                if new_t > due[c] + 1e-9:
                    continue
                new_load = load + demands[c]
                if new_load > capacity + 1e-9:
                    continue
                ret_d = D[c, 0]
                final_t = new_t + service[c] + ret_d
                final_batt = new_batt - ret_d
                if final_batt < -1e-9 or final_t > horizon + 1e-9:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_node = c

            if best_node is None:
                break

            d = D[prev, best_node]
            t += d
            batt -= d
            if t < ready[best_node]:
                t = ready[best_node]
            t += service[best_node]
            load += demands[best_node]
            route.append(best_node)
            remaining.discard(best_node)
            prev = best_node

        if route:
            routes.append(route)
        else:
            c = min(remaining, key=lambda x: D[0, x])
            routes.append([c])
            remaining.discard(c)

    return routes


def _try_insert_station(route, instance, D, pos):
    """Try to insert a station at position pos in the route."""
    stations = instance["stations"]
    for st in stations:
        new_route = route[:pos] + [st] + route[pos:]
        dist, valid, _, _, _ = _simulate(new_route, D, instance)
        if valid:
            return new_route
    return None


def _fix_route(route, instance, D, max_iters=30):
    """Fix a route by inserting stations where needed."""
    stations_set = instance["stations_set"]

    for _ in range(max_iters):
        dist, valid, t, batt, load = _simulate(route, D, instance)
        if valid:
            return route

        # Find first infeasible point
        t_sim = 0.0
        batt_sim = instance["battery"]
        prev = 0
        for i, node in enumerate(route):
            d = D[prev, node]
            t_sim += d
            batt_sim -= d
            if batt_sim < -1e-9:
                new_route = _try_insert_station(route, instance, D, i)
                if new_route is not None:
                    route = new_route
                    break
                else:
                    return None
            if node not in stations_set:
                if t_sim < instance["ready"][node]:
                    t_sim = instance["ready"][node]
                if t_sim > instance["due"][node] + 1e-9:
                    return None
                t_sim += instance["service"][node]
            prev = node
        else:
            # If we get here without breaking, the issue is at the return leg
            new_route = _try_insert_station(route, instance, D, len(route))
            if new_route is not None:
                route = new_route
                continue
            return None

    return route if _simulate(route, D, instance)[1] else None


def _2opt(route, D, instance, time_limit):
    """2-opt improvement for a single route."""
    improved = True
    current_dist, _, _, _, _ = _simulate(route, D, instance)

    while improved:
        if time.time() > time_limit:
            break
        improved = False
        n = len(route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if time.time() > time_limit:
                    return route
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                dist, valid, _, _, _ = _simulate(new_route, D, instance)
                if valid and dist < current_dist - 1e-9:
                    route = new_route
                    current_dist = dist
                    improved = True

    return route


def _or_opt(route, D, instance, time_limit):
    """Or-opt: relocate 1 or 2 consecutive nodes."""
    improved = True
    current_dist, _, _, _, _ = _simulate(route, D, instance)

    while improved:
        if time.time() > time_limit:
            break
        improved = False
        n = len(route)
        for i in range(n):
            for length in [1, 2]:
                if i + length > n:
                    continue
                segment = route[i:i + length]
                rest = route[:i] + route[i + length:]
                if not rest:
                    continue
                for j in range(len(rest) + 1):
                    if time.time() > time_limit:
                        return route
                    new_route = rest[:j] + segment + rest[j:]
                    dist, valid, _, _, _ = _simulate(new_route, D, instance)
                    if valid and dist < current_dist - 1e-9:
                        route = new_route
                        current_dist = dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return route


def _cross_route_swap(routes, D, instance, time_limit):
    """Inter-route improvement: swap customers between routes."""
    improved = True

    while improved:
        if time.time() > time_limit:
            break
        improved = False
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                for i in range(len(routes[a])):
                    for j in range(len(routes[b])):
                        if time.time() > time_limit:
                            return routes
                        ra = list(routes[a])
                        rb = list(routes[b])
                        ra[i], rb[j] = rb[j], ra[i]
                        va = _simulate(ra, D, instance)[1]
                        vb = _simulate(rb, D, instance)[1]
                        if va and vb:
                            old_dist = _total_distance([ra, rb], D)
                            new_dist = _simulate(ra, D, instance)[0] + _simulate(rb, D, instance)[0]
                            if new_dist < old_dist - 1e-9:
                                routes[a] = ra
                                routes[b] = rb
                                improved = True

    return routes


def _savings_merge(routes, D, instance, time_limit):
    """
    Try to merge pairs of routes using Clarke-Wright savings logic.
    Only merge if the combined route is feasible.
    """
    improved = True
    while improved:
        if time.time() > time_limit:
            break
        improved = False
        n = len(routes)
        for a in range(n):
            if time.time() > time_limit:
                break
            for b in range(a + 1, n):
                if time.time() > time_limit:
                    break
                # Try to merge route b into route a
                ra = list(routes[a])
                rb = list(routes[b])
                # Try appending rb to ra
                merged = ra + rb
                dist, valid, _, _, _ = _simulate(merged, D, instance)
                if valid:
                    routes[a] = merged
                    routes.pop(b)
                    improved = True
                    break
                # Try prepending rb to ra
                merged = rb + ra
                dist, valid, _, _, _ = _simulate(merged, D, instance)
                if valid:
                    routes[a] = merged
                    routes.pop(b)
                    improved = True
                    break
            if improved:
                break
    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin for 150s limit

    coords = instance["coords"]
    D = _build_distance_matrix(coords)
    instance["stations_set"] = set(instance["stations"])

    # Construct initial solution using greedy
    routes = _construct_greedy(instance, D, seed=42)

    # Fix routes by inserting stations
    fixed_routes = []
    for r in routes:
        fixed = _fix_route(r, instance, D)
        if fixed is not None:
            fixed_routes.append(fixed)
        else:
            for c in r:
                if c not in instance["stations_set"]:
                    fixed_routes.append([c])

    best_routes = fixed_routes
    best_dist = _total_distance(best_routes, D)

    # Local Search Loop
    max_passes = 50
    for _ in range(max_passes):
        if time.time() > deadline:
            break

        current_time_limit = time.time() + (deadline - time.time()) / (max_passes - _ + 1)

        # 2-opt on each route
        for i in range(len(best_routes)):
            new_route = _2opt(best_routes[i], D, instance, current_time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route

        # Or-opt
        for i in range(len(best_routes)):
            new_route = _or_opt(best_routes[i], D, instance, current_time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route

        # Cross-route swap
        best_routes = _cross_route_swap(best_routes, D, instance, current_time_limit)

        # Savings merge
        best_routes = _savings_merge(best_routes, D, instance, current_time_limit)

        # Check if improved
        new_dist = _total_distance(best_routes, D)
        if new_dist < best_dist - 1e-9:
            best_dist = new_dist
        else:
            break

    # Final validation
    if not _is_valid_solution(best_routes, instance, D):
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[c] for c in customers]

    return best_routes

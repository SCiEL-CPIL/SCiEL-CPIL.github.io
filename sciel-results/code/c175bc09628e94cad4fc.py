"""EV routing with time windows: constructive + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import heapq
import time
import copy
import random

# Global deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 10.0  # seconds before hard limit

def _check_deadline():
    if _DEADLINE is not None and time.time() > _DEADLINE - _DEADLINE_MARGIN:
        return True
    return False


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _route_distance(route, coords):
    """Total Euclidean distance including depot legs."""
    if not route:
        return 0.0
    total = _dist(coords[0], coords[route[0]])
    for i in range(len(route) - 1):
        total += _dist(coords[route[i]], coords[route[i + 1]])
    total += _dist(coords[route[-1]], coords[0])
    return total


def _simulate_route(route, instance):
    """
    Simulate a route. Returns (feasible, total_distance, final_time, final_battery).
    If infeasible, returns (False, inf, inf, 0).
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    total_demand = sum(demands[v] for v in route if v not in stations)
    if total_demand > capacity:
        return (False, float('inf'), float('inf'), 0.0)

    t = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0  # depot

    for v in route:
        d = _dist(coords[prev], coords[v])
        t += d / speed
        battery -= d
        total_dist += d

        if battery < -1e-9:
            return (False, float('inf'), float('inf'), 0.0)

        if v in stations:
            # Recharge: time = (battery_max - battery) * recharge_rate
            charge_time = (battery_max - battery) * recharge_rate
            t += charge_time
            battery = battery_max
        else:
            # Customer: time window
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return (False, float('inf'), float('inf'), 0.0)
            t += service[v]

        prev = v

    # Return to depot
    d = _dist(coords[prev], coords[0])
    t += d / speed
    battery -= d
    total_dist += d

    if battery < -1e-9:
        return (False, float('inf'), float('inf'), 0.0)
    if t > horizon + 1e-9:
        return (False, float('inf'), float('inf'), 0.0)

    return (True, total_dist, t, battery)


def _solve_instance(instance):
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = instance["stations"]
    demands = instance["demands"]
    due = instance["due"]

    # Sort customers by due time
    customers_sorted = sorted(customers, key=lambda v: due[v])

    # Greedy insertion: build routes
    routes = []
    unassigned = set(customers_sorted)

    # Precompute distances
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            dist[i][j] = d
            dist[j][i] = d

    # Try to build routes by inserting customers
    # Start with a few empty routes
    max_routes = 8
    routes = [[] for _ in range(max_routes)]
    route_loads = [0.0] * max_routes
    route_batteries = [instance["battery"]] * max_routes
    route_times = [0.0] * max_routes

    # Greedy: assign each customer to the best route
    # We'll use a simpler approach: nearest insertion
    # Start with first customer in route 0
    if customers_sorted:
        routes[0].append(customers_sorted[0])
        route_loads[0] += demands[customers_sorted[0]]
        unassigned.discard(customers_sorted[0])

    for cust in customers_sorted[1:]:
        if _check_deadline():
            break
        best_route = -1
        best_cost = float('inf')
        for r in range(len(routes)):
            if route_loads[r] + demands[cust] > instance["capacity"]:
                continue
            # Try inserting cust at each position
            route = routes[r]
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                feasible, d, t, b = _simulate_route(new_route, instance)
                if feasible and d < best_cost:
                    best_cost = d
                    best_route = r
                    best_pos = pos
        if best_route >= 0:
            routes[best_route].insert(best_pos, cust)
            route_loads[best_route] += demands[cust]
            unassigned.discard(cust)
        else:
            # Open a new route
            new_r = len(routes)
            if new_r < max_routes:
                routes.append([cust])
                route_loads.append(demands[cust])
                unassigned.discard(cust)
            else:
                # Force into some route (shouldn't happen)
                routes[0].append(cust)
                route_loads[0] += demands[cust]
                unassigned.discard(cust)

    # Remove empty routes
    routes = [r for r in routes if r]

    # Local search: 2-opt and relocate
    improved = True
    iterations = 0
    max_iters = 500
    while improved and iterations < max_iters:
        improved = False
        iterations += 1
        if _check_deadline():
            break

        # 2-opt within each route
        for r in range(len(routes)):
            if len(routes[r]) < 2:
                continue
            for i in range(len(routes[r]) - 1):
                for j in range(i + 1, len(routes[r])):
                    new_route = routes[r][:i] + routes[r][i:j + 1][::-1] + routes[r][j + 1:]
                    f1, d1, t1, b1 = _simulate_route(routes[r], instance)
                    f2, d2, t2, b2 = _simulate_route(new_route, instance)
                    if f2 and (not f1 or d2 < d1 - 1e-9):
                        routes[r] = new_route
                        improved = True

        # Relocate between routes
        for r1 in range(len(routes)):
            if not routes[r1]:
                continue
            for i in range(len(routes[r1])):
                cust = routes[r1][i]
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    for j in range(len(routes[r2]) + 1):
                        new_r1 = routes[r1][:i] + routes[r1][i + 1:]
                        new_r2 = routes[r2][:j] + [cust] + routes[r2][j:]
                        f1, d1, t1, b1 = _simulate_route(new_r1, instance)
                        f2, d2, t2, b2 = _simulate_route(new_r2, instance)
                        if f2 and (f1 or True):
                            old_d = _route_distance(routes[r1], coords) + _route_distance(routes[r2], coords)
                            new_d = _route_distance(new_r1, coords) + _route_distance(new_r2, coords)
                            if new_d < old_d - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    # Remove empty routes again
    routes = [r for r in routes if r]

    return routes


def plan(instance):
    global _DEADLINE
    _DEADLINE = time.time() + 150.0 - 5.0  # 145s effective

    # Solve single instance
    routes = _solve_instance(instance)

    # Verify all customers are served
    served = set()
    for r in routes:
        for v in r:
            if v > 0:
                served.add(v)
    expected = set(range(1, instance["n_customers"] + 1))
    if served != expected:
        # Fallback: one customer per route
        routes = [[v] for v in range(1, instance["n_customers"] + 1)]

    return routes
# EVOLVE-BLOCK-END

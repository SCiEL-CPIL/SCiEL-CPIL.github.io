"""EV routing with time windows and battery; constructive + 2-opt polish."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 145.0  # safety margin under 150s kill


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _route_cost(route, coords, speed):
    # depot (0) at both ends
    total = _dist(coords[0], coords[route[0]])
    for i in range(len(route) - 1):
        total += _dist(coords[route[i]], coords[route[i + 1]])
    total += _dist(coords[route[-1]], coords[0])
    return total


def _simulate(route, coords, demands, ready, due, service,
              battery_max, recharge_rate, horizon, speed):
    """Return (ok, total_distance)."""
    t = 0.0
    bat = battery_max
    total_d = 0.0
    prev = 0
    for node in route:
        d = _dist(coords[prev], coords[node])
        t += d / speed
        bat -= d
        total_d += d
        if bat < -1e-9:
            return False, total_d
        if node == 0:
            prev = node
            continue
        if node in service:
            # customer
            if t > due[node] + 1e-9:
                return False, total_d
            t = max(t, ready[node])
            t += service[node]
            if t > horizon + 1e-9:
                return False, total_d
        else:
            # station: recharge
            need = battery_max - bat
            t += need * recharge_rate
            bat = battery_max
        prev = node
    # return to depot
    d = _dist(coords[prev], coords[0])
    t += d / speed
    bat -= d
    total_d += d
    if bat < -1e-9:
        return False, total_d
    if t > horizon + 1e-9:
        return False, total_d
    return True, total_d


def _try_insert(route, customer, coords, demands, ready, due, service,
                battery_max, recharge_rate, horizon, speed, cap,
                cur_demand):
    """Try inserting customer into best feasible position. Return (best_pos, best_cost) or None."""
    best_pos = None
    best_cost = None
    # positions 0..len(route)
    for pos in range(len(route) + 1):
        new_route = route[:pos] + [customer] + route[pos:]
        # quick demand check
        if cur_demand + demands[customer] > cap:
            return None
        ok, cost = _simulate(new_route, coords, demands, ready, due, service,
                             battery_max, recharge_rate, horizon, speed)
        if ok:
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_pos = pos
    if best_pos is None:
        return None
    return best_pos, best_cost


def _two_opt(route, coords, demands, ready, due, service,
             battery_max, recharge_rate, horizon, speed, cap, cur_demand):
    """In-place 2-opt polish; only accept feasible improvements."""
    n = len(route)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                if new_route == route:
                    continue
                ok, cost = _simulate(new_route, coords, demands, ready, due, service,
                                     battery_max, recharge_rate, horizon, speed)
                if ok:
                    old_cost = _route_cost(route, coords, speed)
                    if cost < old_cost - 1e-9:
                        route[:] = new_route
                        improved = True
    return route


def _solve_one(instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    speed = instance["speed"]

    customers = list(range(1, n_customers + 1))

    # Build routes greedily: start with a customer, insert nearest feasible.
    # Order customers by due time to respect windows.
    order = sorted(customers, key=lambda v: due[v])

    routes = []
    route_demands = []
    unassigned = set(customers)

    # Precompute distance from depot to each customer for seeding.
    while unassigned:
        if time.time() > _DEADLINE:
            break
        # seed: pick unassigned customer closest to depot
        seed = min(unassigned, key=lambda v: _dist(coords[0], coords[v]))
        unassigned.remove(seed)
        route = [seed]
        cur_demand = demands[seed]
        # repeatedly insert best (lowest incremental cost) feasible unassigned customer
        while True:
            best = None  # (cost, customer, pos)
            for c in list(unassigned):
                if cur_demand + demands[c] > capacity:
                    continue
                res = _try_insert(route, c, coords, demands, ready, due, service,
                                  battery_max, recharge_rate, horizon, speed, capacity,
                                  cur_demand)
                if res is None:
                    continue
                pos, cost = res
                # incremental cost
                incr = cost - _route_cost(route, coords, speed)
                if best is None or incr < best[0]:
                    best = (incr, c, pos, cost)
            if best is None:
                break
            _, c, pos, _ = best
            unassigned.remove(c)
            route.insert(pos, c)
            cur_demand += demands[c]
        routes.append(route)
        route_demands.append(cur_demand)

    # Polish each route with 2-opt
    for route in routes:
        if time.time() > _DEADLINE:
            break
        _two_opt(route, coords, demands, ready, due, service,
                 battery_max, recharge_rate, horizon, speed, capacity,
                 sum(demands[v] for v in route))

    # If we ran out of time before assigning all, assign remaining one-per-route (feasible)
    if unassigned:
        for c in sorted(unassigned, key=lambda v: due[v]):
            routes.append([c])

    return routes


def plan(instance):
    t0 = time.time()
    try:
        return _solve_one(instance)
    except Exception:
        # Fallback: one customer per route (always feasible)
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        return [[v] for v in customers]
# EVOLVE-BLOCK-END

"""EV routing with time windows: constructive heuristic + local search."""
# EVOLVE-BLOCK-START
import time as _time
import math
import heapq

_DEADLINE = 130.0  # safety margin below 150s kill
_START = None


def _dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


def _build_dist(coords):
    n = len(coords)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            v = _dist(xi, yi, xj, yj)
            d[i][j] = v
            d[j][i] = v
    return d


def _simulate(route, inst, dist, speed=1.0, recharge_rate=0.5, battery_max=300.0,
              horizon=1000.0, capacity=100.0):
    """Return (total_distance, valid, total_demand)."""
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    demands = inst["demands"]
    stations = set(inst["stations"])
    n_customers = inst["n_customers"]

    total_dist = 0.0
    total_demand = 0.0
    battery = battery_max
    t = 0.0
    prev = 0
    for node in route:
        d = dist[prev][node]
        total_dist += d
        battery -= d
        if battery < -1e-9:
            return total_dist, False, total_demand
        t += d / speed
        if node == 0:
            # depot: refill
            battery = battery_max
        elif node in stations:
            # station: recharge time = (battery) * recharge_rate, then refill
            t += battery * recharge_rate
            battery = battery_max
        else:
            # customer
            total_demand += demands[node]
            if total_demand > capacity + 1e-9:
                return total_dist, False, total_demand
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return total_dist, False, total_demand
            t += service[node]
        prev = node
    # return to depot
    d = dist[prev][0]
    total_dist += d
    battery -= d
    if battery < -1e-9:
        return total_dist, False, total_demand
    t += d / speed
    if t > horizon + 1e-9:
        return total_dist, False, total_demand
    return total_dist, True, total_demand


def _solve_one(instance, deadline):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    station_set = set(stations)
    capacity = instance.get("capacity", 100)
    battery_max = instance.get("battery", 300)
    speed = instance.get("speed", 1.0)
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance.get("horizon", 1000)

    dist = _build_dist(coords)
    customers = list(range(1, n_customers + 1))

    # --- Constructive: nearest neighbor with time-window feasibility, sorted by due ---
    def construct(order_key):
        remaining = list(customers)
        remaining.sort(key=order_key)
        routes = []
        while remaining:
            # start from depot, pick feasible customer
            route = []
            battery = battery_max
            t = 0.0
            prev = 0
            total_demand = 0.0
            # greedy insert
            while remaining:
                best_i = -1
                best_cost = float("inf")
                for idx, c in enumerate(remaining):
                    d = dist[prev][c]
                    b2 = battery - d
                    if b2 < -1e-9:
                        continue
                    t2 = t + d / speed
                    if t2 < ready[c]:
                        t2 = ready[c]
                    if t2 > due[c] + 1e-9:
                        continue
                    if total_demand + demands[c] > capacity + 1e-9:
                        continue
                    # cost: distance + small penalty for waiting
                    cost = d + 0.01 * max(0, ready[c] - (t + d / speed))
                    if cost < best_cost:
                        best_cost = cost
                        best_i = idx
                if best_i == -1:
                    break
                c = remaining.pop(best_i)
                route.append(c)
                d = dist[prev][c]
                battery -= d
                t += d / speed
                if t < ready[c]:
                    t = ready[c]
                t += service[c]
                total_demand += demands[c]
                prev = c
            if route:
                routes.append(route)
        return routes

    # Try a few orderings
    best_routes = None
    best_obj = float("inf")

    orderings = [
        lambda c: due[c],
        lambda c: due[c] - 2 * dist[0][c],
        lambda c: dist[0][c],
        lambda c: (ready[c], due[c]),
        lambda c: -demands[c],
    ]

    for ok in orderings:
        if _time.time() > deadline:
            break
        routes = construct(ok)
        total = 0.0
        valid = True
        for r in routes:
            dd, v, _ = _simulate(r, instance, dist, speed, recharge_rate, battery_max, horizon, capacity)
            if not v:
                valid = False
                break
            total += dd
        if valid and total < best_obj:
            best_obj = total
            best_routes = [list(r) for r in routes]

    if best_routes is None:
        # fallback: one per route
        best_routes = [[c] for c in customers]
        best_obj = float("inf")

    # --- Local search: 2-opt + relocate + swap across routes ---
    def eval_routes(routes):
        total = 0.0
        for r in routes:
            dd, v, _ = _simulate(r, instance, dist, speed, recharge_rate, battery_max, horizon, capacity)
            if not v:
                return float("inf")
            total += dd
        return total

    def try_2opt(route):
        n = len(route)
        if n < 2:
            return False
        improved = False
        for i in range(n - 1):
            if _time.time() > deadline:
                return improved
            for j in range(i + 1, n):
                # reverse route[i..j]
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                dd, v, _ = _simulate(new_route, instance, dist, speed, recharge_rate, battery_max, horizon, capacity)
                if v:
                    old_dd, _, _ = _simulate(route, instance, dist, speed, recharge_rate, battery_max, horizon, capacity)
                    if dd < old_dd - 1e-9:
                        route[:] = new_route
                        improved = True
        return improved

    def try_relocate(routes):
        improved = False
        for ri, route in enumerate(routes):
            for k in range(len(route)):
                if _time.time() > deadline:
                    return improved
                c = route[k]
                rest = route[:k] + route[k + 1:]
                for rj, route2 in enumerate(routes):
                    if rj == ri:
                        continue
                    for pos in range(len(route2) + 1):
                        new2 = route2[:pos] + [c] + route2[pos:]
                        dd, v, _ = _simulate(new2, instance, dist, speed, recharge_rate, battery_max, horizon, capacity)
                        if not v:
                            continue
                        dd_rest, v_rest, _ = _simulate(rest, instance, dist, speed, recharge_rate, battery_max, horizon, capacity)
                        if not v_rest:
                            continue
                        old_dd2, _, _ = _simulate(route2, instance, dist, speed, recharge_rate, battery_max, horizon, capacity)
                        old_dd, _, _ = _simulate(route, instance, dist, speed, recharge_rate, battery_max, horizon, capacity)
                        if dd + dd_rest < old_dd + old_dd2 - 1e-9:
                            route2[:] = new2
                            route[:] = rest
                            improved = True
                            break
        return improved

    def try_swap(routes):
        improved = False
        for ri in range(len(routes)):
            for rj in range(ri + 1, len(routes)):
                if _time.time() > deadline:
                    return improved
                route_i = routes[ri]
                route_j = routes[rj]
                for k in range(len(route_i)):
                    for m in range(len(route_j)):
                        c1 = route_i[k]
                        c2 = route_j[m]
                        new_i = route_i[:k] + [c2] + route_i[k + 1:]
                        new_j = route_j[:m] + [c1] + route_j[m + 1:]
                        dd_i, v_i, _ = _simulate(new_i, instance, dist, speed, recharge_rate, battery_max, horizon, capacity)
                        dd_j, v_j, _ = _simulate(new_j, instance, dist, speed, recharge_rate, battery_max, horizon, capacity)
                        if not v_i or not v_j:
                            continue
                        old_i, _, _ = _simulate(route_i, instance, dist, speed, recharge_rate, battery_max, horizon, capacity)
                        old_j, _, _ = _simulate(route_j, instance, dist, speed, recharge_rate, battery_max, horizon, capacity)
                        if dd_i + dd_j < old_i + old_j - 1e-9:
                            route_i[:] = new_i
                            route_j[:] = new_j
                            improved = True
        return improved

    # Iterate local search
    cur_obj = eval_routes(best_routes)
    for iteration in range(200):
        if _time.time() > deadline:
            break
        improved = False
        for r in best_routes:
            if try_2opt(r):
                improved = True
        if try_relocate(best_routes):
            improved = True
        if try_swap(best_routes):
            improved = True
        new_obj = eval_routes(best_routes)
        if new_obj < cur_obj - 1e-9:
            cur_obj = new_obj
        if not improved:
            break

    return best_routes


def plan(instance):
    global _START
    if _START is None:
        _START = _time.time()
    elapsed = _time.time() - _START
    deadline = _START + _DEADLINE - 1.0
    routes = _solve_one(instance, deadline)
    return routes
# EVOLVE-BLOCK-END

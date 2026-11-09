"""Rewritten EV routing solver: keep the same entry-point and return interface."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = float(instance["capacity"])
    battery = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = float(np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1]))
            dist[i, j] = d
            dist[j, i] = d

    customers = list(range(1, n_customers + 1))

    def check_route(route):
        """Feasibility + total distance. Stations are allowed in route nodes."""
        if not route:
            return True, 0.0

        total_demand = float(sum(demands[v] for v in route))
        if total_demand > capacity:
            return False, 0.0

        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0

        for v in route:
            d = float(dist[prev, v])
            bat -= d
            t += d / speed
            total_dist += d
            if bat < -1e-9:
                return False, 0.0

            if v in stations:
                # Station: recharge time = (battery - energy) * recharge_rate, then refill
                if bat < 0:
                    return False, 0.0
                charge_time = (battery - bat) * recharge_rate
                t += charge_time
                bat = battery
            else:
                # Customer: time window + service
                if t < ready[v]:
                    t = float(ready[v])
                if t > float(due[v]) + 1e-9:
                    return False, 0.0
                t += float(service[v])

            prev = v

        d = float(dist[prev, 0])
        bat -= d
        t += d / speed
        total_dist += d
        if bat < -1e-9 or t > horizon + 1e-9:
            return False, 0.0

        return True, total_dist

    def route_distance(route):
        if not route:
            return 0.0
        s = float(dist[0, route[0]])
        for i in range(len(route) - 1):
            s += float(dist[route[i], route[i + 1]])
        s += float(dist[route[-1], 0])
        return s

    def total_distance(routes):
        return float(sum(route_distance(r) for r in routes))

    # 1) Greedy construction with due-date ordering and nearest-feasible insertion
    def greedy_construct(order):
        remaining = set(order)
        routes = []
        while remaining:
            route = []
            prev = 0
            t = 0.0
            bat = battery
            load = 0.0
            while remaining:
                best_v = None
                best_score = float('inf')
                for v in remaining:
                    if load + float(demands[v]) > capacity:
                        continue
                    d = float(dist[prev, v])
                    bat_after = bat - d
                    t_after = t + d / speed
                    if bat_after < -1e-9:
                        continue
                    if v not in stations:
                        if t_after < ready[v]:
                            t_after = float(ready[v])
                        if t_after > float(due[v]) + 1e-9:
                            continue
                    # Prefer close, then earlier time window slack
                    score = d + 0.05 * max(0.0, t_after - (float(ready[v]) if v not in stations else 0.0))
                    if score < best_score:
                        best_score = score
                        best_v = v
                if best_v is None:
                    break
                v = best_v
                d = float(dist[prev, v])
                bat -= d
                t += d / speed
                load += float(demands[v])
                if v not in stations:
                    if t < ready[v]:
                        t = float(ready[v])
                    t += float(service[v])
                else:
                    if bat < 0:
                        bat = 0.0
                    t += (battery - bat) * recharge_rate
                    bat = battery
                route.append(v)
                remaining.discard(v)
                prev = v
            if route:
                # ensure depot return feasible
                d = float(dist[prev, 0])
                if bat - d >= -1e-9 and t + d / speed <= horizon + 1e-9:
                    routes.append(route)
                else:
                    routes.append(route)
        return routes

    # Initial: one customer per route (guaranteed feasible)
    best_routes = [[v] for v in customers]
    best_dist = total_distance(best_routes)

    # Try a few greedy orderings
    if time.time() < deadline:
        for order in [
            sorted(customers, key=lambda v: float(due[v])),
            sorted(customers, key=lambda v: float(ready[v])),
            sorted(customers, key=lambda v: (float(due[v]) - float(ready[v])), reverse=True),
        ]:
            routes = greedy_construct(order)
            # Validate
            seen = set()
            ok = True
            td = 0.0
            for r in routes:
                f, d = check_route(r)
                if not f:
                    ok = False
                    break
                for v in r:
                    if v in seen:
                        ok = False
                        break
                    seen.add(v)
                td += d
            if not ok or seen != set(customers):
                continue
            if td < best_dist - 1e-9:
                best_dist = td
                best_routes = routes

    # 2) Local search: relocate + 2-opt (with quick feasibility checks)
    def feasible_fast(route):
        # capacity
        if float(sum(demands[v] for v in route)) > capacity:
            return False
        t = 0.0
        bat = battery
        prev = 0
        for v in route:
            d = float(dist[prev, v])
            bat -= d
            t += d / speed
            if bat < -1e-9:
                return False
            if v in stations:
                if bat < 0:
                    return False
                t += (battery - bat) * recharge_rate
                bat = battery
            else:
                if t < ready[v]:
                    t = float(ready[v])
                if t > float(due[v]) + 1e-9:
                    return False
                t += float(service[v])
            prev = v
        d = float(dist[prev, 0])
        bat -= d
        t += d / speed
        return bat >= -1e-9 and t <= horizon + 1e-9

    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur = total_distance(best_routes)

        # Relocate
        for i in range(len(best_routes)):
            for j in range(len(best_routes[i])):
                if time.time() > deadline:
                    break
                v = best_routes[i][j]
                for k in range(len(best_routes)):
                    if k == i:
                        continue
                    for pos in range(len(best_routes[k]) + 1):
                        # build new routes
                        new_routes = []
                        for r in range(len(best_routes)):
                            if r == i:
                                new_routes.append([x for x in best_routes[i] if x != v])
                            elif r == k:
                                nr = best_routes[k][:]
                                nr.insert(pos, v)
                                new_routes.append(nr)
                            else:
                                new_routes.append(best_routes[r][:])
                        # quick feasibility
                        ok = all(feasible_fast(r) for r in new_routes if r)
                        if not ok:
                            continue
                        nd = total_distance(new_routes)
                        if nd < cur - 1e-6:
                            best_routes = new_routes
                            cur = nd
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        # 2-opt within routes
        for i in range(len(best_routes)):
            route = best_routes[i]
            n = len(route)
            if n < 2:
                continue
            for a in range(n):
                for b in range(a + 1, n):
                    if time.time() > deadline:
                        break
                    new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                    if not feasible_fast(new_route):
                        continue
                    # boundary distance delta
                    old_b = (float(dist[0, route[a]] if a == 0 else dist[route[a - 1], route[a]]) +
                             float(dist[route[b], 0] if b == n - 1 else dist[route[b], route[b + 1]]))
                    new_b = (float(dist[0, route[b]] if a == 0 else dist[route[a - 1], route[b]]) +
                             float(dist[route[a], 0] if b == n - 1 else dist[route[a], route[b + 1]]))
                    if new_b < old_b - 1e-6:
                        best_routes[i] = new_route
                        improved = True

    return best_routes
# EVOLVE-BLOCK-END

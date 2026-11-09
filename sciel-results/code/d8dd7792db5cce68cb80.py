"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _build_dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _check_route(route, dist, demands, ready, due, service, n_customers,
                 battery_max, capacity, speed, recharge_rate, horizon):
    """Full feasibility check. Returns (feasible, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0

    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
    if total_demand > capacity:
        return False, 0.0, 0.0

    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0

    for v in route:
        d = dist[prev][v]
        battery -= d
        total_dist += d
        time_t += d / speed

        if battery < -1e-9:
            return False, 0.0, 0.0

        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max

        prev = v

    d = dist[prev][0]
    battery -= d
    total_dist += d
    time_t += d / speed

    if battery < -1e-9 or time_t > horizon + 1e-9:
        return False, 0.0, 0.0

    return True, total_dist, time_t


def _fast_check(route, dist, demands, ready, due, service, n_customers,
                battery_max, capacity, speed, recharge_rate, horizon):
    """Fast feasibility check with early exits. Returns (feasible, total_distance)."""
    if not route:
        return True, 0.0

    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
    if total_demand > capacity:
        return False, 0.0

    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0

    for v in route:
        d = dist[prev][v]
        battery -= d
        total_dist += d
        time_t += d / speed

        if battery < -1e-9:
            return False, 0.0

        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max

        prev = v

    d = dist[prev][0]
    battery -= d
    total_dist += d
    time_t += d / speed

    if battery < -1e-9 or time_t > horizon + 1e-9:
        return False, 0.0

    return True, total_dist


def _route_distance(route, dist):
    """Compute total distance of a route including depot legs."""
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for v in route:
        total += dist[prev][v]
        prev = v
    total += dist[prev][0]
    return total


def _construct_initial_routes(instance, dist):
    """Construct initial routes using a savings-based heuristic."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    routes = [[i] for i in range(1, n_customers + 1)]

    savings = []
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            ci = i + 1
            cj = j + 1
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            savings.append((s, ci, cj))

    savings.sort(reverse=True)

    route_of = {}
    for idx, r in enumerate(routes):
        for c in r:
            route_of[c] = idx

    route_demand = [0] * len(routes)
    for idx, r in enumerate(routes):
        for c in r:
            if c <= n_customers:
                route_demand[idx] += demands[c]

    for s, ci, cj in savings:
        ri = route_of.get(ci, -1)
        rj = route_of.get(cj, -1)
        if ri == -1 or rj == -1 or ri == rj:
            continue

        if route_demand[ri] + route_demand[rj] > capacity:
            continue

        best_new_route = None
        best_dist = float('inf')

        for order in [[routes[ri], routes[rj]], [routes[rj], routes[ri]]]:
            merged = order[0] + order[1]
            feasible, total_dist = _fast_check(
                merged, dist, demands, ready, due, service, n_customers,
                battery_max, capacity, speed, recharge_rate, horizon)
            if feasible and total_dist < best_dist:
                best_dist = total_dist
                best_new_route = merged

        if best_new_route is not None:
            routes[ri] = best_new_route
            routes[rj] = []
            route_demand[ri] += route_demand[rj]
            route_demand[rj] = 0
            for c in best_new_route:
                route_of[c] = ri

    routes = [r for r in routes if r]

    all_served = set()
    for r in routes:
        all_served.update(r)

    for c in range(1, n_customers + 1):
        if c not in all_served:
            inserted = False
            for idx, r in enumerate(routes):
                current_demand = sum(demands[v] for v in r if v <= n_customers)
                if current_demand + demands[c] > capacity:
                    continue
                best_pos = -1
                best_cost = float('inf')
                for pos in range(len(r) + 1):
                    new_route = r[:pos] + [c] + r[pos:]
                    feasible, _ = _fast_check(
                        new_route, dist, demands, ready, due, service,
                        n_customers, battery_max, capacity, speed,
                        recharge_rate, horizon)
                    if feasible:
                        if pos == 0:
                            inc = dist[0][c] + dist[c][r[0]] - dist[0][r[0]]
                        elif pos == len(r):
                            inc = dist[r[-1]][c] + dist[c][0] - dist[r[-1]][0]
                        else:
                            inc = dist[r[pos-1]][c] + dist[c][r[pos]] - dist[r[pos-1]][r[pos]]
                        if inc < best_cost:
                            best_cost = inc
                            best_pos = pos
                if best_pos >= 0:
                    routes[idx] = r[:best_pos] + [c] + r[best_pos:]
                    inserted = True
                    break
            if not inserted:
                routes.append([c])

    return routes


def _two_opt(route, dist, demands, ready, due, service, n_customers,
             battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """2-opt improvement with distance pruning."""
    if len(route) < 2:
        return route

    n = len(route)
    improved = True
    max_passes = 20
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return route
                prev_i = route[i-1] if i > 0 else 0
                next_j = route[j+1] if j < n - 1 else 0

                d_old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[i]][next_j]

                if d_new < d_old - 1e-9:
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, _ = _fast_check(
                        new_route, dist, demands, ready, due, service,
                        n_customers, battery_max, capacity, speed,
                        recharge_rate, horizon)
                    if feasible:
                        route = new_route
                        improved = True
                        break
            if improved:
                break

    return route


def _or_opt(route, dist, demands, ready, due, service, n_customers,
            battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Or-opt: relocate segments of length 1 or 2 within the same route."""
    if len(route) < 3:
        return route

    n = len(route)
    improved = True
    max_passes = 10
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        for seg_len in [1, 2]:
            for i in range(n - seg_len + 1):
                if time.time() > deadline:
                    return route
                segment = route[i:i+seg_len]
                # Remove segment
                rest = route[:i] + route[i+seg_len:]
                if not rest:
                    continue
                # Try inserting segment at every position in rest
                best_pos = -1
                best_dist = float('inf')
                old_dist = _route_distance(route, dist)

                for pos in range(len(rest) + 1):
                    if pos == i or pos == i + 1:
                        continue  # Skip original position
                    new_route = rest[:pos] + segment + rest[pos:]
                    feasible, total_dist = _fast_check(
                        new_route, dist, demands, ready, due, service,
                        n_customers, battery_max, capacity, speed,
                        recharge_rate, horizon)
                    if feasible and total_dist < best_dist:
                        best_dist = total_dist
                        best_pos = pos

                if best_pos >= 0 and best_dist < old_dist - 1e-9:
                    rest = route[:i] + route[i+seg_len:]
                    route = rest[:best_pos] + segment + rest[best_pos:]
                    n = len(route)
                    improved = True
                    break
            if improved:
                break

    return route


def _cross_route_improve(routes, dist, demands, ready, due, service, n_customers,
                         battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Try to relocate customers between routes to reduce total distance."""
    n_routes = len(routes)
    if n_routes < 2:
        return routes

    improved = True
    max_passes = 5
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1

        for ri in range(n_routes):
            if time.time() > deadline:
                return routes
            if not routes[ri]:
                continue
            for ci, customer in enumerate(routes[ri]):
                if customer > n_customers:
                    continue  # Skip stations for now
                # Try moving this customer to another route
                for rj in range(n_routes):
                    if rj == ri or not routes[rj]:
                        continue
                    # Check capacity
                    demand_j = sum(demands[v] for v in routes[rj] if v <= n_customers)
                    if demand_j + demands[customer] > capacity:
                        continue

                    # Try all insertion positions in route j
                    best_pos = -1
                    best_total = float('inf')

                    # Current total distance
                    dist_i = _route_distance(routes[ri], dist)
                    dist_j = _route_distance(routes[rj], dist)
                    current_total = dist_i + dist_j

                    # Remove customer from route i
                    new_route_i = routes[ri][:ci] + routes[ri][ci+1:]

                    for pos in range(len(routes[rj]) + 1):
                        new_route_j = routes[rj][:pos] + [customer] + routes[rj][pos:]
                        feasible_i, dist_i_new = _fast_check(
                            new_route_i, dist, demands, ready, due, service,
                            n_customers, battery_max, capacity, speed,
                            recharge_rate, horizon)
                        if not feasible_i:
                            continue
                        feasible_j, dist_j_new = _fast_check(
                            new_route_j, dist, demands, ready, due, service,
                            n_customers, battery_max, capacity, speed,
                            recharge_rate, horizon)
                        if not feasible_j:
                            continue
                        total_new = dist_i_new + dist_j_new
                        if total_new < best_total:
                            best_total = total_new
                            best_pos = pos

                    if best_pos >= 0 and best_total < current_total - 1e-9:
                        new_route_i = routes[ri][:ci] + routes[ri][ci+1:]
                        new_route_j = routes[rj][:best_pos] + [customer] + routes[rj][best_pos:]
                        routes[ri] = new_route_i
                        routes[rj] = new_route_j
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Remove empty routes
    routes = [r for r in routes if r]
    return routes


def _solve_instance(instance, time_limit=40):
    """Solve a single instance with time limit."""
    start_time = time.time()
    deadline = start_time + time_limit

    coords = instance["coords"]
    dist = _build_dist_matrix(coords)

    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]

    # Construct initial routes
    routes = _construct_initial_routes(instance, dist)

    # Local search loop
    while time.time() < deadline - 2:
        # 2-opt within each route
        for i in range(len(routes)):
            if time.time() > deadline - 2:
                break
            routes[i] = _two_opt(
                routes[i], dist, demands, ready, due, service, n_customers,
                battery_max, capacity, speed, recharge_rate, horizon, deadline - 2)

        # Or-opt within each route
        for i in range(len(routes)):
            if time.time() > deadline - 2:
                break
            routes[i] = _or_opt(
                routes[i], dist, demands, ready, due, service, n_customers,
                battery_max, capacity, speed, recharge_rate, horizon, deadline - 2)

        # Cross-route relocation
        if time.time() < deadline - 3:
            routes = _cross_route_improve(
                routes, dist, demands, ready, due, service, n_customers,
                battery_max, capacity, speed, recharge_rate, horizon, deadline - 3)

    # Final verification
    valid_routes = []
    for r in routes:
        feasible, _, _ = _check_route(
            r, dist, demands, ready, due, service, n_customers,
            battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)

    # Check all customers are served
    all_served = set()
    for r in valid_routes:
        all_served.update(r)

    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        valid_routes.append([c])

    return valid_routes


def plan(instance):
    """Main entry point. Returns routes for the instance."""
    start_time = time.time()
    # 150s total limit. Use ~45s per instance with 15s safety margin.
    per_instance_time = 45

    routes = _solve_instance(instance, per_instance_time)

    return routes
# EVOLVE-BLOCK-END

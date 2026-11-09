"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _dist_matrix(coords):
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


def _check_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """Check if a route is feasible. Returns (feasible, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0

    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]

    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return False, 0.0, 0.0

    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0  # depot

    for v in route:
        d = dist[prev][v]
        battery -= d
        time_t += d / speed
        total_dist += d

        if battery < -1e-9:
            return False, total_dist, time_t

        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v]:
                return False, total_dist, time_t
            time_t += service[v]
        else:
            # Charging station
            if battery < battery_max:
                energy_needed = battery_max - battery
                time_t += energy_needed * recharge_rate
                battery = battery_max

        prev = v

    d = dist[prev][0]
    battery -= d
    time_t += d / speed
    total_dist += d

    if battery < -1e-9:
        return False, total_dist, time_t
    if time_t > horizon:
        return False, total_dist, time_t

    return True, total_dist, time_t


def _construct_clustering_routes(instance, dist, n_clusters=None):
    """Construct initial routes using K-Means style clustering."""
    coords = instance["coords"]
    demands = instance["demands"]
    n_customers = instance["n_customers"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    if n_clusters is None:
        n_clusters = max(2, int(n_customers / 5))

    customers = list(range(1, n_customers + 1))

    # K-Means clustering
    if n_clusters >= len(customers):
        clusters = [[c] for c in customers]
    else:
        # Initialize centroids randomly
        random.shuffle(customers)
        centroids = [np.array(coords[c], dtype=float) for c in customers[:n_clusters]]

        for _ in range(20):
            # Assign customers to nearest centroid
            assignments = [[] for _ in range(n_clusters)]
            for c in customers:
                best_cluster = 0
                best_dist = float('inf')
                for k in range(n_clusters):
                    d = np.linalg.norm(np.array(coords[c]) - centroids[k])
                    if d < best_dist:
                        best_dist = d
                        best_cluster = k
                assignments[best_cluster].append(c)

            # Update centroids
            for k in range(n_clusters):
                if assignments[k]:
                    new_centroid = np.mean([np.array(coords[c]) for c in assignments[k]], axis=0)
                    centroids[k] = new_centroid
            clusters = assignments

    # Build routes from clusters
    routes = []
    for cluster in clusters:
        if not cluster:
            continue

        # Sort by angle from depot
        depot_x, depot_y = coords[0]
        angles = []
        for c in cluster:
            dx = coords[c][0] - depot_x
            dy = coords[c][1] - depot_y
            angle = math.atan2(dy, dx)
            angles.append((angle, c))
        angles.sort()
        cluster = [c for _, c in angles]

        # Split cluster into routes by capacity
        current_route = []
        current_demand = 0
        for c in cluster:
            if current_demand + demands[c] > capacity:
                if current_route:
                    routes.append(current_route)
                current_route = [c]
                current_demand = demands[c]
            else:
                current_route.append(c)
                current_demand += demands[c]
        if current_route:
            routes.append(current_route)

    return routes


def _solve_with_savings(instance, dist):
    """Solve using Clarke-Wright savings heuristic."""
    coords = instance["coords"]
    demands = instance["demands"]
    n_customers = instance["n_customers"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    customers = list(range(1, n_customers + 1))

    # Initial: each customer in its own route
    routes = {c: [c] for c in customers}
    route_of = {c: c for c in customers}  # customer -> route representative
    demand = {c: demands[c] for c in customers}

    # Compute savings
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = dist[0][i] + dist[0][j] - dist[i][j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    # Merge routes based on savings
    for s, i, j in savings:
        if s <= 0:
            continue
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue

        # Check if i and j are endpoints of their respective routes
        route_i = routes[ri]
        route_j = routes[rj]

        # Check if i is at an end of route_i
        i_at_end = (i == route_i[0] or i == route_i[-1])
        j_at_end = (j == route_j[0] or j == route_j[-1])

        if not (i_at_end and j_at_end):
            continue

        # Check capacity
        if demand[ri] + demand[rj] > capacity:
            continue

        # Merge: remove smaller route, append to larger
        if len(route_i) < len(route_j):
            ri, rj = rj, ri
            route_i, route_j = routes[rj], routes[ri]
            i, j = j, i

        # Determine orientation
        if route_i[0] == i:
            route_i = route_i[::-1]  # i is now at end
        if route_j[0] == j:
            route_j = route_j[::-1]  # j is now at end

        # Merge
        new_route = route_i + route_j
        routes[ri] = new_route
        demand[ri] = demand[ri] + demand[rj]
        for c in route_j:
            route_of[c] = ri
        del routes[rj]
        del demand[rj]

    # Convert to list of routes
    result = [r for r in routes.values()]

    # Verify all customers covered
    covered = set()
    for r in result:
        covered.update(r)
    missing = [c for c in customers if c not in covered]
    for c in missing:
        result.append([c])

    return result


def _two_opt_improve(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon, max_iters=100):
    """Apply 2-opt improvement to a single route."""
    n = len(route)
    if n < 2:
        return route

    best_route = route[:]
    best_dist = float('inf')

    # Get current distance
    feasible, cur_dist, _ = _check_route(best_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
    if feasible:
        best_dist = cur_dist

    for _ in range(max_iters):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                feasible, new_dist, _ = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                if feasible and new_dist < best_dist - 1e-9:
                    route = new_route
                    best_route = new_route[:]
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best_route


def _or_opt_improve(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon, max_iters=50):
    """Apply Or-opt improvement to a single route."""
    n = len(route)
    if n < 3:
        return route

    best_route = route[:]
    best_dist = float('inf')

    feasible, cur_dist, _ = _check_route(best_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
    if feasible:
        best_dist = cur_dist

    for _ in range(max_iters):
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(n - seg_len + 1):
                segment = route[i:i+seg_len]
                rest = route[:i] + route[i+seg_len:]
                if not rest:
                    continue
                for j in range(len(rest) + 1):
                    new_route = rest[:j] + segment + rest[j:]
                    if new_route == route:
                        continue
                    feasible, new_dist, _ = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible and new_dist < best_dist - 1e-9:
                        route = new_route
                        best_route = new_route[:]
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best_route


def _solve_instance(instance, time_limit=45):
    start_time = time.time()
    deadline = start_time + time_limit

    coords = instance["coords"]
    dist = _dist_matrix(coords)

    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])

    # Generate multiple initial solutions
    initial_routes_list = []

    # 1. Savings heuristic
    routes1 = _solve_with_savings(instance, dist)
    initial_routes_list.append(routes1)

    # 2. Clustering with different n_clusters
    for nc in [3, 4, 5, 6]:
        routes = _construct_clustering_routes(instance, dist, n_clusters=nc)
        initial_routes_list.append(routes)

    # Evaluate all initial solutions
    best_routes = None
    best_score = float('inf')

    for routes in initial_routes_list:
        # Check feasibility
        total_dist = 0.0
        feasible = True
        for r in routes:
            f, d, _ = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if not f:
                feasible = False
                break
            total_dist += d

        # Check all customers covered
        covered = set()
        for r in routes:
            covered.update(r)
        if len(covered) != n_customers:
            feasible = False

        if feasible and total_dist < best_score:
            best_score = total_dist
            best_routes = [r[:] for r in routes]

    if best_routes is None:
        # Fallback: use the first one even if infeasible
        best_routes = [r[:] for r in initial_routes_list[0]]
        best_score = float('inf')

    # Local search improvement
    while time.time() < deadline - 5:
        improved = False

        # 2-opt on each route
        for i in range(len(best_routes)):
            if time.time() >= deadline - 5:
                break
            new_route = _two_opt_improve(best_routes[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True

        # Or-opt on each route
        for i in range(len(best_routes)):
            if time.time() >= deadline - 5:
                break
            new_route = _or_opt_improve(best_routes[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True

        if not improved:
            break

    # Try inter-route moves (relocate, swap)
    while time.time() < deadline - 3:
        improved = False
        all_customers = []
        for r in best_routes:
            all_customers.extend(r)

        if not all_customers:
            break

        # Relocate
        c = random.choice(all_customers)
        src_idx = -1
        for idx, r in enumerate(best_routes):
            if c in r:
                src_idx = idx
                break
        if src_idx == -1:
            continue

        src_route = best_routes[src_idx]
        pos_c = src_route.index(c)
        src_route.pop(pos_c)

        dst_idx = random.randint(0, len(best_routes) - 1)
        dst_route = best_routes[dst_idx]
        ins_pos = random.randint(0, len(dst_route))
        dst_route.insert(ins_pos, c)

        # Check feasibility
        feasible = True
        total_dist = 0.0
        for r in best_routes:
            if not r:
                continue
            f, d, _ = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if not f:
                feasible = False
                break
            total_dist += d

        if feasible and total_dist < best_score - 1e-9:
            best_routes = [r[:] for r in best_routes]
            best_score = total_dist
            improved = True
        else:
            # Revert
            dst_route.pop(ins_pos)
            src_route.insert(pos_c, c)

        # Remove empty routes
        best_routes = [r for r in best_routes if r]

        if not improved and random.random() < 0.3:
            # Try swap
            if len(best_routes) >= 2:
                r1, r2 = random.sample(range(len(best_routes)), 2)
                if best_routes[r1] and best_routes[r2]:
                    c1 = random.choice(best_routes[r1])
                    c2 = random.choice(best_routes[r2])
                    pos1 = best_routes[r1].index(c1)
                    pos2 = best_routes[r2].index(c2)
                    best_routes[r1][pos1] = c2
                    best_routes[r2][pos2] = c1

                    feasible = True
                    total_dist = 0.0
                    for r in best_routes:
                        f, d, _ = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        if not f:
                            feasible = False
                            break
                        total_dist += d

                    if feasible and total_dist < best_score - 1e-9:
                        best_score = total_dist
                        improved = True
                    else:
                        # Revert
                        best_routes[r1][pos1] = c1
                        best_routes[r2][pos2] = c2

    # Final polish: 2-opt + Or-opt again
    for i in range(len(best_routes)):
        if time.time() >= deadline - 1:
            break
        best_routes[i] = _two_opt_improve(best_routes[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon, max_iters=50)
        best_routes[i] = _or_opt_improve(best_routes[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon, max_iters=30)

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    return best_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()

    # 3 instances, 150s total, leave 10s buffer -> 45s per instance
    per_instance_time = 45

    routes = _solve_instance(instance, per_instance_time)

    return routes
# EVOLVE-BLOCK-END

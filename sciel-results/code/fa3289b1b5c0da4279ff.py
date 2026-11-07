"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import combinations


def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate_route(route_nodes, instance, D):
    """
    Simulate a route (list of customer/station node ids, depot implicit at both ends).
    Returns (feasible, total_distance, arrival_time_at_depot, max_time_used).
    """
    cap = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    n_customers = instance["n_customers"]
    stations_set = set(instance["stations"])

    total_dist = 0.0
    total_demand = 0
    battery = battery_max
    t = 0.0

    prev = 0  # depot
    for node in route_nodes:
        d = D[prev, node]
        total_dist += d
        battery -= d
        if battery < -1e-9:
            return False, total_dist, t, False
        t += d / speed
        if node not in stations_set:
            # customer
            total_demand += demands[node]
            if total_demand > cap:
                return False, total_dist, t, False
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return False, total_dist, t, False
            t += service[node]
        else:
            # station
            energy = battery_max - battery
            t += energy * recharge_rate
            battery = battery_max

        prev = node

    # return to depot
    d = D[prev, 0]
    total_dist += d
    battery -= d
    if battery < -1e-9:
        return False, total_dist, t, False
    t += d / speed

    if t > horizon + 1e-9:
        return False, total_dist, t, False

    return True, total_dist, t, True


def _greedy_build_route(unassigned, instance, D, max_time=None):
    """
    Build one route greedily by nearest insertion with feasibility checks.
    Returns (route_nodes, total_dist) or (None, inf) if no feasible route.
    """
    if not unassigned:
        return None, float('inf')

    cap = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    stations_set = set(instance["stations"])

    # Start with the customer with earliest due time among unassigned
    start_node = min(unassigned, key=lambda v: due[v])
    route = [start_node]
    total_demand = demands[start_node]
    total_dist = D[0, start_node] + D[start_node, 0]
    remaining = [v for v in unassigned if v != start_node]

    changed = True
    while changed:
        changed = False
        best_gain = 0
        best_node = None
        best_pos = None

        for node in remaining:
            if total_demand + demands[node] > cap:
                continue
            # Try inserting at each position
            for pos in range(len(route) + 1):
                # Compute new distance
                if pos == 0:
                    new_dist = D[0, node] + D[node, route[0]] + total_dist - D[0, route[0]]
                elif pos == len(route):
                    new_dist = D[route[-1], node] + D[node, 0] + total_dist - D[route[-1], 0]
                else:
                    new_dist = D[route[pos-1], node] + D[node, route[pos]] + total_dist - D[route[pos-1], route[pos]]

                gain = total_dist - new_dist
                if gain > best_gain:
                    # Quick feasibility check: simulate the full route
                    new_route = route[:pos] + [node] + route[pos:]
                    feasible, _, _, _ = _simulate_route(new_route, instance, D)
                    if feasible:
                        best_gain = gain
                        best_node = node
                        best_pos = pos

        if best_node is not None:
            route.insert(best_pos, best_node)
            total_demand += demands[best_node]
            total_dist = sum(D[route[i], route[i+1]] for i in range(len(route)-1)) + D[0, route[0]] + D[route[-1], 0]
            remaining.remove(best_node)
            changed = True

    return route, total_dist


def _solve_instance(instance):
    coords = instance["coords"]
    D = _build_dist_matrix(coords)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    unassigned = set(customers)

    routes = []
    total_dist = 0.0

    while unassigned:
        route, dist = _greedy_build_route(list(unassigned), instance, D)
        if route is None:
            # Fallback: serve one customer at a time
            for c in sorted(unassigned, key=lambda v: instance["due"][v]):
                routes.append([c])
                total_dist += D[0, c] + D[c, 0]
                unassigned.discard(c)
            break

        routes.append(route)
        total_dist += dist
        for c in route:
            unassigned.discard(c)

    return routes, total_dist


def plan(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    coords = instance["coords"]
    D = _build_dist_matrix(coords)
    n_customers = instance["n_customers"]

    # First pass: greedy construction
    routes, total_dist = _solve_instance(instance)

    # Try to improve with local search if time permits
    elapsed = time.time() - start_time
    if elapsed < 60.0:
        # Try simple 2-opt / Or-opt style improvements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(routes)):
                route = routes[i]
                if len(route) < 2:
                    continue
                # Try reversing segments
                for a in range(len(route)):
                    for b in range(a + 2, len(route)):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        feasible, new_dist, _, _ = _simulate_route(new_route, instance, D)
                        if feasible and new_dist < total_dist - 1e-9:
                            routes[i] = new_route
                            total_dist = sum(
                                sum(D[route[j], route[j+1]] for j in range(len(route)-1)) + D[0, route[0]] + D[route[-1], 0]
                                for route in routes
                            )
                            improved = True

        # Try moving customers between routes
        if time.time() < deadline:
            for _ in range(50):  # limited iterations
                if time.time() > deadline:
                    break
                moved = False
                for i in range(len(routes)):
                    for j in range(len(routes)):
                        if i == j or not routes[i] or not routes[j]:
                            continue
                        for c in list(routes[i]):
                            new_i = [x for x in routes[i] if x != c]
                            new_j = list(routes[j])
                            # Try inserting c into j at best position
                            best_pos = None
                            best_gain = 0
                            dist_i = sum(D[new_i[k], new_i[k+1]] for k in range(len(new_i)-1)) + (D[0, new_i[0]] if new_i else 0) + (D[new_i[-1], 0] if new_i else 0)
                            for pos in range(len(new_j) + 1):
                                cand = new_j[:pos] + [c] + new_j[pos:]
                                feasible, _, _, _ = _simulate_route(cand, instance, D)
                                if not feasible:
                                    continue
                                dist_j = sum(D[cand[k], cand[k+1]] for k in range(len(cand)-1)) + D[0, cand[0]] + D[cand[-1], 0]
                                gain = (sum(D[r[k], r[k+1]] for k in range(len(r)-1)) + D[0, r[0]] + D[r[-1], 0] for r in [routes[i], routes[j]])
                                old_total = (sum(D[r[k], r[k+1]] for k in range(len(r)-1)) + D[0, r[0]] + D[r[-1], 0])
                                new_total = dist_i + dist_j
                                if new_total < old_total - 1e-9:
                                    best_gain = old_total - new_total
                                    best_pos = pos
                            if best_pos is not None:
                                routes[i] = new_i
                                routes[j] = new_j[:best_pos] + [c] + new_j[best_pos:]
                                total_dist = sum(
                                    sum(D[r[k], r[k+1]] for k in range(len(r)-1)) + D[0, r[0]] + D[r[-1], 0]
                                    for r in routes if r
                                )
                                moved = True
                                break
                        if moved:
                            break
                    if moved:
                        break
                if not moved:
                    break

    # Final validation
    for route in routes:
        feasible, _, _, _ = _simulate_route(route, instance, D)
        if not feasible:
            # Fallback to one-customer-per-route
            all_customers = list(range(1, n_customers + 1))
            all_customers.sort(key=lambda v: instance["due"][v])
            return [[v] for v in all_customers]

    # Remove empty routes
    routes = [r for r in routes if r]
    return routes
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    station_set = set(stations)
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))

    def fast_check_route(route):
        """Check if a route is feasible. Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0

        total_demand = 0.0
        for v in route:
            if v not in station_set:
                total_demand += demands[v]
        if total_demand > capacity + 1e-9:
            return False, 0.0

        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0

        for v in route:
            d = dist[prev, v]
            bat -= d
            if bat < -1e-9:
                return False, 0.0
            t += d / speed
            total_dist += d

            if v in station_set:
                charge_time = (battery - bat) * recharge_rate
                t += charge_time
                bat = battery
            else:
                if t < ready[v]:
                    t = ready[v]
                if t > due[v] + 1e-9:
                    return False, 0.0
                t += service[v]
            prev = v

        d = dist[prev, 0]
        bat -= d
        if bat < -1e-9:
            return False, 0.0
        t += d / speed
        total_dist += d

        if t > horizon + 1e-9:
            return False, 0.0

        return True, total_dist

    def total_route_distance(routes):
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += dist[0, route[0]]
            for i in range(len(route) - 1):
                total += dist[route[i], route[i + 1]]
            total += dist[route[-1], 0]
        return total

    def validate_routes(routes):
        all_customers = []
        for route in routes:
            feasible, _ = fast_check_route(route)
            if not feasible:
                return False
            all_customers.extend(route)
        if len(all_customers) != len(set(all_customers)):
            return False
        if set(all_customers) != set(customers):
            return False
        return True

    def savings_construct():
        """Clarke-Wright savings based construction with feasibility checks."""
        routes = [[v] for v in customers]
        route_of = {v: i for i, v in enumerate(customers)}
        route_load = [demands[v] for v in customers]

        savings = []
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                vi, vj = i + 1, j + 1
                s = dist[0, vi] + dist[0, vj] - dist[vi, vj]
                if s > 0.01:
                    savings.append((s, vi, vj))

        savings.sort(reverse=True)

        for s, vi, vj in savings:
            if time.time() > deadline:
                break
            ri = route_of.get(vi)
            rj = route_of.get(vj)
            if ri is None or rj is None or ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            if route_load[ri] + route_load[rj] > capacity + 1e-9:
                continue

            can_merge = False
            new_route = None

            options = []
            if route_i[-1] == vi and route_j[0] == vj:
                options.append(route_i + route_j)
            if route_j[-1] == vj and route_i[0] == vi:
                options.append(route_j + route_i)
            if route_i[0] == vi and route_j[0] == vj:
                options.append(route_i[::-1] + route_j)
            if route_i[-1] == vi and route_j[-1] == vj:
                options.append(route_i + route_j[::-1])
            if route_i[0] == vi and route_j[-1] == vj:
                options.append(route_i[::-1] + route_j[::-1])
            if route_j[0] == vj and route_i[-1] == vi:
                options.append(route_j + route_i[::-1])

            for candidate in options:
                f, _ = fast_check_route(candidate)
                if f:
                    can_merge = True
                    new_route = candidate
                    break

            if can_merge:
                routes[ri] = new_route
                route_load[ri] += route_load[rj]
                routes[rj] = []
                route_load[rj] = 0.0
                for v in new_route:
                    route_of[v] = ri
                non_empty = [r for r in routes if r]
                new_load = [route_load[i] for i, r in enumerate(routes) if r]
                for i, r in enumerate(non_empty):
                    for v in r:
                        route_of[v] = i
                routes = non_empty
                route_load = new_load

        return [r for r in routes if r]

    def greedy_construct(customer_order):
        """Construct routes using greedy nearest neighbor with time windows."""
        remaining = list(customer_order)
        routes = []

        while remaining:
            route = []
            prev = 0
            t = 0.0
            bat = battery
            load = 0.0

            while remaining:
                best_customer = None
                best_score = float('inf')
                best_idx = -1

                for idx, v in enumerate(remaining):
                    d = dist[prev, v]
                    bat_after = bat - d
                    t_after = t + d / speed

                    if bat_after < -1e-9:
                        # Try going through a station first
                        found = False
                        for st in stations:
                            d1 = dist[prev, st]
                            d2 = dist[st, v]
                            bat_st = bat - d1
                            if bat_st < -1e-9:
                                continue
                            t_st = t + d1 / speed
                            charge_time = (battery - bat_st) * recharge_rate
                            t_st += charge_time
                            t_v = t_st + d2 / speed
                            bat_v = battery - d2
                            if bat_v < -1e-9:
                                continue
                            if load + demands[v] > capacity + 1e-9:
                                continue
                            if t_v < ready[v]:
                                t_v = ready[v]
                            if t_v > due[v] + 1e-9:
                                continue
                            time_slack = due[v] - t_v
                            score = d1 + d2 - 0.02 * time_slack
                            if score < best_score:
                                best_score = score
                                best_customer = v
                                best_idx = idx
                                found = True
                        if not found:
                            continue
                        continue

                    if load + demands[v] > capacity + 1e-9:
                        continue

                    if t_after < ready[v]:
                        t_after = ready[v]
                    if t_after > due[v] + 1e-9:
                        continue

                    time_slack = due[v] - t_after
                    score = d - 0.02 * time_slack

                    if score < best_score:
                        best_score = score
                        best_customer = v
                        best_idx = idx

                if best_customer is None:
                    break

                v = best_customer
                d = dist[prev, v]
                bat -= d
                t += d / speed
                load += demands[v]

                if t < ready[v]:
                    t = ready[v]
                t += service[v]

                route.append(v)
                remaining.pop(best_idx)
                prev = v

            if route:
                routes.append(route)

        return routes

    def two_opt_route(route):
        """Apply 2-opt to a single route until no improvement."""
        if len(route) < 2:
            return route
        improved = True
        while improved:
            improved = False
            n = len(route)
            for a in range(n - 1):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n):
                    old_boundary = 0.0
                    if a > 0:
                        old_boundary += dist[route[a - 1], route[a]]
                    else:
                        old_boundary += dist[0, route[a]]
                    if b < n - 1:
                        old_boundary += dist[route[b], route[b + 1]]
                    else:
                        old_boundary += dist[route[b], 0]

                    new_boundary = 0.0
                    if a > 0:
                        new_boundary += dist[route[a - 1], route[b]]
                    else:
                        new_boundary += dist[0, route[b]]
                    if b < n - 1:
                        new_boundary += dist[route[a], route[b + 1]]
                    else:
                        new_boundary += dist[route[a], 0]

                    if new_boundary < old_boundary - 1e-6:
                        new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                        feasible, _ = fast_check_route(new_route)
                        if feasible:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    def relocate_search_fast(routes, max_passes=8):
        """Remove-and-reinsert local search with Or-opt (1-3 consecutive nodes)."""
        routes = [list(r) for r in routes]

        def compute_route_dist(route):
            if not route:
                return 0.0
            d = dist[0, route[0]]
            for i in range(len(route) - 1):
                d += dist[route[i], route[i + 1]]
            d += dist[route[-1], 0]
            return d

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            current_dist = sum(compute_route_dist(r) for r in routes)

            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for seg_len in [1, 2, 3]:
                    if time.time() > deadline:
                        break
                    for j in range(len(routes[i]) - seg_len + 1):
                        if time.time() > deadline:
                            break
                        segment = routes[i][j:j + seg_len]
                        route_i_without = routes[i][:j] + routes[i][j + seg_len:]

                        if not route_i_without:
                            continue

                        feasible_i, _ = fast_check_route(route_i_without)
                        if not feasible_i:
                            continue

                        old_src_dist = compute_route_dist(routes[i])
                        new_src_dist = compute_route_dist(route_i_without)

                        for k in range(len(routes)):
                            if time.time() > deadline:
                                break
                            if k == i:
                                continue
                            route_k = routes[k]
                            for pos in range(len(route_k) + 1):
                                new_route_k = route_k[:pos] + segment + route_k[pos:]
                                feasible, _ = fast_check_route(new_route_k)
                                if not feasible:
                                    continue

                                old_dst_dist = compute_route_dist(route_k)
                                new_dst_dist = compute_route_dist(new_route_k)

                                delta = (new_src_dist + new_dst_dist) - (old_src_dist + old_dst_dist)
                                if delta < -1e-6:
                                    routes[i] = route_i_without
                                    routes[k] = new_route_k
                                    current_dist += delta
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

    def cross_route_swap(routes):
        """Cross-route swap local search."""
        improved_swap = True
        while improved_swap and time.time() < deadline:
            improved_swap = False
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(routes)):
                    if time.time() > deadline:
                        continue
                    for a in range(len(routes[i])):
                        if time.time() > deadline:
                            break
                        for b in range(len(routes[j])):
                            vi = routes[i][a]
                            vj = routes[j][b]
                            if vi == vj:
                                continue
                            old_d = 0.0
                            if a > 0:
                                old_d += dist[routes[i][a - 1], vi]
                            else:
                                old_d += dist[0, vi]
                            if a < len(routes[i]) - 1:
                                old_d += dist[vi, routes[i][a + 1]]
                            else:
                                old_d += dist[vi, 0]
                            if b > 0:
                                old_d += dist[routes[j][b - 1], vj]
                            else:
                                old_d += dist[0, vj]
                            if b < len(routes[j]) - 1:
                                old_d += dist[vj, routes[j][b + 1]]
                            else:
                                old_d += dist[vj, 0]

                            new_d = 0.0
                            if a > 0:
                                new_d += dist[routes[i][a - 1], vj]
                            else:
                                new_d += dist[0, vj]
                            if a < len(routes[i]) - 1:
                                new_d += dist[vj, routes[i][a + 1]]
                            else:
                                new_d += dist[vj, 0]
                            if b > 0:
                                new_d += dist[routes[j][b - 1], vi]
                            else:
                                new_d += dist[0, vi]
                            if b < len(routes[j]) - 1:
                                new_d += dist[vi, routes[j][b + 1]]
                            else:
                                new_d += dist[vi, 0]

                            if new_d >= old_d - 1e-6:
                                continue

                            new_routes = [r[:] for r in routes]
                            new_routes[i][a] = vj
                            new_routes[j][b] = vi

                            f1, _ = fast_check_route(new_routes[i])
                            if not f1:
                                continue
                            f2, _ = fast_check_route(new_routes[j])
                            if not f2:
                                continue

                            routes = new_routes
                            improved_swap = True
                            break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
                if improved_swap:
                    break
        return routes

    def lns_destroy_reconstruct(routes, destroy_ratio=0.3):
        """Large Neighborhood Search: destroy a random subset, reconstruct with greedy insertion."""
        routes = [list(r) for r in routes]
        all_customers_in_routes = []
        for r in routes:
            all_customers_in_routes.extend(r)

        n_destroy = max(2, int(len(all_customers_in_routes) * destroy_ratio))
        n_destroy = min(n_destroy, len(all_customers_in_routes))

        # Randomly select customers to destroy
        to_destroy = random.sample(all_customers_in_routes, n_destroy)
        to_destroy_set = set(to_destroy)

        # Remove destroyed customers from routes
        new_routes = []
        for r in routes:
            new_r = [v for v in r if v not in to_destroy_set]
            if new_r:
                new_routes.append(new_r)

        # Reconstruct: insert destroyed customers one by one using greedy best insertion
        remaining = list(to_destroy)
        # Sort by due date for better insertion
        remaining.sort(key=lambda v: due[v])

        for v in remaining:
            if time.time() > deadline:
                break

            best_score = float('inf')
            best_route_idx = -1
            best_pos = -1

            for ri, route in enumerate(new_routes):
                for pos in range(len(route) + 1):
                    # Compute distance delta for insertion
                    if pos == 0:
                        if pos < len(route):
                            old_d = dist[0, route[0]] + dist[route[0], route[1] if len(route) > 1 else 0]
                        else:
                            old_d = 0.0
                        new_d = dist[0, v] + dist[v, route[0] if pos < len(route) else 0]
                    elif pos == len(route):
                        old_d = dist[route[-1], 0]
                        new_d = dist[route[-1], v] + dist[v, 0]
                    else:
                        old_d = dist[route[pos - 1], route[pos]]
                        new_d = dist[route[pos - 1], v] + dist[v, route[pos]]

                    delta = new_d - old_d
                    if delta >= best_score:
                        continue

                    # Check feasibility
                    new_route = route[:pos] + [v] + route[pos:]
                    feasible, _ = fast_check_route(new_route)
                    if not feasible:
                        continue

                    # Add a small penalty for time tightness
                    # Estimate arrival time at v
                    t = 0.0
                    bat = battery
                    prev = 0
                    for node in new_route:
                        d = dist[prev, node]
                        bat -= d
                        if bat < -1e-9:
                            break
                        t += d / speed
                        if node in station_set:
                            t += (battery - bat) * recharge_rate
                            bat = battery
                        else:
                            if t < ready[node]:
                                t = ready[node]
                            t += service[node]
                        prev = node
                    else:
                        # Check if v is in this route and find its time
                        pass

                    best_score = delta
                    best_route_idx = ri
                    best_pos = pos

            if best_route_idx >= 0:
                new_routes[best_route_idx] = new_routes[best_route_idx][:best_pos] + [v] + new_routes[best_route_idx][best_pos:]
            else:
                # Create a new route with just this customer
                new_routes.append([v])

        return new_routes

    # Initial solution: one customer per route (always feasible)
    best_routes = [[v] for v in customers]
    best_distance = sum(dist[0, v] * 2 for v in customers)

    # Try savings construction
    if time.time() < deadline:
        sw_routes = savings_construct()
        if validate_routes(sw_routes):
            d = total_route_distance(sw_routes)
            if d < best_distance:
                best_routes = sw_routes
                best_distance = d

    # Try greedy construction with different orderings
    orderings = [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: dist[0, v]),
        sorted(customers, key=lambda v: (due[v], dist[0, v])),
        sorted(customers, key=lambda v: (ready[v], dist[0, v])),
    ]

    for ordering in orderings:
        if time.time() > deadline:
            break
        greedy_routes = greedy_construct(ordering)
        if validate_routes(greedy_routes):
            d = total_route_distance(greedy_routes)
            if d < best_distance:
                best_routes = greedy_routes
                best_distance = d

    # Apply 2-opt to all routes
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    # Apply relocate search
    if time.time() < deadline:
        best_routes = relocate_search_fast(best_routes, max_passes=8)

    # Cross-route swap
    if time.time() < deadline:
        best_routes = cross_route_swap(best_routes)

    # LNS phase: repeatedly destroy and reconstruct
    lns_start = time.time()
    lns_deadline = deadline - 15.0  # Reserve 15s for final polish
    lns_iter = 0
    no_improve_count = 0

    while time.time() < lns_deadline:
        lns_iter += 1
        # Vary destroy ratio: start larger, decrease over time
        base_ratio = 0.3
        if lns_iter < 10:
            destroy_ratio = 0.4
        elif lns_iter < 30:
            destroy_ratio = 0.3
        elif lns_iter < 60:
            destroy_ratio = 0.2
        else:
            destroy_ratio = 0.15

        lns_routes = lns_destroy_reconstruct(best_routes, destroy_ratio)

        if not validate_routes(lns_routes):
            no_improve_count += 1
            continue

        # Polish the LNS result
        for i in range(len(lns_routes)):
            if time.time() > lns_deadline:
                break
            lns_routes[i] = two_opt_route(lns_routes[i])

        lns_routes = relocate_search_fast(lns_routes, max_passes=4)

        lns_distance = total_route_distance(lns_routes)

        if lns_distance < best_distance - 1e-6:
            best_routes = [list(r) for r in lns_routes]
            best_distance = lns_distance
            no_improve_count = 0
        else:
            no_improve_count += 1

        # If too many iterations without improvement, increase destruction
        if no_improve_count > 20:
            no_improve_count = 0

    # Final polish phase
    if time.time() < deadline - 5.0:
        # 2-opt on each route
        for i in range(len(best_routes)):
            if time.time() > deadline - 2.0:
                break
            best_routes[i] = two_opt_route(best_routes[i])

        # Relocate search
        if time.time() < deadline - 2.0:
            best_routes = relocate_search_fast(best_routes, max_passes=6)

        # Cross-route swap
        if time.time() < deadline - 2.0:
            best_routes = cross_route_swap(best_routes)

        # Final 2-opt
        if time.time() < deadline - 2.0:
            for i in range(len(best_routes)):
                if time.time() > deadline - 1.0:
                    break
                best_routes[i] = two_opt_route(best_routes[i])

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    # Final validation - if invalid, fall back to one-customer-per-route
    if not validate_routes(best_routes):
        best_routes = [[v] for v in customers]

    return best_routes
# EVOLVE-BLOCK-END

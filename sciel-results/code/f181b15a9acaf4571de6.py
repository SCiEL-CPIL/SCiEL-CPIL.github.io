"""EV routing with time windows and charging stations - LNS-based approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 135.0

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_list = list(stations)
    station_set = set(stations)

    def check_route(route):
        """Check if a route is feasible. Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0

        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity + 1e-9:
            return False, 0.0

        t = 0.0
        bat = battery_max
        prev = 0
        total_dist = 0.0

        for v in route:
            d = dist[prev, v]
            bat -= d
            t += d / speed
            total_dist += d

            if bat < -1e-9:
                return False, 0.0

            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
            prev = v

        d = dist[prev, 0]
        bat -= d
        t += d / speed
        total_dist += d

        if bat < -1e-9:
            return False, 0.0
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

    def greedy_insert_customer(route, customer, dist_matrix):
        """Find best position to insert customer into route. Returns (pos, feasible, new_dist)."""
        n = len(route)
        best_pos = -1
        best_delta = float('inf')

        for pos in range(n + 1):
            # Calculate distance change
            if n == 0:
                delta = dist_matrix[0, customer] + dist_matrix[customer, 0]
            else:
                if pos == 0:
                    delta = dist_matrix[0, customer] + dist_matrix[customer, route[0]] - dist_matrix[0, route[0]]
                elif pos == n:
                    delta = dist_matrix[route[-1], customer] + dist_matrix[customer, 0] - dist_matrix[route[-1], 0]
                else:
                    delta = dist_matrix[route[pos-1], customer] + dist_matrix[customer, route[pos]] - dist_matrix[route[pos-1], route[pos]]

            if delta < best_delta:
                best_delta = delta
                best_pos = pos

        # Check feasibility
        new_route = route[:best_pos] + [customer] + route[best_pos:]
        feasible, _ = check_route(new_route)

        return best_pos, feasible, best_delta

    def greedy_construct():
        """Construct initial solution using nearest neighbor with time windows."""
        remaining = list(customers)
        random.shuffle(remaining)
        routes = []

        while remaining:
            # Start new route
            route = []
            prev = 0
            t = 0.0
            bat = battery_max
            load = 0

            while remaining:
                best_customer = None
                best_score = float('inf')
                best_idx = -1

                for idx, v in enumerate(remaining):
                    d = dist[prev, v]
                    bat_after = bat - d
                    t_after = t + d / speed

                    if bat_after < -1e-9:
                        continue
                    if load + demands[v] > capacity + 1e-9:
                        continue

                    if t_after < ready[v]:
                        t_after = ready[v]
                    if t_after > due[v] + 1e-9:
                        continue

                    score = d + (t_after - ready[v]) * 0.03

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

    def validate_routes(routes):
        """Check if routes are valid."""
        all_customers = []
        for route in routes:
            feasible, _ = check_route(route)
            if not feasible:
                return False
            all_customers.extend(route)
        if len(all_customers) != len(set(all_customers)):
            return False
        if set(all_customers) != set(customers):
            return False
        return True

    def two_opt_route(route):
        """Apply 2-opt to a single route."""
        if len(route) < 2:
            return route
        improved = True
        iter_count = 0
        while improved and iter_count < 80 and time.time() < deadline:
            improved = False
            iter_count += 1
            n = len(route)
            for a in range(n - 1):
                if time.time() > deadline:
                    return route
                for b in range(a + 1, n):
                    old_boundary = 0.0
                    if a > 0:
                        old_boundary += dist[route[a-1], route[a]]
                    else:
                        old_boundary += dist[0, route[a]]
                    if b < n - 1:
                        old_boundary += dist[route[b], route[b+1]]
                    else:
                        old_boundary += dist[route[b], 0]

                    new_boundary = 0.0
                    if a > 0:
                        new_boundary += dist[route[a-1], route[b]]
                    else:
                        new_boundary += dist[0, route[b]]
                    if b < n - 1:
                        new_boundary += dist[route[a], route[b+1]]
                    else:
                        new_boundary += dist[route[a], 0]

                    if new_boundary < old_boundary - 1e-6:
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        feasible, _ = check_route(new_route)
                        if feasible:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    def or_opt_route(route):
        """Apply or-opt to a single route."""
        if len(route) < 3:
            return route
        improved = True
        iter_count = 0
        while improved and iter_count < 50 and time.time() < deadline:
            improved = False
            iter_count += 1
            n = len(route)
            for i in range(n):
                if time.time() > deadline:
                    return route
                v = route[i]
                route_without = route[:i] + route[i+1:]
                if not route_without:
                    continue

                best_pos = -1
                best_delta = 0.0

                old_cost = 0.0
                if i > 0:
                    old_cost += dist[route[i-1], v]
                else:
                    old_cost += dist[0, v]
                if i < n - 1:
                    old_cost += dist[v, route[i+1]]
                else:
                    old_cost += dist[v, 0]
                if i > 0 and i < n - 1:
                    old_cost -= dist[route[i-1], route[i+1]]
                elif i == 0 and n > 1:
                    old_cost -= dist[0, route[1]]
                elif i == n - 1 and n > 1:
                    old_cost -= dist[route[n-2], 0]

                for pos in range(len(route_without) + 1):
                    new_cost = 0.0
                    if pos > 0:
                        new_cost += dist[route_without[pos-1], v]
                    else:
                        new_cost += dist[0, v]
                    if pos < len(route_without):
                        new_cost += dist[v, route_without[pos]]
                    else:
                        new_cost += dist[v, 0]
                    if pos > 0 and pos < len(route_without):
                        new_cost -= dist[route_without[pos-1], route_without[pos]]
                    elif pos == 0 and len(route_without) > 0:
                        new_cost -= dist[0, route_without[0]]
                    elif pos == len(route_without) and len(route_without) > 0:
                        new_cost -= dist[route_without[-1], 0]

                    delta = new_cost - old_cost
                    if delta < best_delta - 1e-6:
                        best_delta = delta
                        best_pos = pos

                if best_pos >= 0 and best_delta < -1e-6:
                    new_route = route_without[:best_pos] + [v] + route_without[best_pos:]
                    feasible, _ = check_route(new_route)
                    if feasible:
                        route = new_route
                        improved = True
                        break
        return route

    def lns_destroy_reinsert(routes, destroy_count, max_passes=3):
        """Large Neighborhood Search: destroy a subset and reinsert."""
        routes = [list(r) for r in routes]
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            current_dist = total_route_distance(routes)

            # Collect all customers with their route and position
            all_customers = []
            for ri, route in enumerate(routes):
                for pi, c in enumerate(route):
                    all_customers.append((c, ri, pi))

            # Destroy: remove random subset
            k = min(destroy_count, len(all_customers))
            if k == 0:
                break
            removed = random.sample(all_customers, k)
            removed_set = set((c, ri, pi) for c, ri, pi in removed)

            # Remove from routes
            for c, ri, pi in removed:
                routes[ri].remove(c)

            # Remove empty routes
            routes = [r for r in routes if r]

            # Reinsert each removed customer
            removed_customers = [c for c, ri, pi in removed]
            random.shuffle(removed_customers)

            for c in removed_customers:
                if time.time() > deadline:
                    break
                best_route = -1
                best_pos = -1
                best_delta = float('inf')

                for ri, route in enumerate(routes):
                    pos, feasible, delta = greedy_insert_customer(route, c, dist)
                    if feasible and delta < best_delta:
                        best_delta = delta
                        best_route = ri
                        best_pos = pos

                # If no existing route works, create new route
                if best_route == -1:
                    new_route = [c]
                    feasible, _ = check_route(new_route)
                    if feasible:
                        routes.append(new_route)
                        improved = True
                else:
                    routes[best_route].insert(best_pos, c)
                    improved = True

            # Check if improved
            new_dist = total_route_distance(routes)
            if new_dist < current_dist - 1e-6:
                improved = True
            else:
                # Revert: this is simplified, just keep going
                pass

            if not improved:
                break

        return routes

    def relocate_search(routes, max_passes=5):
        """Remove-and-reinsert local search."""
        routes = [list(r) for r in routes]
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            current_dist = total_route_distance(routes)

            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for j in range(len(routes[i])):
                    if time.time() > deadline:
                        break
                    v = routes[i][j]
                    route_i_without = routes[i][:j] + routes[i][j+1:]
                    if not route_i_without:
                        new_routes = [r for idx, r in enumerate(routes) if idx != i]
                    else:
                        new_routes = list(routes)
                        new_routes[i] = route_i_without

                    if route_i_without:
                        feasible_i, _ = check_route(route_i_without)
                        if not feasible_i:
                            continue

                    for k in range(len(new_routes)):
                        if time.time() > deadline:
                            break
                        route_k = new_routes[k]
                        for pos in range(len(route_k) + 1):
                            new_route_k = route_k[:pos] + [v] + route_k[pos:]
                            feasible, _ = check_route(new_route_k)
                            if not feasible:
                                continue

                            new_routes[k] = new_route_k
                            new_dist = total_route_distance(new_routes)
                            if new_dist < current_dist - 1e-6:
                                routes = new_routes
                                current_dist = new_dist
                                improved = True
                                break
                            else:
                                new_routes[k] = route_k
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes

    def swap_search(routes, max_passes=3):
        """Cross-route swap local search."""
        routes = [list(r) for r in routes]
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            current_dist = total_route_distance(routes)
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
                            new_routes = []
                            for r in range(len(routes)):
                                if r == i:
                                    nr = routes[i][:]
                                    nr[a] = vj
                                    new_routes.append(nr)
                                elif r == j:
                                    nr = routes[j][:]
                                    nr[b] = vi
                                    new_routes.append(nr)
                                else:
                                    new_routes.append(routes[r][:])

                            feasible = True
                            for route in new_routes:
                                f, _ = check_route(route)
                                if not f:
                                    feasible = False
                                    break

                            if feasible:
                                new_dist = total_route_distance(new_routes)
                                if new_dist < current_dist - 1e-6:
                                    routes = new_routes
                                    current_dist = new_dist
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

    # Generate multiple initial solutions
    best_routes = None
    best_distance = float('inf')

    for trial in range(8):
        if time.time() > deadline:
            break
        random.shuffle(customers)
        greedy_routes = greedy_construct()
        if validate_routes(greedy_routes):
            d = total_route_distance(greedy_routes)
            if d < best_distance:
                best_routes = [list(r) for r in greedy_routes]
                best_distance = d

    if best_routes is None:
        best_routes = [[v] for v in customers]
        best_distance = sum(dist[0, v] * 2 for v in customers)

    # Local search phases
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = or_opt_route(best_routes[i])

    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=5)

    if time.time() < deadline:
        best_routes = swap_search(best_routes, max_passes=3)

    # LNS iterations
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        num_lns = min(15, int(remaining_time / 5))
        for _ in range(num_lns):
            if time.time() > deadline:
                break
            # Destroy 3-5 customers
            destroy_k = random.randint(3, 5)
            best_routes = lns_destroy_reinsert(best_routes, destroy_k, max_passes=2)

            # Quick local search after LNS
            if time.time() < deadline - 3:
                for i in range(len(best_routes)):
                    if time.time() > deadline - 2:
                        break
                    best_routes[i] = two_opt_route(best_routes[i])

    # Final polish
    if time.time() < deadline - 2:
        for i in range(len(best_routes)):
            if time.time() > deadline - 2:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    if time.time() < deadline - 1:
        for i in range(len(best_routes)):
            if time.time() > deadline - 1:
                break
            best_routes[i] = or_opt_route(best_routes[i])

    if time.time() < deadline - 1:
        best_routes = relocate_search(best_routes, max_passes=3)

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    return best_routes
# EVOLVE-BLOCK-END

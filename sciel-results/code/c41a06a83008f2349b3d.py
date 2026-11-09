"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix with NumPy
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    # Precompute "latest start time" for each node given current battery level
    # For a move from prev to v: we need bat >= dist[prev,v]
    # After arriving: t_arrive = t + dist[prev,v]/speed
    # If t_arrive < ready[v]: t_arrive = ready[v]
    # Need t_arrive <= due[v]
    # So: t + dist[prev,v]/speed <= due[v]  (if no waiting)
    # Or: ready[v] <= due[v] (always true) and t + dist[prev,v]/speed can be anything < ready[v]
    # Actually: max(t + dist[prev,v]/speed, ready[v]) <= due[v]
    # This means: t + dist[prev,v]/speed <= due[v]  OR  ready[v] <= due[v] (which is always true)
    # Wait, no: if t + dist/speed < ready[v], then arrival is at ready[v], which must be <= due[v]
    # If t + dist/speed >= ready[v], then arrival is at t+dist/speed, which must be <= due[v]
    # So the constraint is: t + dist[prev,v]/speed <= due[v]
    # Because if t+dist/speed < ready[v], we wait until ready[v] which is <= due[v]
    # If t+dist/speed >= ready[v], we arrive at t+dist/speed which must be <= due[v]
    # So in both cases: t + dist[prev,v]/speed <= due[v]
    # Therefore: t <= due[v] - dist[prev,v]/speed
    # This is the latest time we can depart prev to visit v.

    # Precompute latest_departure[prev][v] = due[v] - dist[prev,v]/speed
    # But we also need to check battery: bat >= dist[prev,v]
    # And after service: t_after = max(t + dist/speed, ready[v]) + service[v]
    # The departure time from prev must satisfy t <= due[v] - dist[prev,v]/speed

    # For O(1) feasibility checks in local search, we precompute:
    # latest_dep[prev][v] = due[v] - dist[prev,v] / speed
    latest_dep = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            latest_dep[i, j] = due[j] - dist[i, j] / speed

    # Precompute battery feasibility: can_move[prev][v] = (dist[prev,v] <= battery)
    # Since battery is constant per route, we just need dist[prev,v] <= battery
    can_move = dist <= battery

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

    def fast_check_route(route):
        """Faster feasibility check with early termination."""
        if not route:
            return True, 0.0

        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
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

    def fast_check_route_v2(route, load):
        """Vectorized-style feasibility check using precomputed latest_dep.
        load is precomputed total demand for the route."""
        if not route:
            return True, 0.0

        if load > capacity:
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

            # Use precomputed latest departure check
            # We need: t_depart <= latest_dep[prev][v]
            # t_depart is the time before traveling to v
            # t_depart = t - d/speed (before adding travel time)
            # Actually t is now arrival time. Let's track departure time.
            # Let's restructure: track departure time from prev
            pass

        # The key insight: for each leg prev->v, we need departure_time_from_prev <= latest_dep[prev][v]
        # departure_time_from_prev is the time when we leave prev
        # Let's track this properly
        
        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0

        for v in route:
            # Check if we can depart prev at time t to reach v in time
            if t > latest_dep[prev, v] + 1e-9:
                return False, 0.0
            
            d = dist[prev, v]
            bat -= d
            if bat < -1e-9:
                return False, 0.0
            
            # Arrival time at v
            t_arrive = t + d / speed
            if t_arrive < ready[v]:
                t_arrive = ready[v]
            t = t_arrive + service[v]
            total_dist += d
            prev = v

        # Return to depot
        d = dist[prev, 0]
        bat -= d
        if bat < -1e-9:
            return False, 0.0
        t += d / speed
        total_dist += d

        if t > horizon + 1e-9:
            return False, 0.0

        return True, total_dist

    def savings_construct():
        """Clarke-Wright savings based construction with feasibility checks."""
        routes = [[v] for v in customers]
        route_of = {v: i for i, v in enumerate(customers)}
        route_load = {i: demands[v] for i, v in enumerate(customers)}

        savings = []
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                vi, vj = i + 1, j + 1
                s = dist[0, vi] + dist[0, vj] - dist[vi, vj]
                if s > 0:
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

            can_merge = False
            new_route = None

            # Try all four merge options
            options = []
            if route_i[-1] == vi and route_j[0] == vj:
                options.append((route_i + route_j, route_load[ri] + route_load[rj]))
            if route_j[-1] == vj and route_i[0] == vi:
                options.append((route_j + route_i, route_load[rj] + route_load[ri]))
            if route_i[0] == vi and route_j[0] == vj:
                options.append((route_i[::-1] + route_j, route_load[ri] + route_load[rj]))
            if route_i[-1] == vi and route_j[-1] == vj:
                options.append((route_i + route_j[::-1], route_load[ri] + route_load[rj]))
            if route_i[0] == vi and route_j[-1] == vj:
                options.append((route_i[::-1] + route_j[::-1], route_load[ri] + route_load[rj]))
            if route_j[0] == vj and route_i[-1] == vi:
                options.append((route_j + route_i[::-1], route_load[rj] + route_load[ri]))

            for candidate, load in options:
                if load > capacity:
                    continue
                f, _ = fast_check_route_v2(candidate, load)
                if f:
                    can_merge = True
                    new_route = candidate
                    new_load = load
                    break

            if can_merge:
                routes[ri] = new_route
                routes[rj] = []
                route_load[ri] = new_load
                route_load[rj] = 0
                for v in new_route:
                    route_of[v] = ri
                # Clean up empty routes
                non_empty = [r for r in routes if r]
                new_route_load = {}
                for i, r in enumerate(non_empty):
                    for v in r:
                        route_of[v] = i
                    new_route_load[i] = sum(demands[v] for v in r)
                routes = non_empty
                route_load = new_route_load

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
            load = 0

            while remaining:
                best_customer = None
                best_score = float('inf')
                best_idx = -1

                for idx, v in enumerate(remaining):
                    d = dist[prev, v]
                    bat_after = bat - d
                    if bat_after < -1e-9:
                        continue
                    if load + demands[v] > capacity:
                        continue

                    # Check if we can depart prev at time t to reach v in time
                    if t > latest_dep[prev, v] + 1e-9:
                        continue

                    t_after = t + d / speed
                    if t_after < ready[v]:
                        t_after = ready[v]
                    t_after += service[v]

                    # Score: prioritize distance, then time tightness
                    time_slack = due[v] - (t + d / speed)
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

    def validate_routes(routes):
        """Check if routes are valid: all customers served exactly once, all feasible."""
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

    def two_opt_route(route):
        """Apply 2-opt to a single route until no improvement."""
        if len(route) < 2:
            return route
        load = sum(demands[v] for v in route)
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
                        feasible, _ = fast_check_route_v2(new_route, load)
                        if feasible:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    # Apply 2-opt to all routes
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    def relocate_search(routes, max_passes=8):
        """Remove-and-reinsert local search with Or-opt (1-3 consecutive nodes)."""
        routes = [list(r) for r in routes]
        route_loads = [sum(demands[v] for v in r) for r in routes]
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            current_dist = total_route_distance(routes)

            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                # Try Or-opt: relocate 1, 2, or 3 consecutive nodes
                for seg_len in [1, 2, 3]:
                    if time.time() > deadline:
                        break
                    for j in range(len(routes[i]) - seg_len + 1):
                        if time.time() > deadline:
                            break
                        segment = routes[i][j:j+seg_len]
                        seg_load = sum(demands[v] for v in segment)
                        route_i_without = routes[i][:j] + routes[i][j+seg_len:]
                        
                        if not route_i_without:
                            new_routes = [r for idx, r in enumerate(routes) if idx != i]
                            new_loads = [l for idx, l in enumerate(route_loads) if idx != i]
                            src_idx = -1
                        else:
                            new_routes = list(routes)
                            new_loads = list(route_loads)
                            new_routes[i] = route_i_without
                            new_loads[i] = route_loads[i] - seg_load
                            src_idx = i

                        if route_i_without:
                            feasible_i, _ = fast_check_route_v2(route_i_without, new_loads[i] if src_idx >= 0 else 0)
                            if not feasible_i:
                                continue
                        else:
                            feasible_i = True

                        for k in range(len(new_routes)):
                            if time.time() > deadline:
                                break
                            route_k = new_routes[k]
                            for pos in range(len(route_k) + 1):
                                new_route_k = route_k[:pos] + segment + route_k[pos:]
                                new_load_k = new_loads[k] + seg_load
                                feasible, _ = fast_check_route_v2(new_route_k, new_load_k)
                                if not feasible:
                                    continue

                                new_routes[k] = new_route_k
                                new_loads[k] = new_load_k
                                new_dist = total_route_distance(new_routes)
                                if new_dist < current_dist - 1e-6:
                                    routes = new_routes
                                    route_loads = new_loads
                                    current_dist = new_dist
                                    improved = True
                                    break
                                else:
                                    new_routes[k] = route_k
                                    new_loads[k] = new_loads[k] - seg_load
                            if improved:
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

    # Apply relocate search
    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=5)

    # Cross-route swap local search (only check 2 modified routes)
    if time.time() < deadline:
        improved_swap = True
        while improved_swap and time.time() < deadline:
            improved_swap = False
            current_dist = total_route_distance(best_routes)
            route_loads = [sum(demands[v] for v in r) for r in best_routes]
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best_routes)):
                    if time.time() > deadline:
                        continue
                    for a in range(len(best_routes[i])):
                        if time.time() > deadline:
                            break
                        for b in range(len(best_routes[j])):
                            vi = best_routes[i][a]
                            vj = best_routes[j][b]
                            if vi == vj:
                                continue
                            # Compute distance delta without full reconstruction
                            # Old contributions involving vi and vj
                            old_d = 0.0
                            # vi in route i at position a
                            if a > 0:
                                old_d += dist[best_routes[i][a-1], vi]
                            else:
                                old_d += dist[0, vi]
                            if a < len(best_routes[i]) - 1:
                                old_d += dist[vi, best_routes[i][a+1]]
                            else:
                                old_d += dist[vi, 0]
                            # vj in route j at position b
                            if b > 0:
                                old_d += dist[best_routes[j][b-1], vj]
                            else:
                                old_d += dist[0, vj]
                            if b < len(best_routes[j]) - 1:
                                old_d += dist[vj, best_routes[j][b+1]]
                            else:
                                old_d += dist[vj, 0]
                            
                            # New contributions
                            new_d = 0.0
                            # vj replaces vi in route i at position a
                            if a > 0:
                                new_d += dist[best_routes[i][a-1], vj]
                            else:
                                new_d += dist[0, vj]
                            if a < len(best_routes[i]) - 1:
                                new_d += dist[vj, best_routes[i][a+1]]
                            else:
                                new_d += dist[vj, 0]
                            # vi replaces vj in route j at position b
                            if b > 0:
                                new_d += dist[best_routes[j][b-1], vi]
                            else:
                                new_d += dist[0, vi]
                            if b < len(best_routes[j]) - 1:
                                new_d += dist[vi, best_routes[j][b+1]]
                            else:
                                new_d += dist[vi, 0]
                            
                            if new_d >= old_d - 1e-6:
                                continue
                            
                            # Build new routes
                            new_routes = [r[:] for r in best_routes]
                            new_routes[i][a] = vj
                            new_routes[j][b] = vi
                            
                            new_load_i = route_loads[i] - demands[vi] + demands[vj]
                            new_load_j = route_loads[j] - demands[vj] + demands[vi]
                            
                            if new_load_i > capacity or new_load_j > capacity:
                                continue

                            # Only check the two modified routes
                            f1, _ = fast_check_route_v2(new_routes[i], new_load_i)
                            if not f1:
                                continue
                            f2, _ = fast_check_route_v2(new_routes[j], new_load_j)
                            if not f2:
                                continue
                            
                            best_routes = new_routes
                            current_dist = total_route_distance(best_routes)
                            improved_swap = True
                            break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
                if improved_swap:
                    break

    # Final polish: multiple rounds of 2-opt + relocate + swap
    polish_rounds = 3
    for round_num in range(polish_rounds):
        if time.time() > deadline:
            break
        
        # 2-opt on each route
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])
        
        # Relocate search
        if time.time() < deadline:
            best_routes = relocate_search(best_routes, max_passes=4)
        
        # Cross-route swap
        if time.time() < deadline:
            improved_swap = True
            while improved_swap and time.time() < deadline:
                improved_swap = False
                current_dist = total_route_distance(best_routes)
                route_loads = [sum(demands[v] for v in r) for r in best_routes]
                for i in range(len(best_routes)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(best_routes)):
                        if time.time() > deadline:
                            continue
                        for a in range(len(best_routes[i])):
                            if time.time() > deadline:
                                break
                            for b in range(len(best_routes[j])):
                                vi = best_routes[i][a]
                                vj = best_routes[j][b]
                                if vi == vj:
                                    continue
                                old_d = 0.0
                                if a > 0:
                                    old_d += dist[best_routes[i][a-1], vi]
                                else:
                                    old_d += dist[0, vi]
                                if a < len(best_routes[i]) - 1:
                                    old_d += dist[vi, best_routes[i][a+1]]
                                else:
                                    old_d += dist[vi, 0]
                                if b > 0:
                                    old_d += dist[best_routes[j][b-1], vj]
                                else:
                                    old_d += dist[0, vj]
                                if b < len(best_routes[j]) - 1:
                                    old_d += dist[vj, best_routes[j][b+1]]
                                else:
                                    old_d += dist[vj, 0]
                                
                                new_d = 0.0
                                if a > 0:
                                    new_d += dist[best_routes[i][a-1], vj]
                                else:
                                    new_d += dist[0, vj]
                                if a < len(best_routes[i]) - 1:
                                    new_d += dist[vj, best_routes[i][a+1]]
                                else:
                                    new_d += dist[vj, 0]
                                if b > 0:
                                    new_d += dist[best_routes[j][b-1], vi]
                                else:
                                    new_d += dist[0, vi]
                                if b < len(best_routes[j]) - 1:
                                    new_d += dist[vi, best_routes[j][b+1]]
                                else:
                                    new_d += dist[vi, 0]
                                
                                if new_d >= old_d - 1e-6:
                                    continue
                                
                                new_routes = [r[:] for r in best_routes]
                                new_routes[i][a] = vj
                                new_routes[j][b] = vi
                                
                                new_load_i = route_loads[i] - demands[vi] + demands[vj]
                                new_load_j = route_loads[j] - demands[vj] + demands[vi]
                                
                                if new_load_i > capacity or new_load_j > capacity:
                                    continue

                                f1, _ = fast_check_route_v2(new_routes[i], new_load_i)
                                if not f1:
                                    continue
                                f2, _ = fast_check_route_v2(new_routes[j], new_load_j)
                                if not f2:
                                    continue
                                
                                best_routes = new_routes
                                current_dist = total_route_distance(best_routes)
                                improved_swap = True
                                break
                            if improved_swap:
                                break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
                if improved_swap:
                    break
        
        # Final 2-opt after swap
        if time.time() < deadline:
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                best_routes[i] = two_opt_route(best_routes[i])

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    return best_routes
# EVOLVE-BLOCK-END

<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def plan(instance):
    start_time = time.time()
    deadline = start_time + 125.0  # Leave 25s safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def simulate_route(route):
        """
        Simulate a route (list of node ids - customers and/or stations).
        Returns (feasible, total_dist, end_time, end_battery)
        """
        if not route:
            return True, 0.0, 0.0, battery_cap

        cur_time = 0.0
        cur_battery = battery_cap
        total_dist = 0.0
        cur_pos = 0

        for node in route:
            d = dist[cur_pos][node]
            cur_time += d / speed
            cur_battery -= d
            total_dist += d

            if cur_battery < -1e-9:
                return False, total_dist, cur_time, cur_battery

            if node in station_set:
                # Recharge: adds (battery_cap - cur_battery) * recharge_rate time, then refills
                charge_time = (battery_cap - cur_battery) * recharge_rate
                cur_time += charge_time
                cur_battery = battery_cap
                cur_pos = node
            else:
                # Customer
                cur_time = max(cur_time, ready[node])
                if cur_time > due[node] + 1e-9:
                    return False, total_dist, cur_time, cur_battery
                cur_time += service[node]
                cur_pos = node

        # Return to depot
        d = dist[cur_pos][0]
        cur_time += d / speed
        cur_battery -= d
        total_dist += d

        if cur_battery < -1e-9:
            return False, total_dist, cur_time, cur_battery

        if cur_time > horizon + 1e-9:
            return False, total_dist, cur_time, cur_battery

        return True, total_dist, cur_time, cur_battery

    def route_total_dist(route):
        if not route:
            return 0.0
        d = dist[0][route[0]]
        for i in range(len(route) - 1):
            d += dist[route[i]][route[i + 1]]
        d += dist[route[-1]][0]
        return d

    def route_demand(route):
        return sum(demands[c] for c in route if c not in station_set)

    def try_insert_station(route, pos, station_id):
        """Try inserting a station at position pos in route. Returns new route if feasible."""
        new_route = route[:pos] + [station_id] + route[pos:]
        feasible, _, _, _ = simulate_route(new_route)
        if feasible and route_demand(new_route) <= capacity:
            return new_route
        return None

    def best_station_insertion(route):
        """
        Given a route, find the best single station insertion that improves feasibility
        or is needed. Returns (new_route, station_id) or (route, None).
        """
        # Check if route is already feasible
        feasible, _, _, _ = simulate_route(route)
        if feasible:
            return route, None

        # Try inserting each station at each position
        best_new = None
        best_station = None
        best_dist = float('inf')

        for st in stations:
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [st] + route[pos:]
                # Quick check: demand unchanged
                if route_demand(new_route) > capacity:
                    continue
                f, d, _, _ = simulate_route(new_route)
                if f and d < best_dist:
                    best_dist = d
                    best_new = new_route
                    best_station = st

        if best_new is not None:
            return best_new, best_station
        return route, None

    # Construction: Savings heuristic
    def construct_savings():
        """
        Clarke-Wright savings construction with battery and time window constraints.
        """
        # Start with each customer in its own route
        routes = [[c] for c in customers]

        # Compute savings
        savings = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
                savings.append((s, i, j))

        savings.sort(reverse=True)

        # Map customer to route index
        cust_to_route = {c: i for i, c in enumerate(customers)}

        for s_val, i, j in savings:
            if time.time() > deadline:
                break
            ci = customers[i]
            cj = customers[j]

            ri = cust_to_route.get(ci)
            rj = cust_to_route.get(cj)
            if ri is None or rj is None or ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            # Check capacity
            if route_demand(route_i) + route_demand(route_j) > capacity:
                continue

            # Try merging: append route_j to route_i (try both orientations)
            merged = None
            for orient_i in [route_i, route_i[::-1]]:
                for orient_j in [route_j, route_j[::-1]]:
                    candidate = orient_i + orient_j
                    feasible, _, _, _ = simulate_route(candidate)
                    if feasible:
                        merged = candidate
                        break
                if merged:
                    break

            if merged is None:
                # Try with station insertion
                for orient_i in [route_i, route_i[::-1]]:
                    for orient_j in [route_j, route_j[::-1]]:
                        candidate = orient_i + orient_j
                        # Try inserting a station
                        for st in stations:
                            for pos in range(len(candidate) + 1):
                                test_route = candidate[:pos] + [st] + candidate[pos:]
                                if route_demand(test_route) > capacity:
                                    continue
                                f, _, _, _ = simulate_route(test_route)
                                if f:
                                    merged = test_route
                                    break
                            if merged:
                                break
                        if merged:
                            break
                    if merged:
                        break

            if merged is not None:
                routes[ri] = merged
                routes[rj] = []
                for c in merged:
                    if c not in station_set:
                        cust_to_route[c] = ri

        # Remove empty routes
        routes = [r for r in routes if r]

        # Ensure all customers are present
        all_served = set()
        for r in routes:
            all_served.update(c for c in r if c not in station_set)
        missing = set(customers) - all_served
        for m in missing:
            routes.append([m])

        return routes

    def construct_nearest_neighbor():
        """Nearest neighbor construction with constraints."""
        unvisited = set(customers)
        routes = []

        while unvisited:
            if time.time() > deadline:
                break
            route = []
            cur_pos = 0
            cur_time = 0.0
            cur_battery = battery_cap
            route_dem = 0

            while unvisited:
                best_cust = None
                best_score = float('inf')

                for cust in unvisited:
                    d = dist[cur_pos][cust]
                    new_time = cur_time + d / speed
                    new_battery = cur_battery - d

                    if new_battery < -1e-9:
                        continue
                    if route_dem + demands[cust] > capacity:
                        continue

                    new_time = max(new_time, ready[cust])
                    if new_time > due[cust] + 1e-9:
                        continue

                    # Check return to depot
                    return_d = dist[cust][0]
                    if new_battery - return_d < -1e-9:
                        continue
                    if new_time + service[cust] + return_d / speed > horizon + 1e-9:
                        continue

                    if d < best_score:
                        best_score = d
                        best_cust = cust

                if best_cust is None:
                    break

                d = dist[cur_pos][best_cust]
                cur_time += d / speed
                cur_battery -= d
                cur_time = max(cur_time, ready[best_cust])
                cur_time += service[best_cust]
                route_dem += demands[best_cust]
                route.append(best_cust)
                unvisited.remove(best_cust)
                cur_pos = best_cust

            if route:
                feasible, _, _, _ = simulate_route(route)
                if feasible:
                    routes.append(route)
                else:
                    # Try with station
                    fixed_route, _ = best_station_insertion(route)
                    if simulate_route(fixed_route)[0]:
                        routes.append(fixed_route)
                    else:
                        # Split into singles
                        for c in route:
                            routes.append([c])
                            unvisited.add(c)
            else:
                # Try single customer
                for cust in list(unvisited):
                    one_route = [cust]
                    feasible, _, _, _ = simulate_route(one_route)
                    if feasible:
                        routes.append(one_route)
                        unvisited.remove(cust)
                        break
                else:
                    # Try with station
                    for cust in list(unvisited):
                        for st in stations:
                            test = [st, cust]
                            if simulate_route(test)[0]:
                                routes.append(test)
                                unvisited.remove(cust)
                                break
                            test2 = [cust, st]
                            if simulate_route(test2)[0]:
                                routes.append(test2)
                                unvisited.remove(cust)
                                break

        # Handle remaining
        if unvisited:
            for cust in list(unvisited):
                routes.append([cust])

        return routes

    # Try both constructions, pick better
    routes1 = construct_savings()
    dist1 = sum(route_total_dist(r) for r in routes1)

    if time.time() < deadline:
        routes2 = construct_nearest_neighbor()
        dist2 = sum(route_total_dist(r) for r in routes2)
        if dist2 < dist1:
            routes = routes2
        else:
            routes = routes1
    else:
        routes = routes1

    # Local search improvements
    def or_opt_move(routes, i, j, k):
        """
        Try moving segment routes[i][j:k] (length 1-3) to another route or position.
        Returns (improved, new_routes) or (False, routes).
        """
        route = routes[i]
        if k - j > 3 or k > len(route) or j < 0:
            return False, routes

        segment = route[j:k]
        # Filter to only customers (stations stay in place)
        custs_in_seg = [c for c in segment if c not in station_set]
        if not custs_in_seg:
            return False, routes

        # Remove segment from route i
        new_route_i = route[:j] + route[k:]
        feasible_i, _, _, _ = simulate_route(new_route_i)
        if not feasible_i:
            return False, routes

        old_total = route_total_dist(route)

        # Try inserting into same route at different position
        for pos in range(len(new_route_i) + 1):
            if pos >= j and pos <= k:
                continue
            new_route_i_test = new_route_i[:pos] + segment + new_route_i[pos:]
            if route_demand(new_route_i_test) > capacity:
                continue
            f, d, _, _ = simulate_route(new_route_i_test)
            if f and d < old_total - 1e-6:
                routes[i] = new_route_i_test
                return True, routes

        # Try inserting into other routes
        for k2 in range(len(routes)):
            if k2 == i:
                continue
            route_k = routes[k2]
            for pos in range(len(route_k) + 1):
                new_route_k = route_k[:pos] + segment + route_k[pos:]
                if route_demand(new_route_k) > capacity:
                    continue
                f, d, _, _ = simulate_route(new_route_k)
                if not f:
                    continue
                new_dist_total = (route_total_dist(new_route_i) if new_route_i else 0) + d
                old_dist_total = (route_total_dist(new_route_i) if new_route_i else 0) + route_total_dist(route_k)
                if new_dist_total < old_dist_total - 1e-6:
                    if not new_route_i:
                        routes[i] = []
                    else:
                        routes[i] = new_route_i
                    routes[k2] = new_route_k
                    return True, routes

        return False, routes

    def two_opt_intra(routes, i):
        """2-opt within route i."""
        route = routes[i]
        if len(route) < 3:
            return False

        best_improvement = 0
        best_a, best_b = -1, -1
        old_d = route_total_dist(route)

        for a in range(len(route) - 1):
            for b in range(a + 1, len(route)):
                new_route = route[:a + 1] + route[a + 1:b + 1][::-1] + route[b + 1:]
                f, d, _, _ = simulate_route(new_route)
                if f:
                    imp = old_d - d
                    if imp > best_improvement:
                        best_improvement = imp
                        best_a, best_b = a, b

        if best_improvement > 1e-6:
            a, b = best_a, best_b
            routes[i] = route[:a + 1] + route[a + 1:b + 1][::-1] + route[b + 1:]
            return True
        return False

    def try_improve(routes):
        """Iterative local search."""
        improved = True
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1

            # 2-opt intra
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                if two_opt_intra(routes, i):
                    improved = True

            # Or-opt moves (1, 2, 3 consecutive)
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                route = routes[i]
                for j in range(len(route)):
                    for length in [1, 2, 3]:
                        k = j + length
                        if k > len(route):
                            continue
                        imp, routes = or_opt_move(routes, i, j, k)
                        if imp:
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            # Station insertion: if any route is infeasible, try to fix
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                f, _, _, _ = simulate_route(routes[i])
                if not f:
                    fixed, _ = best_station_insertion(routes[i])
                    if simulate_route(fixed)[0]:
                        routes[i] = fixed
                        improved = True

            # Remove empty routes
            new_routes = [r for r in routes if r]
            if len(new_routes) != len(routes):
                routes = new_routes
                improved = True

        return routes

    routes = try_improve(routes)

    # Final validation and repair
    all_served = set()
    for r in routes:
        all_served.update(c for c in r if c not in station_set)

    missing = set(customers) - all_served
    for m in missing:
        # Try to insert into existing routes
        inserted = False
        for i in range(len(routes)):
            route = routes[i]
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [m] + route[pos:]
                if route_demand(new_route) > capacity:
                    continue
                f, _, _, _ = simulate_route(new_route)
                if f:
                    routes[i] = new_route
                    inserted = True
                    break
            if inserted:
                break
        if not inserted:
            routes.append([m])

    # Final feasibility check: fix any infeasible routes with stations
    for i in range(len(routes)):
        f, _, _, _ = simulate_route(routes[i])
        if not f:
            fixed, _ = best_station_insertion(routes[i])
            if simulate_route(fixed)[0]:
                routes[i] = fixed

    # Remove empty routes
    routes = [r for r in routes if r]

    return routes
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True

    def simulate_route(route):
        """
        Simulate a route (list of node ids - customers and/or stations).
        Returns (feasible, total_dist, end_time, end_battery)
        """
        if not route:
            return True, 0.0, 0.0, battery_cap

        cur_time = 0.0
        cur_battery = battery_cap
        total_dist = 0.0
        cur_pos = 0

        for node in route:
            d = dist[cur_pos][node]
            cur_time += d / speed
            cur_battery -= d
            total_dist += d

            if cur_battery < -1e-9:
                return False, total_dist, cur_time, cur_battery

            if is_station[node]:
                charge_time = (battery_cap - cur_battery) * recharge_rate
                cur_time += charge_time
                cur_battery = battery_cap
                cur_pos = node
            else:
                cur_time = max(cur_time, ready[node])
                if cur_time > due[node] + 1e-9:
                    return False, total_dist, cur_time, cur_battery
                cur_time += service[node]
                cur_pos = node

        # Return to depot
        d = dist[cur_pos][0]
        cur_time += d / speed
        cur_battery -= d
        total_dist += d

        if cur_battery < -1e-9:
            return False, total_dist, cur_time, cur_battery

        if cur_time > horizon + 1e-9:
            return False, total_dist, cur_time, cur_battery

        return True, total_dist, cur_time, cur_battery

    def route_total_dist(route):
        if not route:
            return 0.0
        d = dist[0][route[0]]
        for i in range(len(route) - 1):
            d += dist[route[i]][route[i + 1]]
        d += dist[route[-1]][0]
        return d

    def route_demand(route):
        return sum(demands[c] for c in route if not is_station[c])

    def best_station_insertion(route):
        """
        Given an infeasible route, find the best single station insertion.
        Returns (new_route, station_id) or (route, None).
        """
        feasible, _, _, _ = simulate_route(route)
        if feasible:
            return route, None

        best_new = None
        best_station = None
        best_dist = float('inf')

        for st in stations:
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [st] + route[pos:]
                f, d, _, _ = simulate_route(new_route)
                if f and d < best_dist:
                    best_dist = d
                    best_new = new_route
                    best_station = st

        if best_new is not None:
            return best_new, best_station
        return route, None

    # Construction: Savings heuristic
    def construct_savings():
        """
        Clarke-Wright savings construction with battery and time window constraints.
        """
        routes = [[c] for c in customers]

        savings = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
                savings.append((s, i, j))

        savings.sort(reverse=True)

        cust_to_route = {c: i for i, c in enumerate(customers)}

        for s_val, i, j in savings:
            if time.time() > deadline:
                break
            ci = customers[i]
            cj = customers[j]

            ri = cust_to_route.get(ci)
            rj = cust_to_route.get(cj)
            if ri is None or rj is None or ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            if route_demand(route_i) + route_demand(route_j) > capacity:
                continue

            merged = None
            for orient_i in [route_i, route_i[::-1]]:
                for orient_j in [route_j, route_j[::-1]]:
                    candidate = orient_i + orient_j
                    feasible, _, _, _ = simulate_route(candidate)
                    if feasible:
                        merged = candidate
                        break
                if merged:
                    break

            if merged is None:
                for orient_i in [route_i, route_i[::-1]]:
                    for orient_j in [route_j, route_j[::-1]]:
                        candidate = orient_i + orient_j
                        for st in stations:
                            for pos in range(len(candidate) + 1):
                                test_route = candidate[:pos] + [st] + candidate[pos:]
                                f, _, _, _ = simulate_route(test_route)
                                if f:
                                    merged = test_route
                                    break
                            if merged:
                                break
                        if merged:
                            break
                    if merged:
                        break

            if merged is not None:
                routes[ri] = merged
                routes[rj] = []
                for c in merged:
                    if not is_station[c]:
                        cust_to_route[c] = ri

        routes = [r for r in routes if r]

        all_served = set()
        for r in routes:
            all_served.update(c for c in r if not is_station[c])
        missing = set(customers) - all_served
        for m in missing:
            routes.append([m])

        return routes

    def construct_nearest_neighbor():
        """Nearest neighbor construction with constraints."""
        unvisited = set(customers)
        routes = []

        while unvisited:
            if time.time() > deadline:
                break
            route = []
            cur_pos = 0
            cur_time = 0.0
            cur_battery = battery_cap
            route_dem = 0

            while unvisited:
                best_cust = None
                best_score = float('inf')

                for cust in unvisited:
                    d = dist[cur_pos][cust]
                    new_time = cur_time + d / speed
                    new_battery = cur_battery - d

                    if new_battery < -1e-9:
                        continue
                    if route_dem + demands[cust] > capacity:
                        continue

                    new_time = max(new_time, ready[cust])
                    if new_time > due[cust] + 1e-9:
                        continue

                    return_d = dist[cust][0]
                    if new_battery - return_d < -1e-9:
                        continue
                    if new_time + service[cust] + return_d / speed > horizon + 1e-9:
                        continue

                    if d < best_score:
                        best_score = d
                        best_cust = cust

                if best_cust is None:
                    break

                d = dist[cur_pos][best_cust]
                cur_time += d / speed
                cur_battery -= d
                cur_time = max(cur_time, ready[best_cust])
                cur_time += service[best_cust]
                route_dem += demands[best_cust]
                route.append(best_cust)
                unvisited.remove(best_cust)
                cur_pos = best_cust

            if route:
                feasible, _, _, _ = simulate_route(route)
                if feasible:
                    routes.append(route)
                else:
                    fixed_route, _ = best_station_insertion(route)
                    if simulate_route(fixed_route)[0]:
                        routes.append(fixed_route)
                    else:
                        for c in route:
                            routes.append([c])
                            unvisited.add(c)
            else:
                for cust in list(unvisited):
                    one_route = [cust]
                    feasible, _, _, _ = simulate_route(one_route)
                    if feasible:
                        routes.append(one_route)
                        unvisited.remove(cust)
                        break
                else:
                    for cust in list(unvisited):
                        for st in stations:
                            test = [st, cust]
                            if simulate_route(test)[0]:
                                routes.append(test)
                                unvisited.remove(cust)
                                break
                            test2 = [cust, st]
                            if simulate_route(test2)[0]:
                                routes.append(test2)
                                unvisited.remove(cust)
                                break

        if unvisited:
            for cust in list(unvisited):
                routes.append([cust])

        return routes

    # Try both constructions, pick better
    routes1 = construct_savings()
    dist1 = sum(route_total_dist(r) for r in routes1)

    if time.time() < deadline:
        routes2 = construct_nearest_neighbor()
        dist2 = sum(route_total_dist(r) for r in routes2)
        if dist2 < dist1:
            routes = routes2
        else:
            routes = routes1
    else:
        routes = routes1

    # --- Efficient local search ---

    def two_opt_intra(routes, i):
        """
        Proper 2-opt within route i using edge-delta evaluation.
        Reverses segment [a+1 .. b] if it reduces distance.
        """
        route = routes[i]
        n = len(route)
        if n < 3:
            return False

        best_imp = 0.0
        best_a, best_b = -1, -1

        for a in range(n - 1):
            # Edge before a: from route[a-1] to route[a], or from depot to route[0] if a==0
            if a == 0:
                edge_prev = dist[0][route[0]]
            else:
                edge_prev = dist[route[a - 1]][route[a]]

            # Edge after b: from route[b] to route[b+1], or from route[n-1] to depot if b==n-1
            for b in range(a + 1, n):
                if b == n - 1:
                    edge_next = dist[route[n - 1]][0]
                else:
                    edge_next = dist[route[b]][route[b + 1]]

                # New edges after reversal:
                # route[a] connects to route[b] (was route[a]->route[a+1], route[b]->route[b+1])
                if a == 0:
                    new_edge1 = dist[0][route[b]]
                else:
                    new_edge1 = dist[route[a - 1]][route[b]]

                if b == n - 1:
                    new_edge2 = dist[route[a]][0]
                else:
                    new_edge2 = dist[route[a]][route[b + 1]]

                delta = new_edge1 + new_edge2 - edge_prev - edge_next
                imp = -delta
                if imp > best_imp:
                    best_imp = imp
                    best_a, best_b = a, b

        if best_imp > 1e-6:
            a, b = best_a, best_b
            new_route = route[:a + 1] + route[a + 1:b + 1][::-1] + route[b + 1:]
            f, _, _, _ = simulate_route(new_route)
            if f:
                routes[i] = new_route
                return True
            # If infeasible, skip
            return False
        return False

    def two_opt_inter(routes, i, j):
        """
        Inter-route 2-opt: try reconnecting routes i and j.
        Route i: ...a - b...  Route j: ...c - d...
        Swap to: ...a - c... and ...b - d...
        Returns True if improvement made.
        """
        route_i = routes[i]
        route_j = routes[j]
        ni = len(route_i)
        nj = len(route_j)
        if ni < 1 or nj < 1:
            return False

        best_imp = 0.0
        best_ai, best_bi, best_cj, best_dj = -1, -1, -1, -1

        for ai in range(ni - 1):
            if ai == 0:
                edge_ai = dist[0][route_i[0]]
            else:
                edge_ai = dist[route_i[ai - 1]][route_i[ai]]
            edge_bi = dist[route_i[ai]][route_i[ai + 1]]

            for cj in range(nj - 1):
                if cj == 0:
                    edge_cj = dist[0][route_j[0]]
                else:
                    edge_cj = dist[route_j[cj - 1]][route_j[cj]]
                edge_dj = dist[route_j[cj]][route_j[cj + 1]]

                # New connection: route_i[ai] -> route_j[cj] and route_i[ai+1] -> route_j[cj+1]
                new_edge1 = dist[route_i[ai]][route_j[cj]]
                new_edge2 = dist[route_i[ai + 1]][route_j[cj + 1]]

                delta = new_edge1 + new_edge2 - edge_bi - edge_dj
                imp = -delta
                if imp > best_imp:
                    best_imp = imp
                    best_ai, best_bi, best_cj, best_dj = ai, ai + 1, cj, cj + 1

        if best_imp > 1e-6:
            ai, bi, cj, dj = best_ai, best_bi, best_cj, best_dj
            # Reconnect: route_i becomes route_i[:ai+1] + route_j[dj:]
            # route_j becomes route_j[:cj+1] + route_i[bi:]
            new_i = route_i[:ai + 1] + route_j[dj:]
            new_j = route_j[:cj + 1] + route_i[bi:]

            # Check feasibility of both
            f1, _, _, _ = simulate_route(new_i)
            f2, _, _, _ = simulate_route(new_j)
            if f1 and f2:
                if route_demand(new_i) <= capacity and route_demand(new_j) <= capacity:
                    routes[i] = new_i
                    routes[j] = new_j
                    return True
        return False

    def or_opt_move(routes, i, j, k):
        """
        Try moving segment routes[i][j:k] (length 1-3) to another route or position.
        """
        route = routes[i]
        if k - j > 3 or k > len(route) or j < 0:
            return False, routes

        segment = route[j:k]
        custs_in_seg = [c for c in segment if not is_station[c]]
        if not custs_in_seg:
            return False, routes

        new_route_i = route[:j] + route[k:]
        feasible_i, _, _, _ = simulate_route(new_route_i)
        if not feasible_i:
            return False, routes

        old_total = route_total_dist(route)

        # Try inserting into same route at different position
        for pos in range(len(new_route_i) + 1):
            if pos >= j and pos <= k:
                continue
            new_route_i_test = new_route_i[:pos] + segment + new_route_i[pos:]
            if route_demand(new_route_i_test) > capacity:
                continue
            f, d, _, _ = simulate_route(new_route_i_test)
            if f and d < old_total - 1e-6:
                routes[i] = new_route_i_test
                return True, routes

        # Try inserting into other routes
        for k2 in range(len(routes)):
            if k2 == i:
                continue
            route_k = routes[k2]
            for pos in range(len(route_k) + 1):
                new_route_k = route_k[:pos] + segment + route_k[pos:]
                if route_demand(new_route_k) > capacity:
                    continue
                f, d, _, _ = simulate_route(new_route_k)
                if not f:
                    continue
                new_dist_total = (route_total_dist(new_route_i) if new_route_i else 0) + d
                old_dist_total = (route_total_dist(new_route_i) if new_route_i else 0) + route_total_dist(route_k)
                if new_dist_total < old_dist_total - 1e-6:
                    if not new_route_i:
                        routes[i] = []
                    else:
                        routes[i] = new_route_i
                    routes[k2] = new_route_k
                    return True, routes

        return False, routes

    def try_improve(routes):
        """Iterative local search with intra 2-opt, inter 2-opt, and or-opt."""
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Intra-route 2-opt
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                if two_opt_intra(routes, i):
                    improved = True

            # Inter-route 2-opt
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(routes)):
                    if time.time() > deadline:
                        break
                    if two_opt_inter(routes, i, j):
                        improved = True
                        break
                if improved:
                    break

            # Or-opt moves
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                route = routes[i]
                for j in range(len(route)):
                    for length in [1, 2, 3]:
                        k = j + length
                        if k > len(route):
                            continue
                        imp, routes = or_opt_move(routes, i, j, k)
                        if imp:
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            # Station insertion for infeasible routes
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                f, _, _, _ = simulate_route(routes[i])
                if not f:
                    fixed, _ = best_station_insertion(routes[i])
                    if simulate_route(fixed)[0]:
                        routes[i] = fixed
                        improved = True

            # Remove empty routes
            new_routes = [r for r in routes if r]
            if len(new_routes) != len(routes):
                routes = new_routes
                improved = True

        return routes

    routes = try_improve(routes)

    # Final validation and repair
    all_served = set()
    for r in routes:
        all_served.update(c for c in r if not is_station[c])

    missing = set(customers) - all_served
    for m in missing:
        inserted = False
        for i in range(len(routes)):
            route = routes[i]
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [m] + route[pos:]
                if route_demand(new_route) > capacity:
                    continue
                f, _, _, _ = simulate_route(new_route)
                if f:
                    routes[i] = new_route
                    inserted = True
                    break
            if inserted:
                break
        if not inserted:
            routes.append([m])

    # Final feasibility check
    for i in range(len(routes)):
        f, _, _, _ = simulate_route(routes[i])
        if not f:
            fixed, _ = best_station_insertion(routes[i])
            if simulate_route(fixed)[0]:
                routes[i] = fixed

    routes = [r for r in routes if r]

    return routes
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

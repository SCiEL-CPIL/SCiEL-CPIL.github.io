"""Rewritten program for EV routing with time windows."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

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
    # Precompute distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def simulate_route(route):
        """
        Simulates a route. Returns (feasible, total_dist, end_time, end_battery).
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
                charge_time = (battery_cap - cur_battery) * recharge_rate
                cur_time += charge_time
                cur_battery = battery_cap
            else:
                # Customer
                if cur_time < ready[node]:
                    cur_time = ready[node]
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

    def route_demand(route):
        return sum(demands[c] for c in route if c not in station_set)

    def construct_savings():
        """
        Clarke-Wright savings construction with battery/time constraints.
        """
        routes = [[c] for c in customers]
        route_idx = {c: i for i, c in enumerate(customers)}
        
        # Calculate savings
        savings = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
                savings.append((s, i, j))
        
        savings.sort(reverse=True)

        for s_val, i, j in savings:
            if time.time() > deadline - 10:
                break
            
            ci = customers[i]
            cj = customers[j]
            
            ri = route_idx.get(ci)
            rj = route_idx.get(cj)
            if ri is None or rj is None or ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            if route_demand(route_i) + route_demand(route_j) > capacity:
                continue

            merged = None
            # Try 4 orientations
            for orient_i in [route_i, route_i[::-1]]:
                for orient_j in [route_j, route_j[::-1]]:
                    candidate = orient_i + orient_j
                    f, d, _, _ = simulate_route(candidate)
                    if f:
                        merged = candidate
                        break
                if merged:
                    break

            if merged is None:
                # Try inserting a station to make it feasible
                for orient_i in [route_i, route_i[::-1]]:
                    for orient_j in [route_j, route_j[::-1]]:
                        candidate = orient_i + orient_j
                        for st in stations:
                            # Try inserting station at various positions
                            # Optimization: Only try positions near the "bottleneck" or simple heuristic
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
                    if c not in station_set:
                        route_idx[c] = ri

        routes = [r for r in routes if r]
        
        # Ensure all customers present
        all_served = set()
        for r in routes:
            all_served.update(c for c in r if c not in station_set)
        missing = set(customers) - all_served
        for m in missing:
            # Try to insert missing customer
            inserted = False
            for i in range(len(routes)):
                for pos in range(len(routes[i]) + 1):
                    new_route = routes[i][:pos] + [m] + routes[i][pos:]
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
                
        return routes

    def two_opt_intra(route):
        """
        Returns improved route if 2-opt finds improvement, else None.
        """
        if len(route) < 3:
            return None
        
        best_route = route
        best_dist = simulate_route(route)[1]
        
        n = len(route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i+1 to j
                new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                f, d, _, _ = simulate_route(new_route)
                if f and d < best_dist - 1e-6:
                    best_dist = d
                    best_route = new_route
        return best_route

    def or_opt_move(routes, i, j, k):
        """
        Move segment [j:k] from route i to another position or route.
        Returns (improved, new_routes)
        """
        route_i = routes[i]
        if k > len(route_i) or j < 0 or k - j > 3:
            return False, routes
        
        segment = route_i[j:k]
        custs_in_seg = [c for c in segment if c not in station_set]
        if not custs_in_seg:
            return False, routes

        # Remove segment from route i
        new_route_i = route_i[:j] + route_i[k:]
        f_i, d_i, _, _ = simulate_route(new_route_i)
        if not f_i:
            return False, routes
        
        old_total_dist = simulate_route(route_i)[1]
        
        # Try inserting into same route i
        for pos in range(len(new_route_i) + 1):
            if pos >= j and pos <= k:
                continue
            new_route_i_test = new_route_i[:pos] + segment + new_route_i[pos:]
            if route_demand(new_route_i_test) > capacity:
                continue
            f, d, _, _ = simulate_route(new_route_i_test)
            if f and d < old_total_dist - 1e-6:
                new_routes = routes[:]
                new_routes[i] = new_route_i_test
                return True, new_routes

        # Try inserting into other routes
        for k2 in range(len(routes)):
            if k2 == i:
                continue
            route_k = routes[k2]
            old_dist_k = simulate_route(route_k)[1]
            
            for pos in range(len(route_k) + 1):
                new_route_k = route_k[:pos] + segment + route_k[pos:]
                if route_demand(new_route_k) > capacity:
                    continue
                f, d, _, _ = simulate_route(new_route_k)
                if f:
                    new_total = d_i + d
                    old_total = old_total_dist + old_dist_k
                    if new_total < old_total - 1e-6:
                        new_routes = routes[:]
                        if new_route_i:
                            new_routes[i] = new_route_i
                        else:
                            new_routes[i] = []
                        new_routes[k2] = new_route_k
                        return True, new_routes
        
        return False, routes

    def try_remove_station(route):
        """
        Try to remove any station from the route if feasible.
        Returns improved route or None.
        """
        best_route = route
        best_dist = simulate_route(route)[1]
        
        for i in range(len(route)):
            if route[i] in station_set:
                new_route = route[:i] + route[i+1:]
                f, d, _, _ = simulate_route(new_route)
                if f and d < best_dist - 1e-6:
                    best_dist = d
                    best_route = new_route
        return best_route

    def local_search(routes):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. 2-Opt Intra
            for i in range(len(routes)):
                if time.time() > deadline - 5:
                    break
                new_route = two_opt_intra(routes[i])
                if new_route and new_route != routes[i]:
                    routes[i] = new_route
                    improved = True

            # 2. Or-Opt Inter/Intra
            for i in range(len(routes)):
                if time.time() > deadline - 5:
                    break
                route = routes[i]
                for j in range(len(route)):
                    for length in [1, 2, 3]:
                        k = j + length
                        if k > len(route):
                            continue
                        imp, new_routes = or_opt_move(routes, i, j, k)
                        if imp:
                            routes = new_routes
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            # 3. Station Removal
            for i in range(len(routes)):
                if time.time() > deadline - 5:
                    break
                new_route = try_remove_station(routes[i])
                if new_route and new_route != routes[i]:
                    routes[i] = new_route
                    improved = True

            # 4. Clean up empty routes
            new_routes = [r for r in routes if r]
            if len(new_routes) != len(routes):
                routes = new_routes
                improved = True

        return routes

    # Initial Construction
    routes = construct_savings()
    
    # Local Search
    routes = local_search(routes)
    
    # Final Repair: Ensure all customers are served and feasible
    all_served = set()
    for r in routes:
        all_served.update(c for c in r if c not in station_set)
    
    missing = set(customers) - all_served
    for m in missing:
        inserted = False
        for i in range(len(routes)):
            for pos in range(len(routes[i]) + 1):
                new_route = routes[i][:pos] + [m] + routes[i][pos:]
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
            # Last resort: add as separate route (might be infeasible, but better than missing)
            # Try to add station if needed
            for st in stations:
                if simulate_route([st, m])[0]:
                    routes.append([st, m])
                    inserted = True
                    break
                if simulate_route([m, st])[0]:
                    routes.append([m, st])
                    inserted = True
                    break
            if not inserted:
                routes.append([m])

    # Final Feasibility Check & Repair
    for i in range(len(routes)):
        f, _, _, _ = simulate_route(routes[i])
        if not f:
            # Try to fix by inserting stations
            fixed = None
            for st in stations:
                for pos in range(len(routes[i]) + 1):
                    test = routes[i][:pos] + [st] + routes[i][pos:]
                    if simulate_route(test)[0]:
                        fixed = test
                        break
                if fixed:
                    break
            if fixed:
                routes[i] = fixed
            else:
                # Split into singletons if still infeasible
                for c in routes[i]:
                    if c in station_set: continue
                    routes.append([c])
                routes[i] = []

    routes = [r for r in routes if r]
    return routes

# EVOLVE-BLOCK-END

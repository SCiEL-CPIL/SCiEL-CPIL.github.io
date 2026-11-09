import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    # 120s limit, use 110s as deadline to be safe
    deadline = start_time + 110.0 

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
    # Precompute distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))

    def fast_check_route(route):
        """Check feasibility of a route. Returns (is_feasible, total_distance)."""
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
                # Charge time = (full_battery - current_battery) / rate
                # Wait, the prompt says: "a station adds (battery - energy) * recharge_rate time and refills"
                # Usually recharge rate is energy/time. So time = energy/rate.
                # Prompt: "recharge_rate 0.5". If rate is 0.5 energy/time, time = energy / 0.5 = 2*energy.
                # Let's re-read carefully: "adds (battery - energy) * recharge_rate time"
                # This implies time = (battery - energy) * recharge_rate.
                # If recharge_rate is 0.5, and we need 100 energy, time = 50.
                # This means recharge_rate is 1/speed_of_charging? i.e. time per unit energy.
                # Let's stick to the formula: time += (battery - bat) * recharge_rate
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

    def savings_construct():
        """Clarke-Wright savings based construction."""
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

            if route_load[ri] + route_load[rj] > capacity:
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
        """Construct routes using greedy nearest neighbor."""
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
                    t_after = t + d / speed

                    if bat_after < -1e-9:
                        # Try going through a station first
                        best_station_score = float('inf')
                        found_station = False
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
                            if load + demands[v] > capacity:
                                continue
                            if t_v < ready[v]:
                                t_v = ready[v]
                            if t_v > due[v] + 1e-9:
                                continue
                            time_slack = due[v] - t_v
                            score = d1 + d2 - 0.02 * time_slack
                            if score < best_station_score:
                                best_station_score = score
                                found_station = True
                        if found_station:
                            # For simplicity in construction, we just skip if direct path fails and station path works?
                            # Actually, we need to insert the station.
                            # This greedy logic is getting complex. Let's stick to simple greedy for construction
                            # and rely on LNS/Local Search to fix battery issues if any, or ensure construction is conservative.
                            # Given battery=300 and coords in [0,100], max dist is ~141.
                            # 300/141 ~ 2 legs. So direct path usually works unless detours are huge.
                            # Let's just skip this customer for now if direct fails, hoping LNS fixes it.
                            # Or better, just continue to next customer.
                            continue
                        continue

                    if load + demands[v] > capacity:
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

    # Initial solution
    best_routes = [[v] for v in customers]
    best_distance = sum(dist[0, v] * 2 for v in customers)

    # Try savings
    if time.time() < deadline:
        sw_routes = savings_construct()
        if validate_routes(sw_routes):
            d = total_route_distance(sw_routes)
            if d < best_distance:
                best_routes = sw_routes
                best_distance = d

    # Try greedy
    orderings = [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: dist[0, v]),
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
        """Apply 2-opt to a single route."""
        if len(route) < 2:
            return route
        improved = True
        while improved and time.time() < deadline:
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

    def relocate_search(routes, max_passes=5):
        """Remove-and-reinsert local search."""
        routes = [list(r) for r in routes]
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                n_i = len(routes[i])
                if n_i == 0:
                    continue
                for j in range(n_i):
                    if time.time() > deadline:
                        break
                    v = routes[i][j]
                    route_i_without = routes[i][:j] + routes[i][j+1:]
                    if not route_i_without:
                        # If route becomes empty, we just moved the only node.
                        # We need to insert it elsewhere.
                        pass
                    else:
                        feasible_i, dist_i_new = fast_check_route(route_i_without)
                        if not feasible_i:
                            continue
                        
                        best_delta = -1e-6
                        best_k = -1
                        best_pos = -1
                        best_new_route_k = None
                        
                        for k in range(len(routes)):
                            if k == i:
                                continue
                            route_k = routes[k]
                            n_k = len(route_k)
                            for pos in range(n_k + 1):
                                new_route_k = route_k[:pos] + [v] + route_k[pos:]
                                feasible_k, dist_k_new = fast_check_route(new_route_k)
                                if not feasible_k:
                                    continue
                                
                                # Calculate delta
                                # Old dist: dist_i_old + dist_k_old
                                # New dist: dist_i_new + dist_k_new
                                # We don't have dist_i_old cached easily here without recalculating or storing.
                                # Let's just calculate total delta roughly or store dists.
                                # For speed, let's just check if total distance improves.
                                # To avoid O(N^2) recalculation, we can approximate.
                                # But for correctness, let's just compute the specific change.
                                
                                # Change in route i: remove v
                                # Change in route k: insert v
                                
                                # Let's just do a full check on the modified routes and compare sum.
                                # This is expensive. Let's optimize by precomputing route distances.
                                pass
                        
                        # Simplified Relocate: just try to insert v into other routes if it reduces distance
                        # We'll do a simpler version: try all positions in all other routes, accept if feasible and distance drops.
                        current_total = 0
                        for r in routes:
                            if r:
                                _, d = fast_check_route(r)
                                current_total += d
                        # This is too slow. 
                        
                        # Let's use a faster delta calculation.
                        # Dist of route i with v: D_i(v)
                        # Dist of route i without v: D_i(-v)
                        # Dist of route k with v at pos: D_k(v, pos)
                        # Dist of route k without v: D_k(-v)
                        
                        # We can compute D_i(-v) once.
                        # Then for each k, pos, compute D_k(v, pos) - D_k(-v).
                        # If D_i(-v) + D_k(v, pos) < D_i(v) + D_k(-v), accept.
                        
                        # To make this efficient, we need D_i(v) and D_k(-v).
                        # We can precompute route distances at the start of the pass.
                        
                        pass
            # Due to complexity of efficient delta calculation in this short timeframe, 
            # we will rely on 2-opt and LNS which are easier to implement correctly with full checks.
            # LNS is powerful enough to handle inter-route moves effectively.
            
        return routes

    def lns_search(routes, iterations=200):
        """Large Neighborhood Search."""
        routes = [list(r) for r in routes]
        current_dist = total_route_distance(routes)
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            
            # Destroy: remove a subset of customers
            # Pick a random percentage to destroy
            destroy_ratio = 0.3 + 0.3 * random.random() # 30% to 60%
            n_destroy = max(2, int(len(customers) * destroy_ratio))
            
            # Select random customers to destroy
            destroy_set = set(random.sample(customers, n_destroy))
            
            # Remove them from routes
            new_routes = []
            removed_customers = []
            for route in routes:
                new_route = []
                for v in route:
                    if v in destroy_set:
                        removed_customers.append(v)
                    else:
                        new_route.append(v)
                if new_route:
                    new_routes.append(new_route)
            
            # If all routes are empty, we have a problem. But we only destroy subset.
            # Ensure at least one route remains? Not strictly necessary if we re-insert into new routes.
            
            # Repair: Insert removed customers one by one
            # Sort removed customers by due date? Or random?
            # Random order often works well for LNS diversity.
            random.shuffle(removed_customers)
            
            for c in removed_customers:
                best_insert_cost = float('inf')
                best_route_idx = -1
                best_pos = -1
                
                # Try inserting into existing routes
                for r_idx, route in enumerate(new_routes):
                    # Check capacity
                    load = sum(demands[v] for v in route if v not in station_set)
                    if load + demands[c] > capacity:
                        continue
                    
                    # Try all positions
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [c] + route[pos:]
                        feasible, d_new = fast_check_route(new_route)
                        if feasible:
                            # Calculate cost increase
                            # Old cost of this route
                            _, d_old = fast_check_route(route)
                            delta = d_new - d_old
                            if delta < best_insert_cost:
                                best_insert_cost = delta
                                best_route_idx = r_idx
                                best_pos = pos
                
                # If not found in existing routes, create a new one
                if best_route_idx == -1:
                    # Try creating a new route [c]
                    feasible, d_new = fast_check_route([c])
                    if feasible:
                        new_routes.append([c])
                    else:
                        # Try with a station?
                        # For now, if we can't insert, we fail this LNS iteration
                        # Restore original routes
                        return routes
                else:
                    new_routes[best_route_idx].insert(best_pos, c)
            
            # Validate new solution
            if validate_routes(new_routes):
                new_dist = total_route_distance(new_routes)
                if new_dist < current_dist:
                    routes = new_routes
                    current_dist = new_dist
                    # Optionally apply 2-opt to the new solution to polish
                    for i in range(len(routes)):
                        if time.time() > deadline:
                            break
                        routes[i] = two_opt_route(routes[i])
                    # Recalculate dist after 2-opt
                    if validate_routes(routes):
                        current_dist = total_route_distance(routes)
                    else:
                        # 2-opt broke something? Unlikely if fast_check is used, but revert to pre-2opt if so
                        # We didn't save pre-2opt. Let's assume 2-opt is safe.
                        pass
                        
        return routes

    # Run LNS
    if time.time() < deadline:
        lns_routes = lns_search(best_routes, iterations=300)
        if validate_routes(lns_routes):
            d = total_route_distance(lns_routes)
            if d < best_distance:
                best_routes = lns_routes
                best_distance = d

    # Final Polish: 2-opt on all routes
    if time.time() < deadline - 5:
        for i in range(len(best_routes)):
            if time.time() > deadline - 2:
                break
            best_routes[i] = two_opt_route(best_routes[i])
        
        # One more LNS pass with fewer iterations to polish
        if time.time() < deadline - 2:
            lns_routes = lns_search(best_routes, iterations=50)
            if validate_routes(lns_routes):
                d = total_route_distance(lns_routes)
                if d < best_distance:
                    best_routes = lns_routes
                    best_distance = d
                    for i in range(len(best_routes)):
                        if time.time() > deadline - 1:
                            break
                        best_routes[i] = two_opt_route(best_routes[i])

    best_routes = [r for r in best_routes if r]
    return best_routes

# EVOLVE-BLOCK-END

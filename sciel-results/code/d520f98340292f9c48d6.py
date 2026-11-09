"""EV routing with time windows – LNS with Clarke-Wright savings init."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import heapq

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    coords = np.array(instance["coords"])
    demands = np.array(instance["demands"])
    ready = np.array(instance["ready"])
    due = np.array(instance["due"])
    service = np.array(instance["service"])
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            dist[i, j] = d
            dist[j, i] = d

    customers = list(range(1, n_customers + 1))

    def simulate_route(route):
        """Simulate a route. Returns (feasible, total_distance, end_time, end_battery)."""
        if not route:
            return True, 0.0, 0.0, battery
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0, 0.0, 0.0

        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0

        for v in route:
            d = dist[prev, v]
            bat -= d
            t += d / speed
            total_dist += d
            if bat < 0:
                return False, 0.0, 0.0, 0.0
            # Time window check
            if t < ready[v]:
                t = ready[v]
            if t > due[v]:
                return False, 0.0, 0.0, 0.0
            t += service[v]
            prev = v

        # Return to depot
        d = dist[prev, 0]
        bat -= d
        t += d / speed
        total_dist += d
        if bat < 0:
            return False, 0.0, 0.0, 0.0
        if t > horizon:
            return False, 0.0, 0.0, 0.0

        return True, total_dist, t, bat

    def total_distance(routes):
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += dist[0, route[0]]
            for i in range(len(route) - 1):
                total += dist[route[i], route[i + 1]]
            total += dist[route[-1], 0]
        return total

    def clark_wright_savings_init():
        """Clarke-Wright savings heuristic to build initial multi-route solution."""
        # Start with each customer as its own route
        routes = [[v] for v in customers]
        # Compute savings: savings(i,j) = d(0,i) + d(0,j) - d(i,j)
        # We want to merge routes that save the most distance
        # Route state: list of routes, each route is a list of customers
        # For each pair of routes, check if merging is beneficial and feasible

        # Build index: customer -> route index
        cust_route = {v: i for i, r in enumerate(routes) for v in r}

        # Compute all possible savings
        savings_list = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                s = dist[0, ci] + dist[0, cj] - dist[ci, cj]
                if s > 0:
                    savings_list.append((s, ci, cj))

        # Sort by savings descending
        savings_list.sort(key=lambda x: -x[0])

        for s, ci, cj in savings_list:
            if time.time() > deadline:
                break
            ri = cust_route.get(ci)
            rj = cust_route.get(cj)
            if ri is None or rj is None or ri == rj:
                continue

            # Try to merge route rj into route ri
            # Check if ci is at an end of route ri and cj is at an end of route rj
            # We need to find valid merge points
            route_i = routes[ri]
            route_j = routes[rj]

            # Try all 4 combinations of ends
            merged = None
            for reverse_i in [False, True]:
                for reverse_j in [False, True]:
                    if reverse_i:
                        new_i = route_i[::-1]
                    else:
                        new_i = route_i[:]
                    if reverse_j:
                        new_j = route_j[::-1]
                    else:
                        new_j = route_j[:]

                    # Check if ci is at an end of new_i and cj is at an end of new_j
                    if new_i[0] == ci or new_i[-1] == ci:
                        if new_j[0] == cj or new_j[-1] == cj:
                            # Try both orientations of the merge
                            # Option 1: new_i + new_j
                            candidate = new_i + new_j
                            feasible, _, _, _ = simulate_route(candidate)
                            if feasible:
                                merged = candidate
                                break
                            # Option 2: new_j + new_i
                            candidate = new_j + new_i
                            feasible, _, _, _ = simulate_route(candidate)
                            if feasible:
                                merged = candidate
                                break
                    if merged:
                        break
                if merged:
                    break

            if merged:
                # Update routes
                routes[ri] = merged
                routes[rj] = []
                for v in merged:
                    cust_route[v] = ri
                # Remove empty routes
                routes = [r for idx, r in enumerate(routes) if r]
                # Rebuild cust_route
                cust_route = {v: i for i, r in enumerate(routes) for v in r}

        return [r for r in routes if r]

    def lns_destroy_reconstruct(routes, n_destroy):
        """LNS: destroy random customers, reconstruct with DP-based insertion."""
        # Flatten all customers
        all_custs = []
        for r in routes:
            all_custs.extend(r)

        if not all_custs:
            return routes

        # Select customers to destroy
        if n_destroy >= len(all_custs):
            destroy_set = set(all_custs)
        else:
            destroy_set = set(random.sample(all_custs, n_destroy))

        # Remove destroyed customers from routes
        remaining_routes = []
        for r in routes:
            new_r = [v for v in r if v not in destroy_set]
            if new_r:
                remaining_routes.append(new_r)

        # Reconstruct: insert destroyed customers optimally
        destroyed = list(destroy_set)
        # Sort by due time for better insertion
        destroyed.sort(key=lambda v: due[v])

        for v in destroyed:
            if time.time() > deadline:
                break
            # Find best insertion position across all routes
            best_route_idx = None
            best_pos = None
            best_cost = float('inf')

            # Also consider creating a new route
            # Try inserting into existing routes
            for ri, route in enumerate(remaining_routes):
                for pos in range(len(route) + 1):
                    # Check capacity
                    route_load = sum(demands[x] for x in route)
                    if route_load + demands[v] > capacity:
                        continue

                    new_route = route[:pos] + [v] + route[pos:]
                    feasible, d, _, _ = simulate_route(new_route)
                    if feasible:
                        # Cost: increase in total distance
                        if pos == 0:
                            old = dist[0, route[0]] if route else 0
                            new = dist[0, v] + dist[v, route[0]] if route else dist[0, v] + dist[v, 0]
                            if not route:
                                old = 0
                                new = dist[0, v] + dist[v, 0]
                            else:
                                old = dist[0, route[0]]
                                new = dist[0, v] + dist[v, route[0]]
                        elif pos == len(route):
                            old = dist[route[-1], 0]
                            new = dist[route[-1], v] + dist[v, 0]
                        else:
                            old = dist[route[pos-1], route[pos]]
                            new = dist[route[pos-1], v] + dist[v, route[pos]]

                        cost = new - old
                        if cost < best_cost:
                            best_cost = cost
                            best_route_idx = ri
                            best_pos = pos

            # Option: create new route
            new_route = [v]
            feasible, d, _, _ = simulate_route(new_route)
            if feasible:
                cost = d
                if cost < best_cost:
                    best_cost = cost
                    best_route_idx = len(remaining_routes)
                    best_pos = 0

            if best_route_idx is not None:
                if best_route_idx == len(remaining_routes):
                    remaining_routes.append([v])
                else:
                    remaining_routes[best_route_idx].insert(best_pos, v)
            else:
                # Can't insert anywhere, add as new route (shouldn't happen)
                remaining_routes.append([v])

        return remaining_routes

    def local_search(routes):
        """2-opt and Or-opt local search within and between routes."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_dist = total_distance(routes)

            # 2-opt within each route
            for i in range(len(routes)):
                route = routes[i]
                if len(route) < 2:
                    continue
                for a in range(len(route)):
                    for b in range(a + 1, len(route)):
                        if time.time() > deadline:
                            break
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        feasible, d, _, _ = simulate_route(new_route)
                        if feasible:
                            old_boundary = 0.0
                            if a > 0:
                                old_boundary += dist[route[a-1], route[a]]
                            else:
                                old_boundary += dist[0, route[a]]
                            if b < len(route) - 1:
                                old_boundary += dist[route[b], route[b+1]]
                            else:
                                old_boundary += dist[route[b], 0]

                            new_boundary = 0.0
                            if a > 0:
                                new_boundary += dist[route[a-1], route[b]]
                            else:
                                new_boundary += dist[0, route[b]]
                            if b < len(route) - 1:
                                new_boundary += dist[route[a], route[b+1]]
                            else:
                                new_boundary += dist[route[a], 0]

                            if new_boundary < old_boundary - 1e-6:
                                routes[i] = new_route
                                improved = True
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

            # Or-opt: move 1-3 consecutive customers to another position
            for i in range(len(routes)):
                route = routes[i]
                n = len(route)
                if n < 2:
                    continue
                for seg_len in [1, 2, 3]:
                    if n < seg_len + 1:
                        continue
                    for a in range(n - seg_len + 1):
                        if time.time() > deadline:
                            break
                        segment = route[a:a+seg_len]
                        # Remove segment
                        new_route = route[:a] + route[a+seg_len:]
                        # Try inserting at each position
                        for pos in range(len(new_route) + 1):
                            if pos <= a or pos >= a + seg_len:
                                continue
                            candidate = new_route[:pos] + segment + new_route[pos:]
                            feasible, d, _, _ = simulate_route(candidate)
                            if feasible:
                                # Compute distance change
                                old_d = 0.0
                                if a > 0:
                                    old_d += dist[route[a-1], route[a]]
                                else:
                                    old_d += dist[0, route[a]]
                                if a + seg_len < n:
                                    old_d += dist[route[a+seg_len-1], route[a+seg_len]]
                                else:
                                    old_d += dist[route[a+seg_len-1], 0]

                                new_d = 0.0
                                if pos > 0:
                                    new_d += dist[candidate[pos-1], candidate[pos]]
                                else:
                                    new_d += dist[0, candidate[0]]
                                if pos + seg_len < len(candidate):
                                    new_d += dist[candidate[pos+seg_len-1], candidate[pos+seg_len]]
                                else:
                                    new_d += dist[candidate[pos+seg_len-1], 0]

                                if new_d < old_d - 1e-6:
                                    routes[i] = candidate
                                    improved = True
                                    break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

        return routes

    # Step 1: Clarke-Wright savings initialization
    best_routes = clark_wright_savings_init()
    best_dist = total_distance(best_routes)

    # Validate initial solution
    all_custs = []
    valid = True
    for r in best_routes:
        feasible, _, _, _ = simulate_route(r)
        if not feasible:
            valid = False
            break
        all_custs.extend(r)

    if not valid or set(all_custs) != set(customers) or len(all_custs) != len(set(all_custs)):
        # Fallback: one customer per route
        best_routes = [[v] for v in customers]
        best_dist = sum(dist[0, v] * 2 for v in customers)

    # Step 2: Local search
    best_routes = local_search(best_routes)
    best_dist = total_distance(best_routes)

    # Step 3: LNS loop
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline - 5.0:
        # Vary destruction size
        n_destroy = rng.randint(3, 8)
        candidate = lns_destroy_reconstruct(best_routes, n_destroy)

        # Validate candidate
        all_custs = []
        valid = True
        for r in candidate:
            feasible, _, _, _ = simulate_route(r)
            if not feasible:
                valid = False
                break
            all_custs.extend(r)

        if valid and set(all_custs) == set(customers) and len(all_custs) == len(set(all_custs)):
            cand_dist = total_distance(candidate)
            if cand_dist < best_dist - 1e-6:
                best_routes = candidate
                best_dist = cand_dist
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # Occasional local search
        if no_improve_count % 10 == 0 and time.time() < deadline - 10.0:
            best_routes = local_search(best_routes)
            best_dist = total_distance(best_routes)

        if no_improve_count >= max_no_improve:
            # Try larger destruction
            n_destroy = rng.randint(10, 15)
            candidate = lns_destroy_reconstruct(best_routes, n_destroy)
            all_custs = []
            valid = True
            for r in candidate:
                feasible, _, _, _ = simulate_route(r)
                if not feasible:
                    valid = False
                    break
                all_custs.extend(r)
            if valid and set(all_custs) == set(customers) and len(all_custs) == len(set(all_custs)):
                cand_dist = total_distance(candidate)
                if cand_dist < best_dist - 1e-6:
                    best_routes = candidate
                    best_dist = cand_dist
                    no_improve_count = 0
                else:
                    no_improve_count = 0  # Reset to try again
            else:
                no_improve_count = 0

    # Final local search
    if time.time() < deadline - 2.0:
        best_routes = local_search(best_routes)
        best_dist = total_distance(best_routes)

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    return best_routes
# EVOLVE-BLOCK-END

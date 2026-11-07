"""EV routing with time windows and charging - combined approach with LNS."""
import numpy as np
import math
import time
import random
import heapq

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n_customers = instance["n_customers"]
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute pairwise distances
    n_nodes = len(coords)
    coords_arr = np.array(coords, dtype=float)
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dx = coords_arr[i][0] - coords_arr[j][0]
            dy = coords_arr[i][1] - coords_arr[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i][j] = d
            dist[j][i] = d

    station_set = set(stations)

    def simulate_route(route):
        """Simulate a route, return (total_distance, valid)."""
        if not route:
            return 0.0, True
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        total_dist = 0.0
        for node in route:
            d = dist[prev][node]
            total_dist += d
            if bat < d - 1e-9:
                return float('inf'), False
            t += d / speed
            bat -= d
            load += demands[node]
            if load > capacity:
                return float('inf'), False
            if 1 <= node <= n_customers:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return float('inf'), False
                t += service[node]
            elif node in station_set:
                deficit = battery_cap - bat
                if deficit > 0:
                    t += deficit * recharge_rate
                bat = battery_cap
            prev = node
        d_ret = dist[prev][0]
        total_dist += d_ret
        if bat < d_ret - 1e-9:
            return float('inf'), False
        t += d_ret / speed
        if t > horizon + 1e-9:
            return float('inf'), False
        return total_dist, True

    def build_route_with_charging(cust_seq):
        """
        Given a sequence of customer IDs, insert charging stops as needed.
        Returns (route_list, total_distance) or (None, inf) if infeasible.
        """
        if not cust_seq:
            return [], 0.0

        route = []
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        total_dist = 0.0

        for cust in cust_seq:
            d_direct = dist[prev][cust]
            if bat >= d_direct - 1e-9:
                t_arr = t + d_direct / speed
                if t_arr < ready[cust]:
                    t_arr = ready[cust]
                if t_arr > due[cust] + 1e-9:
                    return None, float('inf')
                t = t_arr + service[cust]
                bat -= d_direct
                load += demands[cust]
                if load > capacity:
                    return None, float('inf')
                total_dist += d_direct
                route.append(cust)
                prev = cust
            else:
                best_st = None
                best_cost = float('inf')
                best_t_cust = 0.0
                best_bat_after = 0.0

                for st in stations:
                    d_to_st = dist[prev][st]
                    if bat < d_to_st - 1e-9:
                        continue
                    t_st = t + d_to_st / speed
                    bat_at_st = bat - d_to_st
                    rt = max(0.0, (battery_cap - bat_at_st) * recharge_rate)
                    t_st += rt
                    bat_st = battery_cap
                    d_st_cust = dist[st][cust]
                    if bat_st < d_st_cust - 1e-9:
                        continue
                    t_cust = t_st + d_st_cust / speed
                    if t_cust < ready[cust]:
                        t_cust = ready[cust]
                    if t_cust > due[cust] + 1e-9:
                        continue
                    extra_dist = d_to_st + d_st_cust - d_direct
                    t_after = t_cust + service[cust]
                    bat_after = bat_st - d_st_cust
                    load_new = load + demands[cust]
                    if load_new > capacity:
                        continue
                    if extra_dist < best_cost - 1e-9:
                        best_cost = extra_dist
                        best_st = st
                        best_t_cust = t_after
                        best_bat_after = bat_after

                if best_st is None:
                    return None, float('inf')

                route.append(best_st)
                route.append(cust)
                t = best_t_cust
                bat = best_bat_after
                load += demands[cust]
                total_dist += best_cost
                prev = cust

        d_ret = dist[prev][0]
        if bat >= d_ret - 1e-9:
            t_ret = t + d_ret / speed
            if t_ret > horizon + 1e-9:
                return None, float('inf')
            total_dist += d_ret
        else:
            found = False
            for st in stations:
                d_to_st = dist[prev][st]
                if bat < d_to_st - 1e-9:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                rt = max(0.0, (battery_cap - bat_at_st) * recharge_rate)
                t_st += rt
                bat_st = battery_cap
                d_st_dep = dist[st][0]
                if bat_st < d_st_dep - 1e-9:
                    continue
                t_dep = t_st + d_st_dep / speed
                if t_dep > horizon + 1e-9:
                    continue
                total_dist += d_to_st + d_st_dep
                route.append(st)
                found = True
                break
            if not found:
                return None, float('inf')

        return route, total_dist

    def route_distance(route):
        if not route:
            return 0.0
        total = 0.0
        prev = 0
        for node in route:
            total += dist[prev][node]
            prev = node
        total += dist[prev][0]
        return total

    def total_distance(routes):
        return sum(simulate_route(r)[0] for r in routes)

    # Clarke-Wright Savings construction
    def savings_construct():
        """Construct initial solution using Clarke-Wright savings heuristic."""
        # Start with individual routes for each customer
        routes = []
        for c in range(1, n_customers + 1):
            r, d = build_route_with_charging([c])
            if r is not None:
                routes.append(r)
            else:
                routes.append([c])

        # Compute savings for merging route i and route j
        # savings(i, j) = dist(depot, i) + dist(depot, j) - dist(i, j)
        # where i is last customer in route_i, j is first customer in route_j

        def get_first_last(route):
            """Get first and last customer in a route."""
            first = None
            last = None
            for node in route:
                if 1 <= node <= n_customers:
                    if first is None:
                        first = node
                    last = node
            return first, last

        # Build savings list
        savings_list = []
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                fi, li = get_first_last(routes[i])
                fj, lj = get_first_last(routes[j])
                if fi is None or fj is None:
                    continue

                # Try all 4 combinations of endpoints
                # Merge i->j: last of i, first of j
                s1 = dist[0][li] + dist[0][fj] - dist[li][fj]
                # Merge j->i: last of j, first of i
                s2 = dist[0][lj] + dist[0][fi] - dist[lj][fi]
                # Reverse i then merge: first of i, first of j
                s3 = dist[0][fi] + dist[0][fj] - dist[fi][fj]
                # Reverse j then merge: last of i, last of j
                s4 = dist[0][li] + dist[0][lj] - dist[li][lj]

                best_s = max(s1, s2, s3, s4)
                if best_s > 0:
                    savings_list.append((best_s, i, j))

        savings_list.sort(key=lambda x: -x[0])

        # Apply savings in order
        route_of_customer = {}
        for i, r in enumerate(routes):
            for node in r:
                if 1 <= node <= n_customers:
                    route_of_customer[node] = i

        for s, i, j in savings_list:
            if time.time() > deadline - 30.0:
                break
            if i >= len(routes) or j >= len(routes):
                continue
            if i == j:
                continue

            ri = routes[i]
            rj = routes[j]

            fi, li = get_first_last(ri)
            fj, lj = get_first_last(rj)

            if fi is None or fj is None:
                continue

            # Try to merge
            merged = False

            # Check load constraint
            load_i = sum(demands[n] for n in ri if 1 <= n <= n_customers)
            load_j = sum(demands[n] for n in rj if 1 <= n <= n_customers)
            if load_i + load_j > capacity:
                continue

            # Try merging: reverse ri if needed, then append rj
            # We need to find the best way to merge

            best_merged_route = None
            best_merged_dist = float('inf')

            # Option 1: ri + rj
            seq1 = [c for c in ri if 1 <= c <= n_customers] + [c for c in rj if 1 <= c <= n_customers]
            r1, d1 = build_route_with_charging(seq1)
            if r1 is not None and d1 < best_merged_dist:
                best_merged_dist = d1
                best_merged_route = r1

            # Option 2: ri + reversed rj
            seq2 = [c for c in ri if 1 <= c <= n_customers] + [c for c in reversed(rj) if 1 <= c <= n_customers]
            r2, d2 = build_route_with_charging(seq2)
            if r2 is not None and d2 < best_merged_dist:
                best_merged_dist = d2
                best_merged_route = r2

            # Option 3: reversed ri + rj
            seq3 = [c for c in reversed(ri) if 1 <= c <= n_customers] + [c for c in rj if 1 <= c <= n_customers]
            r3, d3 = build_route_with_charging(seq3)
            if r3 is not None and d3 < best_merged_dist:
                best_merged_dist = d3
                best_merged_route = r3

            # Option 4: reversed ri + reversed rj
            seq4 = [c for c in reversed(ri) if 1 <= c <= n_customers] + [c for c in reversed(rj) if 1 <= c <= n_customers]
            r4, d4 = build_route_with_charging(seq4)
            if r4 is not None and d4 < best_merged_dist:
                best_merged_dist = d4
                best_merged_route = r4

            if best_merged_route is not None:
                # Merge routes i and j
                routes[i] = best_merged_route
                routes[j] = []
                # Update route_of_customer
                for node in best_merged_route:
                    if 1 <= node <= n_customers:
                        route_of_customer[node] = i

        # Remove empty routes
        routes = [r for r in routes if r]
        return routes

    # Greedy construction
    def greedy_construct():
        """Greedy construction: build routes one at a time from remaining customers."""
        routes = []
        remaining = set(range(1, n_customers + 1))

        while remaining and time.time() < deadline - 20.0:
            best_route = None
            best_dist = float('inf')
            best_used = None

            start_candidates = sorted(remaining, key=lambda v: due[v])

            for start_cust in start_candidates[:8]:
                route_seq = [start_cust]
                used = {start_cust}
                current_load = demands[start_cust]

                while time.time() < deadline - 20.0:
                    best_next = None
                    best_metric = float('inf')

                    for c in remaining - used:
                        if current_load + demands[c] > capacity:
                            continue
                        new_seq = route_seq + [c]
                        test_route, d_total = build_route_with_charging(new_seq)
                        if test_route is not None and d_total < best_metric:
                            best_metric = d_total
                            best_next = c

                    if best_next is None:
                        break
                    route_seq.append(best_next)
                    used.add(best_next)
                    current_load += demands[best_next]

                final_route, d_total = build_route_with_charging(route_seq)
                if final_route is not None:
                    d = route_distance(final_route)
                    if d < best_dist:
                        best_dist = d
                        best_route = final_route
                        best_used = set(c for c in route_seq if 1 <= c <= n_customers)

            if best_route is None:
                for c in remaining:
                    sr, _ = build_route_with_charging([c])
                    if sr is not None:
                        routes.append(sr)
                        remaining.discard(c)
                        break
                else:
                    break
            else:
                routes.append(best_route)
                remaining -= best_used

        for c in remaining:
            sr, _ = build_route_with_charging([c])
            if sr is not None:
                routes.append(sr)
            else:
                routes.append([c])

        return routes

    # Large Neighborhood Search
    def lns_optimize(routes, max_iterations=200):
        """Large neighborhood search with destroy and repair."""
        best_routes = [r[:] for r in routes]
        best_cost = total_distance(best_routes)
        current_routes = [r[:] for r in routes]
        current_cost = best_cost

        temperature = 2.0
        cooling = 0.995

        for it in range(max_iterations):
            if time.time() > deadline - 10.0:
                break

            # Destroy: remove k random customers
            k = random.randint(3, min(8, n_customers // 3))
            all_customers = []
            for r in current_routes:
                for node in r:
                    if 1 <= node <= n_customers:
                        all_customers.append((node, r))

            if len(all_customers) < k:
                continue

            # Select customers to destroy
            selected = random.sample(all_customers, min(k, len(all_customers)))
            to_remove = set(node for node, _ in selected)

            # Remove from routes
            new_routes = []
            removed_customers = []
            for r in current_routes:
                new_r = [node for node in r if node not in to_remove]
                if new_r:
                    new_routes.append(new_r)
                for node in r:
                    if node in to_remove:
                        removed_customers.append(node)

            # Repair: insert removed customers one by one
            # Sort by due date for feasibility
            removed_customers.sort(key=lambda c: due[c])

            for cust in removed_customers:
                if time.time() > deadline - 10.0:
                    break

                best_pos = None
                best_route_idx = None
                best_delta = float('inf')

                # Try inserting into each route at each position
                for ri, r in enumerate(new_routes):
                    # Try each position
                    for pos in range(len(r) + 1):
                        # Create new route with cust inserted
                        new_seq = [c for c in r if 1 <= c <= n_customers]
                        # Insert cust at appropriate position
                        cust_positions = [i for i, c in enumerate(new_seq) if 1 <= c <= n_customers]
                        if not cust_positions:
                            insert_pos = 0
                        else:
                            # Find position based on position in original route
                            # Simplified: try all positions
                            insert_pos = pos

                        # Build sequence with cust inserted
                        seq_with_cust = new_seq[:insert_pos] + [cust] + new_seq[insert_pos:]
                        test_route, d_test = build_route_with_charging(seq_with_cust)
                        if test_route is None:
                            continue

                        # Check load
                        load = sum(demands[n] for n in seq_with_cust if 1 <= n <= n_customers)
                        if load > capacity:
                            continue

                        # Calculate delta
                        old_d = simulate_route(r)[0] if r else 0.0
                        delta = d_test - old_d

                        if delta < best_delta:
                            best_delta = delta
                            best_pos = pos
                            best_route_idx = ri

                # Also try creating a new route with just this customer
                single_route, d_single = build_route_with_charging([cust])
                if single_route is not None:
                    if d_single < best_delta:
                        best_delta = d_single
                        best_route_idx = -1  # New route
                        best_pos = 0

                if best_route_idx is not None:
                    if best_route_idx == -1:
                        new_routes.append(single_route)
                    else:
                        r = new_routes[best_route_idx]
                        new_seq = [c for c in r if 1 <= c <= n_customers]
                        # Find actual position to insert
                        # We need to map position back
                        # Simplified: just append and let build_route handle it
                        # Actually, let's rebuild properly
                        cust_in_route = [c for c in r if 1 <= c <= n_customers]
                        # Insert at best_pos in the customer sequence
                        new_cust_seq = cust_in_route[:best_pos] + [cust] + cust_in_route[best_pos:]
                        new_r, _ = build_route_with_charging(new_cust_seq)
                        if new_r is not None:
                            new_routes[best_route_idx] = new_r
                else:
                    # Could not insert, create new route
                    sr, _ = build_route_with_charging([cust])
                    if sr is not None:
                        new_routes.append(sr)

            # Evaluate new solution
            new_cost = total_distance(new_routes)

            # Accept if better or with probability based on temperature
            if new_cost < current_cost:
                current_routes = new_routes
                current_cost = new_cost
                if current_cost < best_cost:
                    best_routes = [r[:] for r in current_routes]
                    best_cost = current_cost
            else:
                # Simulated annealing acceptance
                delta = new_cost - current_cost
                if random.random() < math.exp(-delta / temperature):
                    current_routes = new_routes
                    current_cost = new_cost

            temperature *= cooling

        return best_routes, best_cost

    # Local search: 2-opt, relocate, swap, or-opt
    def local_search(routes, max_iterations=50):
        """Apply local search improvements."""
        best_routes = [r[:] for r in routes]
        best_cost = total_distance(best_routes)
        current_routes = [r[:] for r in routes]

        for it in range(max_iterations):
            if time.time() > deadline - 10.0:
                break

            improved = False

            # 2-opt within each route
            for i in range(len(current_routes)):
                if time.time() > deadline - 10.0:
                    break

                cust_seq = [c for c in current_routes[i] if 1 <= c <= n_customers]
                if len(cust_seq) < 2:
                    continue

                best_seq = cust_seq[:]
                best_route, best_d = build_route_with_charging(cust_seq)
                if best_route is None:
                    continue

                for _ in range(3):
                    changed = False
                    for a in range(len(best_seq)):
                        for b in range(a + 1, len(best_seq)):
                            new_seq = best_seq[:a] + best_seq[a:b+1][::-1] + best_seq[b+1:]
                            r, d = build_route_with_charging(new_seq)
                            if r is not None and d < best_d - 0.05:
                                best_d = d
                                best_seq = new_seq[:]
                                best_route = r
                                changed = True
                    if not changed:
                        break

                if best_d < simulate_route(current_routes[i])[0] - 0.05:
                    current_routes[i] = best_route
                    improved = True

            # Relocate between routes
            for i in range(len(current_routes)):
                if time.time() > deadline - 10.0:
                    break
                for j in range(len(current_routes[i])):
                    cust = current_routes[i][j]
                    if not (1 <= cust <= n_customers):
                        continue
                    for k in range(len(current_routes)):
                        if k == i or time.time() > deadline - 10.0:
                            continue
                        for pos in range(len(current_routes[k]) + 1):
                            # Try relocating
                            new_i_seq = [c for idx, c in enumerate(current_routes[i]) if idx != j]
                            new_k_seq = current_routes[k][:pos] + [cust] + current_routes[k][pos:]

                            r_i, d_i = build_route_with_charging(new_i_seq)
                            r_k, d_k = build_route_with_charging(new_k_seq)

                            if r_i is None or r_k is None:
                                continue

                            set_i = set(c for c in r_i if 1 <= c <= n_customers)
                            set_k = set(c for c in r_k if 1 <= c <= n_customers)
                            if set_i & set_k:
                                continue

                            old_dist = simulate_route(current_routes[i])[0] + simulate_route(current_routes[k])[0]
                            new_dist = d_i + d_k

                            if new_dist < old_dist - 0.1:
                                current_routes[i] = r_i if r_i else []
                                current_routes[k] = r_k
                                if not current_routes[i]:
                                    del current_routes[i]
                                    if k > i:
                                        k -= 1
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            # Or-opt: move 2-3 consecutive customers
            if not improved:
                for i in range(len(current_routes)):
                    if time.time() > deadline - 10.0:
                        break
                    for seg_len in [2, 3]:
                        for j in range(len(current_routes[i]) - seg_len + 1):
                            seg = current_routes[i][j:j+seg_len]
                            if not all(1 <= c <= n_customers for c in seg):
                                continue

                            new_i_seq = current_routes[i][:j] + current_routes[i][j+seg_len:]

                            for k in range(len(current_routes)):
                                if time.time() > deadline - 10.0:
                                    break
                                target_route = current_routes[k] if k != i else new_i_seq
                                for pos in range(len(target_route) + 1):
                                    new_k_seq = target_route[:pos] + seg + target_route[pos:]

                                    if k == i:
                                        if pos == j or pos == j + seg_len:
                                            continue
                                        r_i, d_i = build_route_with_charging(new_k_seq)
                                        if r_i is not None:
                                            old_dist = simulate_route(current_routes[i])[0]
                                            if d_i < old_dist - 0.05:
                                                current_routes[i] = r_i
                                                improved = True
                                                break
                                    else:
                                        r_i, d_i = build_route_with_charging(new_i_seq)
                                        r_k, d_k = build_route_with_charging(new_k_seq)
                                        if r_i is not None and r_k is not None:
                                            old_dist = simulate_route(current_routes[i])[0] + simulate_route(current_routes[k])[0]
                                            if d_i + d_k < old_dist - 0.05:
                                                current_routes[i] = r_i if r_i else []
                                                current_routes[k] = r_k
                                                if not current_routes[i]:
                                                    del current_routes[i]
                                                    if k > i:
                                                        k -= 1
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

            # Update best
            current_cost = total_distance(current_routes)
            if current_cost < best_cost:
                best_routes = [r[:] for r in current_routes]
                best_cost = current_cost

        return best_routes, best_cost

    # Main: try multiple construction strategies and optimize
    best_routes = None
    best_total = float('inf')

    # Strategy 1: Savings construction
    if time.time() < deadline - 30.0:
        routes = savings_construct()
        covered = set()
        for r in routes:
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)
        if covered != set(range(1, n_customers + 1)):
            for c in set(range(1, n_customers + 1)) - covered:
                sr, _ = build_route_with_charging([c])
                if sr is not None:
                    routes.append(sr)
                else:
                    routes.append([c])
        td = total_distance(routes)
        if td < best_total:
            best_total = td
            best_routes = routes[:]

    # Strategy 2: Greedy construction
    if time.time() < deadline - 25.0:
        routes = greedy_construct()
        covered = set()
        for r in routes:
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)
        if covered != set(range(1, n_customers + 1)):
            for c in set(range(1, n_customers + 1)) - covered:
                sr, _ = build_route_with_charging([c])
                if sr is not None:
                    routes.append(sr)
                else:
                    routes.append([c])
        td = total_distance(routes)
        if td < best_total:
            best_total = td
            best_routes = routes[:]

    if best_routes is None:
        best_routes = []
        for c in range(1, n_customers + 1):
            sr, _ = build_route_with_charging([c])
            if sr is not None:
                best_routes.append(sr)
            else:
                best_routes.append([c])
        best_total = total_distance(best_routes)

    # Apply local search
    if time.time() < deadline - 20.0:
        best_routes, best_total = local_search(best_routes, max_iterations=30)

    # Apply LNS
    if time.time() < deadline - 15.0:
        lns_routes, lns_cost = lns_optimize(best_routes, max_iterations=100)
        if lns_cost < best_total:
            best_routes = lns_routes
            best_total = lns_cost

    # Final validation and cleanup
    valid_routes = []
    covered = set()
    for r in best_routes:
        d, valid = simulate_route(r)
        if valid:
            valid_routes.append(r)
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)

    for cust in range(1, n_customers + 1):
        if cust not in covered:
            sr, _ = build_route_with_charging([cust])
            if sr is not None:
                valid_routes.append(sr)
            else:
                valid_routes.append([cust])
            covered.add(cust)

    return valid_routes

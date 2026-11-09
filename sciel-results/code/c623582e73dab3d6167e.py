"""EV routing plan – stronger cheapest-insertion + Or-opt + relocate + swap + 2-opt."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def plan(instance):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill

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

    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    # Precompute a "latest arrival" bound: for customer v, we need to arrive by due[v]
    # and leave by due[v] + service[v]. The trip back to depot must fit in horizon.
    # So latest we can arrive at v is: min(due[v], horizon - service[v] - dist[v, 0] / speed)

    latest_arrival = np.minimum(due, horizon - service - dist[:, 0] / speed)

    # Also, battery constraint on return: we need dist[v, 0] battery remaining after service
    # Since battery only decreases on travel, the return distance must be <= remaining battery.
    # We'll handle this in the route check.

    def check_route(route):
        """Check feasibility of a single route. Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0

        total_demand = 0.0
        for v in route:
            total_demand += demands[v]
        if total_demand > capacity + 1e-9:
            return False, 0.0

        t = 0.0
        bat = float(battery)
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

    def cheapest_insertion_construct():
        """Construct routes via cheapest insertion with time windows and EV constraints."""
        # Sort customers by tightness (due date ascending) for insertion order
        order = sorted(customers, key=lambda v: due[v])

        routes = []
        # We'll use a list of lists; each route is a list of customer node ids

        for v in order:
            best_cost = float('inf')
            best_route_idx = -1
            best_pos = -1
            best_new_route = None

            # Try inserting into each existing route at each position
            for ri, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [v] + route[pos:]
                    feasible, rd = check_route(new_route)
                    if not feasible:
                        continue
                    # Cost: increase in total distance from inserting v at pos in route
                    # Old distance contribution of the segment:
                    if pos == 0:
                        old_seg = dist[0, route[0]]
                    elif pos == len(route):
                        old_seg = dist[route[-1], 0]
                    else:
                        old_seg = dist[route[pos - 1], route[pos]]
                    # New distance contribution:
                    if pos == 0:
                        new_seg = dist[0, v] + dist[v, route[0]] - dist[0, route[0]]
                    elif pos == len(route):
                        new_seg = dist[route[-1], v] + dist[v, 0] - dist[route[-1], 0]
                    else:
                        new_seg = dist[route[pos - 1], v] + dist[v, route[pos]] - dist[route[pos - 1], route[pos]]
                    # Actually, the total route distance change is:
                    # new route dist - old route dist
                    # Let's just compute the full route distance for correctness
                    # But that's O(n) per insertion. For 25 customers it's fine.
                    # Better: compute the delta properly.
                    # Old route distance: dist[0, route[0]] + sum(dist[route[i],route[i+1]]) + dist[route[-1],0]
                    # New route distance: same but with v inserted.
                    # The delta is: (dist[prev, v] + dist[v, nxt]) - dist[prev, nxt]
                    # where prev/nxt depend on pos.
                    if pos == 0:
                        prev_node = 0
                        nxt_node = route[0]
                    elif pos == len(route):
                        prev_node = route[-1]
                        nxt_node = 0
                    else:
                        prev_node = route[pos - 1]
                        nxt_node = route[pos]

                    delta = dist[prev_node, v] + dist[v, nxt_node] - dist[prev_node, nxt_node]
                    if delta < best_cost:
                        best_cost = delta
                        best_route_idx = ri
                        best_pos = pos
                        best_new_route = new_route

            # If no existing route works, start a new route
            if best_route_idx == -1:
                new_route = [v]
                feasible, _ = check_route(new_route)
                if feasible:
                    routes.append(new_route)
                # If even a single customer is infeasible, skip (shouldn't happen)
            else:
                routes[best_route_idx] = best_new_route

        return [r for r in routes if r]

    def two_opt_route(route):
        """Apply 2-opt to a single route until no improvement or deadline."""
        if len(route) < 2:
            return route
        n = len(route)
        improved = True
        passes = 0
        while improved and passes < 50:
            improved = False
            passes += 1
            for a in range(n - 1):
                if time.time() > deadline:
                    return route
                for b in range(a + 1, n):
                    # Compute old and new boundary costs
                    if a == 0:
                        old1 = dist[0, route[a]]
                    else:
                        old1 = dist[route[a - 1], route[a]]
                    if b == n - 1:
                        old2 = dist[route[b], 0]
                    else:
                        old2 = dist[route[b], route[b + 1]]

                    if a == 0:
                        new1 = dist[0, route[b]]
                    else:
                        new1 = dist[route[a - 1], route[b]]
                    if b == n - 1:
                        new2 = dist[route[a], 0]
                    else:
                        new2 = dist[route[a], route[b + 1]]

                    if new1 + new2 < old1 + old2 - 1e-6:
                        new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                        feasible, _ = check_route(new_route)
                        if feasible:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    def or_opt_route(route):
        """Apply Or-opt (move 1, 2, or 3 consecutive nodes) to a single route."""
        n = len(route)
        if n < 2:
            return route
        improved = True
        passes = 0
        while improved and passes < 30:
            improved = False
            passes += 1
            for seg_len in [1, 2, 3]:
                if time.time() > deadline:
                    return route
                for i in range(n - seg_len + 1):
                    if time.time() > deadline:
                        return route
                    segment = route[i:i + seg_len]
                    # Remove segment
                    base = route[:i] + route[i + seg_len:]
                    if not base:
                        continue
                    # Try reinserting at each position in base
                    for pos in range(len(base) + 1):
                        if pos >= i:
                            # Avoid inserting back at same or later position (would be same)
                            # Actually if pos > i, it's a different position; if pos == i, same
                            # We skip pos == i to avoid no-op
                            if pos == i:
                                continue
                        new_route = base[:pos] + segment + base[pos:]
                        # Quick distance delta check before full feasibility
                        # Just do full check for simplicity
                        feasible, rd = check_route(new_route)
                        if not feasible:
                            continue
                        # Compare distances
                        old_rd = 0.0
                        old_rd += dist[0, route[0]]
                        for k in range(len(route) - 1):
                            old_rd += dist[route[k], route[k + 1]]
                        old_rd += dist[route[-1], 0]
                        if rd < old_rd - 1e-6:
                            route = new_route
                            n = len(route)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return route

    def relocate_search(routes, max_passes=8):
        """Remove-and-reinsert local search across routes."""
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
                    route_i_without = routes[i][:j] + routes[i][j + 1:]
                    if not route_i_without:
                        new_routes = [r for idx, r in enumerate(routes) if idx != i]
                    else:
                        new_routes = list(routes)
                        new_routes[i] = route_i_without

                    if route_i_without:
                        feasible_i, _ = check_route(route_i_without)
                        if not feasible_i:
                            continue
                    else:
                        feasible_i = True

                    best_new_dist = current_dist
                    best_new_routes = None
                    best_target = -1
                    best_pos = -1

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
                            new_routes[k] = route_k  # restore

                            if new_dist < best_new_dist - 1e-6:
                                best_new_dist = new_dist
                                best_new_routes = None  # we'll rebuild
                                best_target = k
                                best_pos = pos

                    if best_target != -1:
                        # Rebuild
                        if best_target < i:
                            target_idx = best_target
                        else:
                            target_idx = best_target - 1 if best_target > i else best_target
                        # Actually, let's just rebuild properly
                        new_routes = list(routes)
                        new_routes[i] = route_i_without
                        if not route_i_without:
                            new_routes = [r for idx, r in enumerate(new_routes) if idx != i]
                        # Find the target route index in new_routes
                        # This is getting complex; let's use a different approach
                        # Just iterate and try all, keep the best
                        pass

                    # Simpler: just try all and apply first improvement
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

    def swap_search(routes, max_passes=5):
        """Cross-route swap local search."""
        routes = [list(r) for r in routes]
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            current_dist = total_route_distance(routes)
            n_routes = len(routes)

            for i in range(n_routes):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_routes):
                    if time.time() > deadline:
                        break
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

    # Initial solution
    best_routes = [[v] for v in customers]
    best_distance = sum(dist[0, v] * 2 for v in customers)

    # Try cheapest insertion construction
    if time.time() < deadline:
        ci_routes = cheapest_insertion_construct()
        if ci_routes and all(check_route(r)[0] for r in ci_routes):
            all_cust = []
            for r in ci_routes:
                all_cust.extend(r)
            if set(all_cust) == set(customers) and len(all_cust) == len(customers):
                d = total_route_distance(ci_routes)
                if d < best_distance:
                    best_routes = ci_routes
                    best_distance = d

    # Try cheapest insertion with different orderings
    orderings = [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: dist[0, v]),
        sorted(customers, key=lambda v: (due[v], dist[0, v])),
        sorted(customers, key=lambda v: (latest_arrival[v], dist[0, v])),
    ]

    for ordering in orderings:
        if time.time() > deadline:
            break
        # Temporarily override the order
        orig_order = customers[:]
        # We'll call a variant of cheapest insertion with this order
        # For simplicity, we'll just try the same function but it uses due[v] internally
        # Let's create a modified version
        def ci_with_order(ord_list):
            routes = []
            for v in ord_list:
                best_cost = float('inf')
                best_ri = -1
                best_pos = -1
                best_new = None

                for ri, route in enumerate(routes):
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [v] + route[pos:]
                        feasible, _ = check_route(new_route)
                        if not feasible:
                            continue
                        if pos == 0:
                            prev_node = 0
                            nxt_node = route[0]
                        elif pos == len(route):
                            prev_node = route[-1]
                            nxt_node = 0
                        else:
                            prev_node = route[pos - 1]
                            nxt_node = route[pos]
                        delta = dist[prev_node, v] + dist[v, nxt_node] - dist[prev_node, nxt_node]
                        if delta < best_cost:
                            best_cost = delta
                            best_ri = ri
                            best_pos = pos
                            best_new = new_route

                if best_ri == -1:
                    routes.append([v])
                else:
                    routes[best_ri] = best_new
            return [r for r in routes if r]

        ci_routes = ci_with_order(ordering)
        if ci_routes:
            all_cust = []
            for r in ci_routes:
                all_cust.extend(r)
            if set(all_cust) == set(customers) and len(all_cust) == len(customers):
                d = total_route_distance(ci_routes)
                if d < best_distance:
                    best_routes = ci_routes
                    best_distance = d

    # Apply 2-opt to all routes
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    # Apply Or-opt
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = or_opt_route(best_routes[i])

    # Apply relocate search
    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=8)

    # Apply swap search
    if time.time() < deadline:
        best_routes = swap_search(best_routes, max_passes=5)

    # Final polish: 2-opt + Or-opt + relocate
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = or_opt_route(best_routes[i])

    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=5)

    # Final 2-opt
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    # Validate: all customers served exactly once
    all_cust = []
    for r in best_routes:
        all_cust.extend(r)
    if set(all_cust) != set(customers) or len(all_cust) != len(customers):
        # Fallback
        return [[v] for v in customers]

    # Validate feasibility
    for r in best_routes:
        if not check_route(r)[0]:
            return [[v] for v in customers]

    return best_routes
# EVOLVE-BLOCK-END

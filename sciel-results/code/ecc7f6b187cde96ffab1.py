"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _route_state(route, D, instance):
    """Return (time, battery, dist, feasible, demand) for a route."""
    if not route:
        return 0.0, instance.get("battery", 300), 0.0, True, 0
    speed = instance["speed"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = instance["_stations_set"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    t = 0.0
    bat = float(battery_cap)
    dist = 0.0
    demand = 0
    prev = 0
    for v in route:
        d = D[prev, v]
        t += d
        bat -= d
        dist += d
        if bat < -1e-9:
            return t, bat, dist, False, demand
        if v in stations:
            energy_needed = battery_cap - bat
            t += energy_needed * recharge_rate
            bat = float(battery_cap)
        else:
            demand += demands[v]
            if demand > instance["capacity"] + 1e-9:
                return t, bat, dist, False, demand
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return t, bat, dist, False, demand
            t += service[v]
        prev = v
    d = D[prev, 0]
    t += d
    bat -= d
    dist += d
    if bat < -1e-9:
        return t, bat, dist, False, demand
    if t > horizon + 1e-9:
        return t, bat, dist, False, demand
    return t, bat, dist, True, demand


def _route_feasible(route, D, instance):
    if not route:
        return True
    _, _, _, feas, _ = _route_state(route, D, instance)
    return feas


def _route_dist(route, D):
    if not route:
        return 0.0
    total = D[0, route[0]]
    for i in range(len(route) - 1):
        total += D[route[i], route[i + 1]]
    total += D[route[-1], 0]
    return total


def _total_dist(routes, D):
    return sum(_route_dist(r, D) for r in routes)


def _construct_greedy(instance, D):
    """Greedy insertion construction for EVRP."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    demands = instance["demands"]
    cap = instance["capacity"]
    ready = instance["ready"]
    due = instance["due"]
    battery_cap = instance["battery"]
    horizon = instance["horizon"]
    recharge_rate = instance["recharge_rate"]
    stations = instance["_stations_set"]

    # Sort customers by due date (earliest first)
    sorted_customers = sorted(customers, key=lambda v: due[v])

    routes = []
    route_demands = []

    for v in sorted_customers:
        best_pos = None
        best_route_idx = -1
        best_cost = float('inf')

        for ri, route in enumerate(routes):
            if route_demands[ri] + demands[v] > cap + 1e-9:
                continue

            for pos in range(len(route) + 1):
                new_route = route[:pos] + [v] + route[pos:]
                if not _route_feasible(new_route, D, instance):
                    continue
                if pos == 0:
                    old_cost = D[0, route[0]] if route else 0
                    new_cost = D[0, v]
                elif pos == len(route):
                    old_cost = D[route[-1], 0]
                    new_cost = D[v, 0]
                else:
                    old_cost = D[route[pos - 1], route[pos]]
                    new_cost = D[route[pos - 1], v] + D[v, route[pos]]
                delta_cost = new_cost - old_cost
                if delta_cost < best_cost:
                    best_cost = delta_cost
                    best_pos = pos
                    best_route_idx = ri

        if best_route_idx >= 0:
            routes[best_route_idx].insert(best_pos, v)
            route_demands[best_route_idx] += demands[v]
        else:
            routes.append([v])
            route_demands.append(demands[v])

    return routes


def _construct_savings(instance, D, deadline):
    """Clarke-Wright savings construction adapted for EVRP."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    cap = instance["capacity"]
    demands = instance["demands"]

    routes = [[v] for v in customers]
    route_of = {v: i for i, r in enumerate(routes) for v in r}
    active = set(range(len(routes)))

    savings_list = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            a = customers[i]
            b = customers[j]
            s = D[0, a] + D[0, b] - D[a, b]
            savings_list.append((s, a, b))
    savings_list.sort(reverse=True)

    for s, a, b in savings_list:
        if time.time() > deadline:
            break
        if s <= 0:
            break
        ri = route_of.get(a, -1)
        rj = route_of.get(b, -1)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        if ri not in active or rj not in active:
            continue

        best_merge = None
        best_dist = float('inf')

        a_at_end = (routes[ri][-1] == a)
        a_at_start = (routes[ri][0] == a)
        b_at_end = (routes[rj][-1] == b)
        b_at_start = (routes[rj][0] == b)

        candidates = []
        if a_at_end and b_at_start:
            candidates.append((ri, rj, False, False))
        if b_at_end and a_at_start:
            candidates.append((rj, ri, False, False))
        if a_at_start and b_at_start:
            candidates.append((ri, rj, True, False))
            candidates.append((rj, ri, True, False))
        if a_at_end and b_at_end:
            candidates.append((ri, rj, False, True))
            candidates.append((rj, ri, True, True))
        if a_at_start and b_at_end:
            candidates.append((ri, rj, True, True))
            candidates.append((rj, ri, True, True))
        if a_at_end and b_at_end:
            candidates.append((ri, rj, False, True))
        if a_at_start and b_at_start:
            candidates.append((ri, rj, True, False))

        for first, second, rev_first, rev_second in candidates:
            if first not in active or second not in active:
                continue
            new_route = routes[first][:]
            if rev_first:
                new_route = new_route[::-1]
            second_route = routes[second][:]
            if rev_second:
                second_route = second_route[::-1]
            new_route = new_route + second_route

            if sum(demands[v] for v in new_route) > cap + 1e-9:
                continue
            if not _route_feasible(new_route, D, instance):
                continue

            nd = _route_dist(new_route, D)
            if nd < best_dist:
                best_dist = nd
                best_merge = (first, second, new_route)

        if best_merge is not None:
            first, second, new_route = best_merge
            routes[first] = new_route
            active.discard(second)
            for v in new_route:
                route_of[v] = first

    final_routes = [routes[i] for i in range(len(routes)) if i in active and routes[i]]
    all_assigned = set()
    for r in final_routes:
        all_assigned.update(r)
    missing = set(customers) - all_assigned
    for v in missing:
        final_routes.append([v])
    return final_routes


def _2opt_fast(route, D, instance, max_iter=200):
    """Fast 2-opt with precomputed feasibility hints."""
    if len(route) < 2:
        return route
    best = route[:]
    n = len(best)

    for _ in range(max_iter):
        improved = False
        for i in range(n - 1):
            a = best[i - 1] if i > 0 else 0
            b = best[i]
            for j in range(i + 1, n):
                c = best[j]
                d = best[j + 1] if j + 1 < n else 0
                old_seg = D[a, b] + D[c, d]
                new_seg = D[a, c] + D[b, d]
                if new_seg >= old_seg - 1e-9:
                    continue
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if not _route_feasible(new_route, D, instance):
                    continue
                best = new_route
                n = len(best)
                improved = True
                break
            if improved:
                break
        if not improved:
            break
    return best


def _3opt_fast(route, D, instance, max_iter=50):
    """3-opt: reverse a segment between two positions."""
    if len(route) < 3:
        return route
    best = route[:]
    n = len(best)

    for _ in range(max_iter):
        improved = False
        for i in range(n - 1):
            a = best[i - 1] if i > 0 else 0
            b = best[i]
            for j in range(i + 1, n):
                c = best[j]
                d = best[j + 1] if j + 1 < n else 0
                # Try reversing segment [i..j]
                old_seg = D[a, b] + D[c, d]
                new_seg = D[a, c] + D[b, d]
                if new_seg < old_seg - 1e-9:
                    new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    if _route_feasible(new_route, D, instance):
                        best = new_route
                        n = len(best)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    return best


def _or_opt_fast(route, D, instance, max_iter=50):
    """Fast Or-opt: relocate single nodes."""
    if len(route) < 3:
        return route
    best = route[:]
    n = len(best)
    best_dist = _route_dist(best, D)

    for _ in range(max_iter):
        improved = False
        for i in range(n):
            node = best[i]
            remaining = best[:i] + best[i + 1:]
            for j in range(len(remaining) + 1):
                if j == i or j == i - 1:
                    continue
                candidate = remaining[:j] + [node] + remaining[j:]
                if candidate == best:
                    continue
                if not _route_feasible(candidate, D, instance):
                    continue
                cd = _route_dist(candidate, D)
                if cd < best_dist - 1e-9:
                    best = candidate
                    n = len(best)
                    best_dist = cd
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _relocate_inter(routes, D, instance, max_iters=30):
    """Inter-route relocate moves with delta-cost evaluation."""
    for _ in range(max_iters):
        improved = False
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for j in range(len(routes)):
                if i == j:
                    continue
                for ai in range(len(routes[i])):
                    v = routes[i][ai]
                    new_i = routes[i][:]
                    new_i.pop(ai)
                    if not _route_feasible(new_i, D, instance):
                        continue
                    old_di = _route_dist(routes[i], D)
                    new_di = _route_dist(new_i, D)
                    for aj in range(len(routes[j]) + 1):
                        new_j = routes[j][:]
                        new_j.insert(aj, v)
                        if not _route_feasible(new_j, D, instance):
                            continue
                        old_dj = _route_dist(routes[j], D)
                        new_dj = _route_dist(new_j, D)
                        if new_di + new_dj < old_di + old_dj - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
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


def _swap_inter(routes, D, instance, max_iters=10):
    """Inter-route swap moves."""
    for _ in range(max_iters):
        improved = False
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for j in range(i + 1, len(routes)):
                if not routes[j]:
                    continue
                for ai in range(len(routes[i])):
                    for aj in range(len(routes[j])):
                        vi = routes[i][ai]
                        vj = routes[j][aj]
                        new_i = routes[i][:]
                        new_j = routes[j][:]
                        new_i[ai] = vj
                        new_j[aj] = vi
                        if not _route_feasible(new_i, D, instance):
                            continue
                        if not _route_feasible(new_j, D, instance):
                            continue
                        old_d = _route_dist(routes[i], D) + _route_dist(routes[j], D)
                        new_d = _route_dist(new_i, D) + _route_dist(new_j, D)
                        if new_d < old_d - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
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


def _optimize_routes(routes, D, instance, deadline):
    """Run local search on routes with time guard."""
    # Intra-route optimization
    for i in range(len(routes)):
        if time.time() > deadline - 40:
            break
        routes[i] = _2opt_fast(routes[i], D, instance, max_iter=100)
        routes[i] = _or_opt_fast(routes[i], D, instance, max_iter=30)
        routes[i] = _3opt_fast(routes[i], D, instance, max_iter=30)

    # Inter-route optimization
    if time.time() < deadline - 25:
        routes = _relocate_inter(routes, D, instance, max_iters=20)
        routes = _swap_inter(routes, D, instance, max_iters=8)

    # Second intra-route pass after inter-route moves
    for i in range(len(routes)):
        if time.time() > deadline - 15:
            break
        routes[i] = _2opt_fast(routes[i], D, instance, max_iter=80)
        routes[i] = _or_opt_fast(routes[i], D, instance, max_iter=25)
        routes[i] = _3opt_fast(routes[i], D, instance, max_iter=20)

    # Final high-precision polish
    for i in range(len(routes)):
        if time.time() > deadline - 3:
            break
        routes[i] = _2opt_fast(routes[i], D, instance, max_iter=150)
        routes[i] = _or_opt_fast(routes[i], D, instance, max_iter=40)

    return [r for r in routes if r]


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Precompute
    coords = instance["coords"]
    D = _dist_matrix(coords)
    instance["_stations_set"] = set(instance.get("stations", []))

    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))

    # Try both construction methods, keep the better one
    routes_greedy = _construct_greedy(instance, D)
    dist_greedy = _total_dist(routes_greedy, D)

    # Verify all customers covered
    all_assigned = set()
    for r in routes_greedy:
        all_assigned.update(r)
    missing = set(customers) - all_assigned
    for v in missing:
        routes_greedy.append([v])

    # Try savings construction if time permits
    if time.time() < deadline - 60:
        routes_savings = _construct_savings(instance, D, deadline - 50)
        all_assigned = set()
        for r in routes_savings:
            all_assigned.update(r)
        missing = set(customers) - all_assigned
        for v in missing:
            routes_savings.append([v])
        dist_savings = _total_dist(routes_savings, D)

        if dist_savings < dist_greedy:
            routes = routes_savings
        else:
            routes = routes_greedy
    else:
        routes = routes_greedy

    # Optimize
    routes = _optimize_routes(routes, D, instance, deadline)

    # Final feasibility check and cleanup
    valid_routes = []
    all_served = set()
    for r in routes:
        if _route_feasible(r, D, instance):
            valid_routes.append(r)
            all_served.update(r)
    
    # If any customer is missing, add them as single-node routes
    missing = set(customers) - all_served
    for v in missing:
        valid_routes.append([v])

    return valid_routes
# EVOLVE-BLOCK-END

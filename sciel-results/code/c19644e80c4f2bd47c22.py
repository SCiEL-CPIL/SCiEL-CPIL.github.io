"""Rewritten EV routing solver with LNS + SA and vectorized feasibility."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

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

    # Precompute latest departure time: latest_dep[i][j] = due[j] - dist[i][j]/speed
    latest_dep = due[np.newaxis, :] - dist / speed

    def route_distance(route):
        """Compute total Euclidean distance of a route including depot legs."""
        if not route:
            return 0.0
        d = dist[0, route[0]]
        for i in range(len(route) - 1):
            d += dist[route[i], route[i + 1]]
        d += dist[route[-1], 0]
        return d

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def check_route(route):
        """Check feasibility of a single route. Returns (feasible, total_dist)."""
        if not route:
            return True, 0.0
        load = sum(demands[v] for v in route)
        if load > capacity:
            return False, 0.0
        t = 0.0
        bat = battery
        prev = 0
        td = 0.0
        for v in route:
            d = dist[prev, v]
            bat -= d
            if bat < -1e-9:
                return False, 0.0
            t += d / speed
            td += d
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
        td += d
        if t > horizon + 1e-9:
            return False, 0.0
        return True, td

    def check_route_fast(route, load):
        """Faster check assuming load is precomputed and <= capacity."""
        if not route:
            return True, 0.0
        if load > capacity:
            return False, 0.0
        t = 0.0
        bat = battery
        prev = 0
        td = 0.0
        for v in route:
            d = dist[prev, v]
            bat -= d
            if bat < -1e-9:
                return False, 0.0
            # Quick time-window check using latest_dep
            if t > latest_dep[prev, v] + 1e-9:
                return False, 0.0
            t += d / speed
            td += d
            if t < ready[v]:
                t = ready[v]
            t += service[v]
            prev = v
        d = dist[prev, 0]
        bat -= d
        if bat < -1e-9:
            return False, 0.0
        t += d / speed
        td += d
        if t > horizon + 1e-9:
            return False, 0.0
        return True, td

    def validate(routes):
        """Check all customers served exactly once and all routes feasible."""
        seen = set()
        for r in routes:
            if not r:
                continue
            f, _ = check_route(r)
            if not f:
                return False
            for v in r:
                if v in seen:
                    return False
                seen.add(v)
        return seen == set(customers)

    # --- Construction: Greedy with multiple orderings ---
    def greedy_construct(order):
        remaining = list(order)
        routes = []
        while remaining:
            route = []
            prev = 0
            t = 0.0
            bat = battery
            load = 0.0
            while remaining:
                best = None
                best_score = float('inf')
                best_idx = -1
                for idx, v in enumerate(remaining):
                    d = dist[prev, v]
                    if bat - d < -1e-9:
                        continue
                    if load + demands[v] > capacity:
                        continue
                    if t > latest_dep[prev, v] + 1e-9:
                        continue
                    t_arr = t + d / speed
                    if t_arr < ready[v]:
                        t_arr = ready[v]
                    if t_arr > due[v] + 1e-9:
                        continue
                    slack = due[v] - t_arr
                    score = d - 0.01 * slack
                    if score < best_score:
                        best_score = score
                        best = v
                        best_idx = idx
                if best is None:
                    break
                v = best
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

    # --- Construction: Savings ---
    def savings_construct():
        routes = [[v] for v in customers]
        route_of = {v: i for i, v in enumerate(customers)}
        route_load = [demands[v] for v in customers]

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
            ri, rj = min(ri, rj), max(ri, rj)
            route_i = routes[ri]
            route_j = routes[rj]
            load = route_load[ri] + route_load[rj]
            if load > capacity:
                continue

            best_new = None
            # Try concatenations
            if route_i[-1] == vi and route_j[0] == vj:
                cand = route_i + route_j
            elif route_j[-1] == vj and route_i[0] == vi:
                cand = route_j + route_i
            elif route_i[0] == vi and route_j[0] == vj:
                cand = route_i[::-1] + route_j
            elif route_i[-1] == vi and route_j[-1] == vj:
                cand = route_i + route_j[::-1]
            elif route_i[0] == vi and route_j[-1] == vj:
                cand = route_i[::-1] + route_j[::-1]
            elif route_j[0] == vj and route_i[-1] == vi:
                cand = route_j + route_i[::-1]
            else:
                continue

            f, _ = check_route_fast(cand, load)
            if f:
                best_new = cand
                break

            if best_new is None:
                continue

            routes[ri] = best_new
            routes[rj] = []
            route_load[ri] = load
            route_load[rj] = 0
            for v in best_new:
                route_of[v] = ri
            # Compact
            non_empty = [(i, r) for i, r in enumerate(routes) if r]
            new_routes = []
            new_loads = []
            for idx, r in non_empty:
                new_routes.append(r)
                new_loads.append(route_load[idx])
                for v in r:
                    route_of[v] = len(new_routes) - 1
            routes = new_routes
            route_load = new_loads

        return [r for r in routes if r]

    # --- Local Search: 2-opt within a route ---
    def two_opt_route(route):
        if len(route) < 2:
            return route
        load = sum(demands[v] for v in route)
        improved = True
        while improved:
            improved = False
            n = len(route)
            for a in range(n - 1):
                if time.time() > deadline:
                    return route
                for b in range(a + 1, n):
                    old_b = 0.0
                    if a > 0:
                        old_b += dist[route[a-1], route[a]]
                    else:
                        old_b += dist[0, route[a]]
                    if b < n - 1:
                        old_b += dist[route[b], route[b+1]]
                    else:
                        old_b += dist[route[b], 0]

                    new_b = 0.0
                    if a > 0:
                        new_b += dist[route[a-1], route[b]]
                    else:
                        new_b += dist[0, route[b]]
                    if b < n - 1:
                        new_b += dist[route[a], route[b+1]]
                    else:
                        new_b += dist[route[a], 0]

                    if new_b < old_b - 1e-6:
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        f, _ = check_route_fast(new_route, load)
                        if f:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    # --- Local Search: Relocate / Or-opt ---
    def relocate_search(routes, max_passes=6):
        routes = [list(r) for r in routes]
        route_loads = [sum(demands[v] for v in r) for r in routes]
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            cur_dist = total_distance(routes)

            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for seg_len in [1, 2, 3]:
                    if time.time() > deadline:
                        break
                    for j in range(len(routes[i]) - seg_len + 1):
                        if time.time() > deadline:
                            break
                        seg = routes[i][j:j+seg_len]
                        seg_load = sum(demands[v] for v in seg)
                        rest_i = routes[i][:j] + routes[i][j+seg_len:]

                        if rest_i:
                            f_i, _ = check_route_fast(rest_i, route_loads[i] - seg_load)
                            if not f_i:
                                continue

                        # Try inserting into each route at each position
                        best_delta = 0.0
                        best_k = -1
                        best_pos = -1
                        for k in range(len(routes)):
                            if time.time() > deadline:
                                break
                            rk = routes[k] if k != i else rest_i
                            for pos in range(len(rk) + 1):
                                new_rk = rk[:pos] + seg + rk[pos:]
                                new_load_k = (route_loads[k] if k != i else route_loads[i] - seg_load) + seg_load
                                if new_load_k > capacity:
                                    continue
                                f, _ = check_route_fast(new_rk, new_load_k)
                                if not f:
                                    continue
                                # Compute delta
                                old_d = 0.0
                                # Remove seg from route i
                                if j > 0:
                                    old_d += dist[rest_i[j-1], seg[0]] if j < len(routes[i]) else 0
                                # This is getting complex; just compute full
                                # For speed, approximate:
                                delta = 0.0
                                # Old edges around segment in route i
                                if j > 0:
                                    delta += dist[rest_i[j-1], seg[0]]
                                if j + seg_len < len(routes[i]):
                                    delta += dist[seg[-1], routes[i][j+seg_len]]
                                # New edges in route k
                                if pos > 0:
                                    delta -= dist[rk[pos-1], rk[pos]] if pos < len(rk) else dist[rk[pos-1], 0]
                                if pos < len(rk):
                                    delta -= dist[rk[pos], rk[pos+1]] if pos+1 < len(rk) else dist[rk[pos], 0]
                                if pos > 0:
                                    delta += dist[rk[pos-1], seg[0]]
                                if pos < len(rk):
                                    delta += dist[seg[-1], rk[pos]]
                                if j > 0 and j + seg_len < len(routes[i]):
                                    delta += dist[rest_i[j-1], routes[i][j+seg_len]]

                                if delta < best_delta - 1e-6:
                                    best_delta = delta
                                    best_k = k
                                    best_pos = pos

                        if best_k >= 0 and best_delta < -1e-6:
                            # Apply move
                            new_routes = [list(r) for r in routes]
                            new_routes[i] = rest_i
                            new_routes[best_k] = new_routes[best_k][:best_pos] + seg + new_routes[best_k][best_pos:]
                            new_loads = list(route_loads)
                            new_loads[i] = route_loads[i] - seg_load
                            new_loads[best_k] = route_loads[best_k] + seg_load
                            if new_loads[best_k] > capacity:
                                continue
                            routes = new_routes
                            route_loads = new_loads
                            cur_dist += best_delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes

    # --- Local Search: Cross-route swap ---
    def cross_swap(routes):
        routes = [list(r) for r in routes]
        route_loads = [sum(demands[v] for v in r) for r in routes]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur_dist = total_distance(routes)
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
                            if time.time() > deadline:
                                continue
                            vi = routes[i][a]
                            vj = routes[j][b]
                            if vi == vj:
                                continue
                            # Delta distance
                            old_d = 0.0
                            if a > 0: old_d += dist[routes[i][a-1], vi]
                            else: old_d += dist[0, vi]
                            if a < len(routes[i])-1: old_d += dist[vi, routes[i][a+1]]
                            else: old_d += dist[vi, 0]
                            if b > 0: old_d += dist[routes[j][b-1], vj]
                            else: old_d += dist[0, vj]
                            if b < len(routes[j])-1: old_d += dist[vj, routes[j][b+1]]
                            else: old_d += dist[vj, 0]

                            new_d = 0.0
                            if a > 0: new_d += dist[routes[i][a-1], vj]
                            else: new_d += dist[0, vj]
                            if a < len(routes[i])-1: new_d += dist[vj, routes[i][a+1]]
                            else: new_d += dist[vj, 0]
                            if b > 0: new_d += dist[routes[j][b-1], vi]
                            else: new_d += dist[0, vi]
                            if b < len(routes[j])-1: new_d += dist[vi, routes[j][b+1]]
                            else: new_d += dist[vi, 0]

                            if new_d >= old_d - 1e-6:
                                continue

                            nl_i = route_loads[i] - demands[vi] + demands[vj]
                            nl_j = route_loads[j] - demands[vj] + demands[vi]
                            if nl_i > capacity or nl_j > capacity:
                                continue

                            new_routes = [list(r) for r in routes]
                            new_routes[i][a] = vj
                            new_routes[j][b] = vi
                            f1, _ = check_route_fast(new_routes[i], nl_i)
                            if not f1:
                                continue
                            f2, _ = check_route_fast(new_routes[j], nl_j)
                            if not f2:
                                continue

                            routes = new_routes
                            route_loads = list(route_loads)
                            route_loads[i] = nl_i
                            route_loads[j] = nl_j
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # --- LNS (Large Neighborhood Search) with SA acceptance ---
    def lns(routes, n_iters=200):
        routes = [list(r) for r in routes]
        cur_dist = total_distance(routes)
        best_routes = [list(r) for r in routes]
        best_dist = cur_dist
        T = 5.0
        T_min = 0.01
        cooling = (T_min / T) ** (1.0 / max(n_iters, 1))

        for it in range(n_iters):
            if time.time() > deadline:
                break
            T *= cooling

            # Destroy: remove k random customers
            k = random.randint(3, 7)
            all_custs = [v for r in routes for v in r]
            removed = random.sample(all_custs, min(k, len(all_custs)))
            removed_set = set(removed)

            # Rebuild routes without removed customers
            new_routes = []
            for r in routes:
                nr = [v for v in r if v not in removed_set]
                if nr:
                    new_routes.append(nr)

            # Reinsert removed customers one by one
            random.shuffle(removed)
            for v in removed:
                best_pos = None
                best_delta = float('inf')
                for ri, r in enumerate(new_routes):
                    for pos in range(len(r) + 1):
                        cand = r[:pos] + [v] + r[pos:]
                        load = sum(demands[x] for x in cand)
                        if load > capacity:
                            continue
                        f, _ = check_route_fast(cand, load)
                        if not f:
                            continue
                        # Delta distance
                        old_d = 0.0
                        if pos > 0: old_d += dist[r[pos-1], r[pos]] if pos < len(r) else dist[r[pos-1], 0]
                        if pos < len(r): old_d += dist[r[pos], r[pos+1]] if pos+1 < len(r) else dist[r[pos], 0]
                        new_d = 0.0
                        if pos > 0: new_d += dist[r[pos-1], v]
                        if pos < len(r): new_d += dist[v, r[pos]]
                        delta = new_d - old_d
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = (ri, pos)

                if best_pos is None:
                    # Try single-route
                    new_routes.append([v])
                else:
                    ri, pos = best_pos
                    new_routes[ri] = new_routes[ri][:pos] + [v] + new_routes[ri][pos:]

            # Validate
            if not validate(new_routes):
                continue

            new_dist = total_distance(new_routes)
            delta = new_dist - cur_dist
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
                routes = new_routes
                cur_dist = new_dist
                if cur_dist < best_dist:
                    best_dist = cur_dist
                    best_routes = [list(r) for r in routes]

        return best_routes

    # === Main algorithm ===
    # 1. Generate initial solutions
    best_routes = None
    best_dist = float('inf')

    orderings = [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: dist[0, v]),
        sorted(customers, key=lambda v: (due[v], dist[0, v])),
        sorted(customers, key=lambda v: (ready[v], dist[0, v])),
        sorted(customers, key=lambda v: dist[0, v] + dist[v, 0]),
    ]

    for ordering in orderings:
        if time.time() > deadline:
            break
        gr = greedy_construct(ordering)
        if validate(gr):
            d = total_distance(gr)
            if d < best_dist:
                best_dist = d
                best_routes = [list(r) for r in gr]

    # Savings
    if time.time() < deadline:
        sr = savings_construct()
        if validate(sr):
            d = total_distance(sr)
            if d < best_dist:
                best_dist = d
                best_routes = [list(r) for r in sr]

    if best_routes is None:
        # Fallback: one customer per route
        best_routes = [[v] for v in customers]
        best_dist = total_distance(best_routes)

    # 2. Local search polish
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=4)

    if time.time() < deadline:
        best_routes = cross_swap(best_routes)

    # 3. LNS with SA
    if time.time() < deadline:
        remaining_time = deadline - time.time()
        n_lns = max(50, int(remaining_time * 10))  # ~10 LNS iterations per second
        best_routes = lns(best_routes, n_iters=n_lns)

    # 4. Final polish
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=3)

    if time.time() < deadline:
        best_routes = cross_swap(best_routes)

    # 5. One more LNS pass if time remains
    if time.time() < deadline - 10:
        remaining_time = deadline - time.time() - 10
        n_lns = max(30, int(remaining_time * 8))
        best_routes = lns(best_routes, n_iters=n_lns)

    # Final 2-opt
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    best_routes = [r for r in best_routes if r]
    return best_routes
# EVOLVE-BLOCK-END

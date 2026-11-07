"""Savings-based construction + aggressive local search for EV routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import heapq


def plan(instance):
    start = time.time()
    deadline = start + 140.0

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(d2)

    customers = list(range(1, n_customers + 1))

    # ---------- Simulation with charging ----------
    def simulate_with_charge(route, allow_charge=True):
        """Simulate route, auto-inserting charges at stations if needed.
        Returns (ok, total_dist, end_time, final_route_with_stations)."""
        if not route:
            return True, 0.0, 0.0, []
        load = float(demands[route].sum())
        if load > capacity + 1e-9:
            return False, 0.0, 0.0, []

        t = 0.0
        battery = battery_cap
        total = 0.0
        prev = 0
        final_route = []

        for v in route:
            d = dist[prev, v]
            if d > battery + 1e-9:
                if not allow_charge:
                    return False, 0.0, 0.0, []
                best_station = None
                best_extra = float('inf')
                for st in stations:
                    d_to_st = dist[prev, st]
                    d_st_to_v = dist[st, v]
                    extra = d_to_st + d_st_to_v - d
                    if extra < best_extra and battery_cap >= d_st_to_v + 1e-9:
                        best_extra = extra
                        best_station = st
                if best_station is None:
                    return False, 0.0, 0.0, []
                st = best_station
                d_to_st = dist[prev, st]
                battery -= d_to_st
                t += d_to_st / speed
                total += d_to_st
                charge_time = (battery_cap - battery) * recharge_rate
                t += charge_time
                battery = battery_cap
                prev = st
                final_route.append(st)
                d_st_to_v = dist[st, v]
                battery -= d_st_to_v
                t += d_st_to_v / speed
                total += d_st_to_v
                prev = v
            else:
                battery -= d
                t += d / speed
                total += d
                prev = v

            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0, []
            t += service[v]
            final_route.append(v)

        d = dist[prev, 0]
        if d > battery + 1e-9:
            if allow_charge:
                best_station = None
                best_extra = float('inf')
                for st in stations:
                    d_to_st = dist[prev, st]
                    d_st_to_0 = dist[st, 0]
                    extra = d_to_st + d_st_to_0 - d
                    if extra < best_extra and battery_cap >= d_st_to_0 + 1e-9:
                        best_extra = extra
                        best_station = st
                if best_station is None:
                    return False, 0.0, 0.0, []
                st = best_station
                d_to_st = dist[prev, st]
                battery -= d_to_st
                t += d_to_st / speed
                total += d_to_st
                charge_time = (battery_cap - battery) * recharge_rate
                t += charge_time
                battery = battery_cap
                prev = st
                final_route.append(st)
                d_st_to_0 = dist[st, 0]
                battery -= d_st_to_0
                t += d_st_to_0 / speed
                total += d_st_to_0
                prev = 0
            else:
                return False, 0.0, 0.0, []
        else:
            battery -= d
            t += d / speed
            total += d
            prev = 0

        if t > horizon + 1e-9:
            return False, 0.0, 0.0, []
        return True, total, t, final_route

    def simulate(route):
        """Quick feasibility + distance check."""
        if not route:
            return True, 0.0, 0.0
        load = float(demands[route].sum())
        if load > capacity + 1e-9:
            return False, 0.0, 0.0
        t = 0.0
        battery = battery_cap
        total = 0.0
        prev = 0
        for v in route:
            d = dist[prev, v]
            if d > battery + 1e-9:
                return False, 0.0, 0.0
            battery -= d
            t += d / speed
            total += d
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0
            t += service[v]
            prev = v
        d = dist[prev, 0]
        if d > battery + 1e-9:
            return False, 0.0, 0.0
        t += d / speed
        if t > horizon + 1e-9:
            return False, 0.0, 0.0
        return True, total, t

    def route_dist(route):
        if not route:
            return 0.0
        t = dist[0, route[0]]
        for i in range(len(route) - 1):
            t += dist[route[i], route[i + 1]]
        t += dist[route[-1], 0]
        return t

    # ---------- Savings-based construction ----------
    def savings_construct():
        """Clarke-Wright savings heuristic with TW/battery feasibility."""
        # Start with each customer as its own route
        routes = [[v] for v in customers]
        route_loads = [float(demands[v]) for v in customers]
        # Map: customer -> route index
        cust_to_route = {v: i for i, v in enumerate(customers)}

        # Compute savings: saving(i, j) = dist(0,i) + dist(0,j) - dist(i,j)
        # Only valid if i is last in its route and j is first in its route
        # For simplicity, compute all pairwise savings
        savings = []
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                ci = i + 1  # customer node id
                cj = j + 1
                s = dist[0, ci] + dist[0, cj] - dist[ci, cj]
                savings.append((s, ci, cj))

        # Sort by savings descending
        savings.sort(key=lambda x: -x[0])

        for s, ci, cj in savings:
            if time.time() > deadline - 30:
                break
            if s < -1e-9:
                break
            ri = cust_to_route.get(ci)
            rj = cust_to_route.get(cj)
            if ri is None or rj is None or ri == rj:
                continue
            # Check if ci is at the end of route ri or cj is at the start
            route_i = routes[ri]
            route_j = routes[rj]
            if not route_i or not route_j:
                continue

            # Try merging: ci must be last in ri, cj must be first in rj
            # OR cj must be last in rj, ci must be first in ri
            merged = None
            if route_i[-1] == ci and route_j[0] == cj:
                # Merge ri + rj
                cand = route_i + route_j
                load = route_loads[ri] + route_loads[rj]
                if load <= capacity + 1e-9:
                    ok, _, _ = simulate(cand)
                    if ok:
                        merged = cand
            elif route_j[-1] == cj and route_i[0] == ci:
                # Merge rj + ri
                cand = route_j + route_i
                load = route_loads[ri] + route_loads[rj]
                if load <= capacity + 1e-9:
                    ok, _, _ = simulate(cand)
                    if ok:
                        merged = cand
            elif route_i[-1] == cj and route_j[0] == ci:
                # Merge ri + reversed rj
                cand = route_i + route_j[::-1]
                load = route_loads[ri] + route_loads[rj]
                if load <= capacity + 1e-9:
                    ok, _, _ = simulate(cand)
                    if ok:
                        merged = cand
            elif route_j[-1] == ci and route_i[0] == cj:
                # Merge rj + reversed ri
                cand = route_j + route_i[::-1]
                load = route_loads[ri] + route_loads[rj]
                if load <= capacity + 1e-9:
                    ok, _, _ = simulate(cand)
                    if ok:
                        merged = cand

            if merged is not None:
                # Remove route_j, update route_i
                routes[ri] = merged
                route_loads[ri] += route_loads[rj]
                # Update cust_to_route for all in merged
                for v in merged:
                    cust_to_route[v] = ri
                # Mark route_j as empty
                routes[rj] = []
                route_loads[rj] = 0.0

        # Filter empty routes
        routes = [r for r in routes if r]
        return routes

    # ---------- Local search operators ----------
    def two_opt(route, max_iter=200):
        if len(route) < 3:
            return route
        best = list(route)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            L = len(best)
            for i in range(L - 1):
                if time.time() > deadline - 10:
                    return best
                a = best[i - 1] if i > 0 else 0
                b = best[i]
                for j in range(i + 1, L):
                    c = best[j]
                    d = best[j + 1] if j + 1 < L else 0
                    old = dist[a, b] + dist[c, d]
                    new = dist[a, c] + dist[b, d]
                    if new < old - 1e-9:
                        cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                        ok, tot, _ = simulate(cand)
                        if ok:
                            best = cand
                            improved = True
                            break
                if improved:
                    break
        return best

    def or_opt(route, max_iter=100):
        if len(route) < 3:
            return route
        best = list(route)
        best_dist = route_dist(best)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            L = len(best)
            for seg_len in (1, 2):
                if time.time() > deadline - 10:
                    return best
                for i in range(L - seg_len + 1):
                    seg = best[i:i + seg_len]
                    rest = best[:i] + best[i + seg_len:]
                    if not rest:
                        continue
                    for pos in range(len(rest) + 1):
                        cand = rest[:pos] + seg + rest[pos:]
                        if cand == best:
                            continue
                        ok, tot, _ = simulate(cand)
                        if ok and tot < best_dist - 1e-9:
                            best = cand
                            best_dist = tot
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best

    # ---------- Inter-route: relocate ----------
    def relocate_opt(routes, max_iter=100):
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n_routes = len(routes)
            for r1 in range(n_routes):
                if time.time() > deadline - 15:
                    return routes
                for r2 in range(n_routes):
                    if r1 == r2:
                        continue
                    route1 = routes[r1]
                    route2 = routes[r2]
                    if not route1 or not route2:
                        continue
                    for i in range(len(route1)):
                        v = route1[i]
                        load2 = float(demands[route2].sum()) + demands[v]
                        if load2 > capacity + 1e-9:
                            continue
                        new_route1 = route1[:i] + route1[i+1:]
                        best_pos = -1
                        best_cost = float('inf')
                        L2 = len(route2)
                        for pos in range(L2 + 1):
                            cand = route2[:pos] + [v] + route2[pos:]
                            ok, tot, _ = simulate(cand)
                            if ok and tot < best_cost:
                                best_cost = tot
                                best_pos = pos
                        if best_pos >= 0:
                            new_route2 = route2[:best_pos] + [v] + route2[best_pos:]
                            ok1, d1, _ = simulate(new_route1)
                            if ok1:
                                old_total = route_dist(route1) + route_dist(route2)
                                new_total = d1 + best_cost
                                if new_total < old_total - 1e-9:
                                    routes[r1] = new_route1
                                    routes[r2] = new_route2
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # ---------- Inter-route: swap ----------
    def swap_opt(routes, max_iter=50):
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n_routes = len(routes)
            for r1 in range(n_routes):
                if time.time() > deadline - 15:
                    return routes
                for r2 in range(r1 + 1, n_routes):
                    route1 = routes[r1]
                    route2 = routes[r2]
                    if not route1 or not route2:
                        continue
                    for i in range(len(route1)):
                        v1 = route1[i]
                        for j in range(len(route2)):
                            v2 = route2[j]
                            load1 = float(demands[route1].sum()) - demands[v1] + demands[v2]
                            load2 = float(demands[route2].sum()) - demands[v2] + demands[v1]
                            if load1 > capacity + 1e-9 or load2 > capacity + 1e-9:
                                continue
                            new_route1 = route1[:i] + [v2] + route1[i+1:]
                            new_route2 = route2[:j] + [v1] + route2[j+1:]
                            ok1, d1, _ = simulate(new_route1)
                            ok2, d2, _ = simulate(new_route2)
                            if ok1 and ok2:
                                old_total = route_dist(route1) + route_dist(route2)
                                new_total = d1 + d2
                                if new_total < old_total - 1e-9:
                                    routes[r1] = new_route1
                                    routes[r2] = new_route2
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # ---------- Main ----------
    # Phase 1: Savings construction
    routes = savings_construct()

    # Phase 2: Intra-route local search
    for r in range(len(routes)):
        if time.time() > deadline - 20:
            break
        routes[r] = two_opt(routes[r])
        routes[r] = or_opt(routes[r])

    # Phase 3: Inter-route optimization
    if time.time() < deadline - 20:
        routes = relocate_opt(routes)
    if time.time() < deadline - 15:
        routes = swap_opt(routes)

    # Phase 4: Second pass local search
    for r in range(len(routes)):
        if time.time() > deadline - 5:
            break
        routes[r] = two_opt(routes[r], max_iter=100)
        routes[r] = or_opt(routes[r], max_iter=50)

    # Phase 5: Final inter-route
    if time.time() < deadline - 8:
        routes = relocate_opt(routes, max_iter=50)

    # Ensure all customers are served
    served = set()
    for r in routes:
        for v in r:
            served.add(v)
    missing = [v for v in customers if v not in served]
    for v in missing:
        routes.append([v])

    return routes
# EVOLVE-BLOCK-END

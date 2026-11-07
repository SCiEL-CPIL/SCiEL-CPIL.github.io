"""EV routing: Clarke-Wright savings + aggressive local search with battery-aware feasibility."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import heapq
import copy


def plan(instance):
    start = time.time()
    deadline = start + 145.0

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
    n_cust = len(customers)

    # ---------- Fast battery-aware simulation ----------
    def simulate(route):
        """Check if route is feasible (capacity, battery, time windows, horizon). Returns (ok, total_dist)."""
        if not route:
            return True, 0.0
        load = sum(demands[v] for v in route)
        if load > capacity + 1e-9:
            return False, 0.0

        t = 0.0
        battery = battery_cap
        total = 0.0
        prev = 0
        dist_local = dist
        stations_local = stations
        battery_cap_local = battery_cap
        speed_local = speed
        recharge_rate_local = recharge_rate
        ready_local = ready
        due_local = due
        service_local = service

        for v in route:
            d = dist_local[prev, v]
            if d > battery + 1e-9:
                # Find best station to minimize detour
                best_st = -1
                best_detour = float('inf')
                best_charge_time = 0.0
                for st in stations_local:
                    d_to_st = dist_local[prev, st]
                    if d_to_st <= battery + 1e-9:
                        d_st_to_v = dist_local[st, v]
                        if d_st_to_v <= battery_cap_local + 1e-9: # Can reach v from st
                            # Calculate detour cost: d_to_st + d_st_to_v - d
                            detour = d_to_st + d_st_to_v - d
                            if detour < best_detour:
                                # Calculate charge time if needed
                                # Battery at st arrival: battery - d_to_st
                                # Battery needed for d_st_to_v: d_st_to_v
                                # If (battery - d_to_st) < d_st_to_v, charge time is (d_st_to_v - (battery - d_to_st)) / recharge_rate
                                # Wait, the logic in original was:
                                # battery -= d_to_st
                                # if battery < d_st_to_v: charge_time = (d_st_to_v - battery) / recharge_rate; battery = battery_cap
                                # This implies we charge enough to reach v? Or full?
                                # "a station adds (battery - energy) * recharge_rate time and refills" -> Refills to full.
                                # So charge_time = (battery_cap - (battery - d_to_st)) / recharge_rate?
                                # No, "refills" means battery becomes battery_cap.
                                # Time = (battery_cap - current_battery) / recharge_rate.
                                # Current battery at station = battery - d_to_st.
                                # So time = (battery_cap - (battery - d_to_st)) / recharge_rate.
                                # But we only *need* to charge enough to reach v?
                                # The rule says "refills". So it always fills to full.
                                # So time is fixed for a given arrival battery.
                                # Let's stick to the original logic but optimize selection.
                                pass
                        if d_st_to_v > battery + 1e-9:
                             # Need to charge.
                             # Arrival battery: battery - d_to_st
                             # Charge to full: time = (battery_cap - (battery - d_to_st)) / recharge_rate
                             # Then travel d_st_to_v.
                             # Is it feasible? Yes, if we can reach st.
                             pass
                        # Simplified: Just check feasibility and detour.
                        # Feasibility: d_to_st <= battery.
                        # After charge, battery is full. d_st_to_v <= battery_cap.
                        # So if d_to_st <= battery, it's feasible.
                        # We want min detour.
                        if d_to_st <= battery + 1e-9:
                            detour = d_to_st + d_st_to_v - d
                            if detour < best_detour:
                                best_detour = detour
                                best_st = st
                                # Pre-calculate time contribution for this station choice
                                # t += d_to_st/speed
                                # charge_time = (battery_cap - (battery - d_to_st)) / recharge_rate
                                # t += charge_time
                                # t += d_st_to_v/speed
                                pass
                if best_st == -1:
                    return False, 0.0
                st = best_st
                d_to_st = dist_local[prev, st]
                d_st_to_v = dist_local[st, v]
                battery -= d_to_st
                t += d_to_st / speed_local
                total += d_to_st
                # Charge to full
                charge_time = (battery_cap_local - battery) / recharge_rate_local
                t += charge_time
                battery = battery_cap_local
                # Travel to v
                if d_st_to_v > battery + 1e-9:
                    return False, 0.0
                battery -= d_st_to_v
                t += d_st_to_v / speed_local
                total += d_st_to_v
            else:
                battery -= d
                t += d / speed_local
                total += d

            if t < ready_local[v]:
                t = ready_local[v]
            if t > due_local[v] + 1e-9:
                return False, 0.0
            t += service_local[v]
            prev = v

        # Return to depot
        d = dist_local[prev, 0]
        if d > battery + 1e-9:
            best_st = -1
            best_detour = float('inf')
            for st in stations_local:
                d_to_st = dist_local[prev, st]
                if d_to_st <= battery + 1e-9:
                    d_st_to_0 = dist_local[st, 0]
                    if d_st_to_0 <= battery_cap_local + 1e-9:
                        detour = d_to_st + d_st_to_0 - d
                        if detour < best_detour:
                            best_detour = detour
                            best_st = st
            if best_st == -1:
                return False, 0.0
            st = best_st
            d_to_st = dist_local[prev, st]
            d_st_to_0 = dist_local[st, 0]
            battery -= d_to_st
            t += d_to_st / speed_local
            total += d_to_st
            charge_time = (battery_cap_local - battery) / recharge_rate_local
            t += charge_time
            battery = battery_cap_local
            if d_st_to_0 > battery + 1e-9:
                return False, 0.0
            battery -= d_st_to_0
            t += d_st_to_0 / speed_local
            total += d_st_to_0
        else:
            battery -= d
            t += d / speed_local
            total += d

        if t > horizon + 1e-9:
            return False, 0.0
        return True, total

    # ---------- Route distance (for optimization, ignoring charging detours) ----------
    def route_dist(route):
        if not route:
            return 0.0
        t = dist[0, route[0]]
        for i in range(len(route) - 1):
            t += dist[route[i], route[i + 1]]
        t += dist[route[-1], 0]
        return t

    def total_distance(routes):
        return sum(route_dist(r) for r in routes)

    # ---------- Clarke-Wright Savings Construction ----------
    def cw_construct():
        """Build routes using Clarke-Wright savings heuristic."""
        # Start with each customer in its own route
        routes = [[c] for c in customers]
        route_of = {c: i for i, c in enumerate(customers)}
        
        # Precompute savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
        savings = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                s = dist[0, ci] + dist[0, cj] - dist[ci, cj]
                savings.append((s, ci, cj))
        
        # Sort by savings descending
        savings.sort(reverse=True)
        
        for s, ci, cj in savings:
            if time.time() > deadline - 30.0:
                break
            if s <= 0:
                break
            ri = route_of[ci]
            rj = route_of[cj]
            if ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]
            
            # Check capacity
            load_i = sum(demands[v] for v in route_i)
            load_j = sum(demands[v] for v in route_j)
            if load_i + load_j > capacity + 1e-9:
                continue
            
            # Check if ci and cj are at ends of their routes
            ci_at_end = (ci == route_i[0] or ci == route_i[-1])
            cj_at_end = (cj == route_j[0] or cj == route_j[-1])
            if not ci_at_end or not cj_at_end:
                continue
            
            # Determine merged route
            if ci == route_i[0] and cj == route_j[0]:
                new_route = route_i + route_j
            elif ci == route_i[-1] and cj == route_j[0]:
                new_route = route_i + route_j
            elif ci == route_i[0] and cj == route_j[-1]:
                new_route = route_i + route_j[::-1]
            else:
                new_route = route_i + route_j[::-1]
            
            # Check feasibility
            ok, _ = simulate(new_route)
            if ok:
                routes[ri] = new_route
                routes[rj] = []
                for v in new_route:
                    route_of[v] = ri
        
        # Remove empty routes
        routes = [r for r in routes if r]
        return routes

    # ---------- 2-opt ----------
    def two_opt(route, max_iter=500):
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
            for i in range(L - 1):
                a = best[i - 1] if i > 0 else 0
                b = best[i]
                for j in range(i + 1, L):
                    c = best[j]
                    d_node = best[j + 1] if j + 1 < L else 0
                    old = dist[a, b] + dist[c, d_node]
                    new = dist[a, c] + dist[b, d_node]
                    if new < old - 1e-9:
                        cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                        ok, _ = simulate(cand)
                        if ok and route_dist(cand) < best_dist - 1e-9:
                            best = cand
                            best_dist = route_dist(cand)
                            improved = True
                            break
                if improved:
                    break
        return best

    # ---------- Or-opt ----------
    def or_opt(route, max_iter=300):
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
                for i in range(L - seg_len + 1):
                    seg = best[i:i + seg_len]
                    rest = best[:i] + best[i + seg_len:]
                    if not rest:
                        continue
                    for pos in range(len(rest) + 1):
                        cand = rest[:pos] + seg + rest[pos:]
                        if cand == best:
                            continue
                        ok, _ = simulate(cand)
                        if ok and route_dist(cand) < best_dist - 1e-9:
                            best = cand
                            best_dist = route_dist(cand)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best

    # ---------- Relocate between routes ----------
    def relocate_opt(routes, max_iter=50):
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n_routes = len(routes)
            for r1 in range(n_routes):
                if len(routes[r1]) == 0:
                    continue
                for r2 in range(n_routes):
                    if r1 == r2:
                        continue
                    route1 = list(routes[r1])
                    route2 = list(routes[r2])
                    for i in range(len(route1)):
                        v = route1[i]
                        load2 = sum(demands[x] for x in route2)
                        if load2 + demands[v] > capacity + 1e-9:
                            continue
                        new_route1 = route1[:i] + route1[i+1:]
                        best_pos = -1
                        best_cost = float('inf')
                        for pos in range(len(route2) + 1):
                            cand = route2[:pos] + [v] + route2[pos:]
                            ok, _ = simulate(cand)
                            if ok:
                                cost = route_dist(cand)
                                if cost < best_cost:
                                    best_cost = cost
                                    best_pos = pos
                        if best_pos >= 0:
                            new_route2 = route2[:best_pos] + [v] + route2[best_pos:]
                            ok1, _ = simulate(new_route1)
                            if ok1:
                                old_total = route_dist(route1) + route_dist(route2)
                                new_total = route_dist(new_route1) + best_cost
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

    # ---------- Swap between routes ----------
    def swap_opt(routes, max_iter=30):
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n_routes = len(routes)
            for r1 in range(n_routes):
                for r2 in range(r1 + 1, n_routes):
                    route1 = list(routes[r1])
                    route2 = list(routes[r2])
                    for i in range(len(route1)):
                        v1 = route1[i]
                        for j in range(len(route2)):
                            v2 = route2[j]
                            load1 = sum(demands[x] for x in route1) - demands[v1] + demands[v2]
                            load2 = sum(demands[x] for x in route2) - demands[v2] + demands[v1]
                            if load1 > capacity + 1e-9 or load2 > capacity + 1e-9:
                                continue
                            new_route1 = route1[:i] + [v2] + route1[i+1:]
                            new_route2 = route2[:j] + [v1] + route2[j+1:]
                            ok1, _ = simulate(new_route1)
                            ok2, _ = simulate(new_route2)
                            if ok1 and ok2:
                                old_total = route_dist(route1) + route_dist(route2)
                                new_total = route_dist(new_route1) + route_dist(new_route2)
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
    # Phase 1: Clarke-Wright construction
    routes = cw_construct()

    # Phase 2: Intensive Local Search Loop
    # We alternate between intra-route and inter-route moves until no improvement or time out.
    
    def try_two_opt_all():
        improved = False
        for i in range(len(routes)):
            if time.time() > deadline - 5.0:
                return False
            new_route = two_opt(routes[i])
            if route_dist(new_route) < route_dist(routes[i]) - 1e-9:
                routes[i] = new_route
                improved = True
        return improved

    def try_or_opt_all():
        improved = False
        for i in range(len(routes)):
            if time.time() > deadline - 5.0:
                return False
            new_route = or_opt(routes[i])
            if route_dist(new_route) < route_dist(routes[i]) - 1e-9:
                routes[i] = new_route
                improved = True
        return improved

    def try_relocate_all():
        if time.time() > deadline - 5.0:
            return False
        old_dist = total_distance(routes)
        relocate_opt(routes)
        new_dist = total_distance(routes)
        if new_dist < old_dist - 1e-9:
            return True
        return False

    def try_swap_all():
        if time.time() > deadline - 5.0:
            return False
        old_dist = total_distance(routes)
        swap_opt(routes)
        new_dist = total_distance(routes)
        if new_dist < old_dist - 1e-9:
            return True
        return False

    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() < deadline - 10.0:
        improved = False
        
        # Intra-route moves
        if try_two_opt_all():
            improved = True
        if try_or_opt_all():
            improved = True
            
        # Inter-route moves
        if try_relocate_all():
            improved = True
        if try_swap_all():
            improved = True

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break

    # Final polish with tighter time check
    if time.time() < deadline - 2.0:
        for r in range(len(routes)):
            if time.time() > deadline - 1.0:
                break
            routes[r] = two_opt(routes[r])
            routes[r] = or_opt(routes[r])

    # Ensure all customers served
    if not routes:
        routes = [[v] for v in customers]
    else:
        served = set()
        for r in routes:
            for v in r:
                served.add(v)
        missing = [v for v in customers if v not in served]
        for v in missing:
            routes.append([v])

    return routes
# EVOLVE-BLOCK-END

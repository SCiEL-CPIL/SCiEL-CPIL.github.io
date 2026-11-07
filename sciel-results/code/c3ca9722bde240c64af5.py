# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin vs 150s wall-clock kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = int(instance["n_customers"])
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n_nodes = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))

    customers = list(range(1, n_customers + 1))
    cust_set = set(customers)
    station_set = set(stations)

    # Pre-compute station distances for quick detour calculation
    # We need dist(prev, st) + dist(st, next) - dist(prev, next)
    
    # ---- Core simulation (depot implicit at both ends) ----
    def simulate(route):
        t = 0.0
        bat = battery_cap
        cd = 0.0
        cur = 0
        for node in route:
            d = dist[cur, node]
            t += d / speed
            bat -= d
            cd += d
            if bat < -1e-9:
                return None
            if node in cust_set:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return None
                t += service[node]
            elif node in station_set:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            cur = node
        d = dist[cur, 0]
        t += d / speed
        bat -= d
        cd += d
        if bat < -1e-9 or t > horizon + 1e-9:
            return None
        return (cd, t, bat)

    def route_demand(route):
        s = 0.0
        for v in route:
            if v in cust_set:
                s += demands[v]
        return s

    # ---- Battery repair: Insert the cheapest single station if needed ----
    def best_station_fix(route):
        s = simulate(route)
        if s is not None:
            return list(route), s[0]
        
        # Find the index where battery fails
        # We need to re-simulate to find the exact failure point
        t = 0.0
        bat = battery_cap
        cur = 0
        fail_idx = -1
        for i, node in enumerate(route):
            d = dist[cur, node]
            t += d / speed
            bat -= d
            if bat < -1e-9:
                fail_idx = i
                break
            if node in cust_set:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    # Time window violation, station won't fix this unless it's before?
                    # Actually, if time is violated, inserting a station adds time, so it makes it worse.
                    # So if time is violated, we can't fix it with a station.
                    return None, None
                t += service[node]
            elif node in station_set:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            cur = node
            
        # If we didn't break on battery, check final return leg
        if fail_idx == -1:
            d = dist[cur, 0]
            bat -= d
            if bat < -1e-9:
                fail_idx = len(route) # Failure on return leg
            else:
                # Check time
                t += d / speed
                if t > horizon + 1e-9:
                    return None, None # Time violation on return, station makes it worse
                return None, None # Should not happen if simulate returned None but we didn't catch battery?
        
        # We need to insert a station at some position p <= fail_idx
        # Try positions from fail_idx down to 0
        # For each position, find the station that minimizes detour
        best_route = None
        best_d = None
        
        # We iterate p from fail_idx down to 0
        # If p == fail_idx, we insert before route[fail_idx]
        # If p == 0, we insert at the start
        
        for p in range(fail_idx + 1):
            prev_node = route[p-1] if p > 0 else 0
            next_node = route[p]
            
            # Find best station for this gap
            min_detour = float('inf')
            best_st = None
            for st in stations:
                # Detour = dist(prev, st) + dist(st, next) - dist(prev, next)
                det = dist[prev_node, st] + dist[st, next_node] - dist[prev_node, next_node]
                if det < min_detour:
                    min_detour = det
                    best_st = st
            
            if best_st is None:
                continue
                
            # Try this candidate
            cand = list(route[:p]) + [best_st] + list(route[p:])
            # Check demand (stations don't add demand, so same as route_demand(route))
            # But wait, if we already checked route_demand(route) <= capacity, it's still valid.
            
            sc = simulate(cand)
            if sc is not None:
                if best_d is None or sc[0] < best_d:
                    best_route, best_d = cand, sc[0]
                # Since we are going backwards from failure point, the first valid one is likely good.
                # But a station earlier might be better?
                # Usually inserting closer to the failure is better to minimize time/battery waste.
                # We can break here to save time, as inserting further back usually increases detour.
                break
        
        if best_route is not None:
            return best_route, best_d
            
        # If that didn't work, try other stations or positions?
        # For now, return None
        return None, None

    def total_dist(routes):
        tot = 0.0
        for r in routes:
            s = simulate(r)
            if s is None:
                return float("inf")
            tot += s[0]
        return tot

    # ---- Construction: greedy nearest-insertion over a given order ----
    def greedy_construct(order):
        routes = []
        for c in order:
            placed = False
            best = None  # (route_idx, fixed_route, dist)
            for ri, r in enumerate(routes):
                if route_demand(r) + demands[c] > capacity:
                    continue
                for pos in range(len(r) + 1):
                    cand = list(r[:pos]) + [c] + list(r[pos:])
                    fixed, cd = best_station_fix(cand)
                    if fixed is None:
                        continue
                    if best is None or cd < best[2]:
                        best = (ri, fixed, cd)
            if best is not None:
                routes[best[0]] = best[1]
                placed = True
            if not placed:
                fixed, cd = best_station_fix([c])
                if fixed is None:
                    fixed = [c]  # single customer is always battery-feasible
                routes.append(fixed)
        return routes

    # ---- Local-search operators (all require re-simulation) ----
    def two_opt(route, max_iter=80):
        if len(route) < 2:
            return route
        cur = simulate(route)
        if cur is None:
            return route
        cur_d = cur[0]
        improved = True
        it = 0
        n = len(route)
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Analytical distance check for 2-opt
                    # Old: dist[route[i-1], route[i]] + dist[route[j], route[j+1]] (approx, handling boundaries)
                    # New: dist[route[i-1], route[j]] + dist[route[i], route[j+1]]
                    # This is an approximation because we don't know the exact indices of neighbors in the list
                    # route is a list.
                    # i, j are indices in route.
                    # Neighbors:
                    # i-1, i, i+1 ... j-1, j, j+1
                    # If i=0, prev is depot.
                    # If j+1 == n, next is depot.
                    
                    prev_i = route[i-1] if i > 0 else 0
                    next_i = route[i+1] if i < n-1 else 0 # Wait, 2-opt reverses i..j
                    # The segment is route[i] ... route[j]
                    # Original edges: (prev_i, route[i]) and (route[j], next_j)
                    # New edges: (prev_i, route[j]) and (route[i], next_j)
                    next_j = route[j+1] if j < n-1 else 0
                    
                    d_old = dist[prev_i, route[i]] + dist[route[j], next_j]
                    d_new = dist[prev_i, route[j]] + dist[route[i], next_j]
                    
                    if d_new >= d_old - 1e-7:
                        continue # No distance improvement, skip simulation
                    
                    nr = list(route[:i]) + list(reversed(route[i:j + 1])) + list(route[j + 1:])
                    if route_demand(nr) > capacity:
                        continue
                    sn = simulate(nr)
                    if sn is None:
                        continue
                    if sn[0] < cur_d - 1e-7:
                        route, cur_d = nr, sn[0]
                        improved = True
        return route

    def relocate(routes, blocklen=1):
        nR = len(routes)
        oldD = [(simulate(r)[0] if r else 0.0) for r in routes]
        for i in range(nR):
            ri_full = routes[i]
            old_di = oldD[i]
            for pi in range(len(ri_full) - blocklen + 1):
                block = ri_full[pi:pi + blocklen]
                block_demand = 0.0
                for v in block:
                    if v in cust_set:
                        block_demand += demands[v]
                rest_i = ri_full[:pi] + ri_full[pi + blocklen:]
                sim_i = simulate(rest_i) if rest_i else (0.0, 0.0, battery_cap)
                if sim_i is None:
                    continue
                di = sim_i[0] if rest_i else 0.0
                for j in range(nR):
                    if i == j:
                        continue
                    old_dj = oldD[j]
                    if route_demand(routes[j]) + block_demand > capacity:
                        continue
                    rj_base = routes[j]
                    for pj in range(len(rj_base) + 1):
                        rj = rj_base[:pj] + list(block) + rj_base[pj:]
                        sim_j = simulate(rj)
                        if sim_j is None:
                            continue
                        if di + sim_j[0] < old_di + old_dj - 1e-7:
                            routes[i] = rest_i
                            routes[j] = rj
                            routes = [r for r in routes if r]
                            return routes, True
        return routes, False

    def swap(routes):
        nR = len(routes)
        oldD = [(simulate(r)[0] if r else 0.0) for r in routes]
        for i in range(nR):
            old_di = oldD[i]
            for j in range(i + 1, nR):
                old_dj = oldD[j]
                for pi in range(len(routes[i])):
                    ci = routes[i][pi]
                    if ci not in cust_set:
                        continue
                    for pj in range(len(routes[j])):
                        cj = routes[j][pj]
                        if cj not in cust_set:
                            continue
                        if route_demand(routes[i]) - demands[ci] + demands[cj] > capacity:
                            continue
                        if route_demand(routes[j]) - demands[cj] + demands[ci] > capacity:
                            continue
                        ni = list(routes[i])
                        nj = list(routes[j])
                        ni[pi] = cj
                        nj[pj] = ci
                        si = simulate(ni)
                        sj = simulate(nj)
                        if si is None or sj is None:
                            continue
                        if si[0] + sj[0] < old_di + old_dj - 1e-7:
                            routes[i] = ni
                            routes[j] = nj
                            return routes, True
        return routes, False

    def deep_search(routes, budget_end):
        cur = total_dist(routes)
        if cur == float("inf"):
            return routes, cur
        while time.time() < budget_end:
            improved = False
            # 2-opt pass
            nr = [two_opt(r) for r in routes]
            nd = total_dist(nr)
            if nd < cur - 1e-7:
                routes, cur = nr, nd
                improved = True
            # relocate size 1
            nr, imp = relocate(routes, 1)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
                    nr = [two_opt(r) for r in routes]
                    nd2 = total_dist(nr)
                    if nd2 < cur - 1e-7:
                        routes, cur = nr, nd2
            # relocate size 2 (or-opt)
            nr, imp = relocate(routes, 2)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
            # swap
            nr, imp = swap(routes)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
            if not improved:
                break
        return routes, cur

    # ---- Build diverse candidates ----
    candidates = []
    order_angle = sorted(customers, key=lambda v: math.atan2(coords[v, 1] - coords[0, 1],
                                                            coords[v, 0] - coords[0, 0]))
    candidates.append(greedy_construct(order_angle))
    order_due = sorted(customers, key=lambda v: due[v])
    candidates.append(greedy_construct(order_due))
    rnd = random.Random(12345)
    for _ in range(3):
        o = customers[:]
        rnd.shuffle(o)
        candidates.append(greedy_construct(o))

    # quick 2-opt, then rank
    scored = []
    for cand in candidates:
        if cand is None:
            continue
        c2 = [two_opt(r) for r in cand]
        d = total_dist(c2)
        if d < float("inf"):
            scored.append((d, [list(r) for r in c2]))
    scored.sort(key=lambda x: x[0])

    if scored:
        global_best, global_d = scored[0][1], scored[0][0]
    else:
        global_best, global_d = None, float("inf")

    # deep search each candidate (in quality order), keep global best
    for d0, cand in scored:
        if time.time() > deadline - 12.0:
            break
        r, dd = deep_search(cand, deadline - 10.0)
        if dd < global_d - 1e-9:
            global_best, global_d = r, dd

    # spend spare time on extra random restarts (safely, only accept improvements)
    extra = random.Random(777)
    k = 0
    while time.time() < deadline - 18.0 and k < 40:
        o = customers[:]
        extra.shuffle(o)
        cand = greedy_construct(o)
        if cand is None:
            continue
        cand = [two_opt(r) for r in cand]
        d = total_dist(cand)
        if d == float("inf"):
            k += 1
            continue
        r, dd = deep_search(cand, deadline - 12.0)
        if dd < global_d - 1e-9:
            global_best, global_d = r, dd
        k += 1

    routes = global_best

    # ---- Final verification + guaranteed-valid fallback ----
    def all_feasible(rs):
        served = []
        for r in rs:
            s = simulate(r)
            if s is None:
                return False
            for v in r:
                if v in cust_set:
                    served.append(v)
        return len(served) == len(customers) and len(set(served)) == len(customers)

    if routes is None or not all_feasible(routes):
        routes = [[c] for c in customers]

    return routes
# EVOLVE-BLOCK-END

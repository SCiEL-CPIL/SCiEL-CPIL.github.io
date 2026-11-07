# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import heapq


def plan(instance):
    t0 = time.time()
    deadline = t0 + 132.0

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
    is_customer = np.zeros(n_nodes, dtype=bool)
    is_station = np.zeros(n_nodes, dtype=bool)
    for c in customers:
        is_customer[c] = True
    for s in stations:
        is_station[s] = True

    # Precompute customer demand array indexed by node
    cust_demand = np.zeros(n_nodes)
    for c in customers:
        cust_demand[c] = demands[c]

    def simulate_fast(route):
        """Fast simulation with early exit. Returns (dist, time, battery) or None."""
        t = 0.0
        bat = battery_cap
        cd = 0.0
        cur = 0
        for node in route:
            d = dist[cur, node]
            t += d / speed
            bat -= d
            if bat < -1e-9:
                return None
            cd += d
            if is_customer[node]:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return None
                t += service[node]
            elif is_station[node]:
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
            if is_customer[v]:
                s += cust_demand[v]
        return s

    def total_dist(routes):
        tot = 0.0
        for r in routes:
            s = simulate_fast(r)
            if s is None:
                return float("inf")
            tot += s[0]
        return tot

    def best_station_fix(route):
        """Insert cheapest single station if needed to make route feasible."""
        s = simulate_fast(route)
        if s is not None:
            return list(route), s[0]
        # Find where battery goes negative
        t = 0.0
        bat = battery_cap
        cur = 0
        neg_pos = -1
        for idx, node in enumerate(route):
            d = dist[cur, node]
            t += d / speed
            bat -= d
            if bat < -1e-9 and neg_pos == -1:
                neg_pos = idx
            if is_customer[node]:
                if t < ready[node]:
                    t = ready[node]
                t += service[node]
            elif is_station[node]:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            cur = node
        if neg_pos == -1:
            # Check return leg
            d = dist[cur, 0]
            bat -= d
            if bat < -1e-9:
                neg_pos = len(route)
            else:
                return None, None  # Not a battery issue

        # Try inserting stations near where battery runs low
        best_route = None
        best_d = float('inf')
        search_start = max(0, neg_pos - 2)
        search_end = min(len(route) + 1, neg_pos + 3)
        for p in range(search_start, search_end + 1):
            for st in stations:
                cand = list(route[:p]) + [st] + list(route[p:])
                if route_demand(cand) > capacity:
                    continue
                sc = simulate_fast(cand)
                if sc is not None:
                    if sc[0] < best_d:
                        best_route, best_d = cand, sc[0]
        if best_route is not None:
            return best_route, best_d
        # Broader search if local didn't work
        for p in range(len(route) + 1):
            for st in stations:
                cand = list(route[:p]) + [st] + list(route[p:])
                if route_demand(cand) > capacity:
                    continue
                sc = simulate_fast(cand)
                if sc is not None:
                    if sc[0] < best_d:
                        best_route, best_d = cand, sc[0]
        if best_route is not None:
            return best_route, best_d
        return None, None

    def greedy_construct(order):
        routes = []
        for c in order:
            placed = False
            best = None
            for ri, r in enumerate(routes):
                if route_demand(r) + demands[c] > capacity:
                    continue
                # Find best position by distance delta
                best_pos = -1
                best_delta = float('inf')
                for pos in range(len(r) + 1):
                    prev = r[pos - 1] if pos > 0 else 0
                    nxt = r[pos] if pos < len(r) else 0
                    delta = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = pos
                if best_pos == -1:
                    continue
                cand = list(r[:best_pos]) + [c] + list(r[best_pos:])
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
                    fixed = [c]
                routes.append(fixed)
        return routes

    def two_opt(route, max_iter=120):
        if len(route) < 2:
            return route
        cur = simulate_fast(route)
        if cur is None:
            return route
        cur_d = cur[0]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Quick distance delta check
                    a = route[i - 1] if i > 0 else 0
                    b = route[i]
                    c = route[j]
                    d = route[j + 1] if j + 1 < n else 0
                    old_len = dist[a, b] + dist[c, d]
                    new_len = dist[a, c] + dist[b, d]
                    if new_len >= old_len - 1e-7:
                        continue
                    nr = list(route[:i]) + list(reversed(route[i:j + 1])) + list(route[j + 1:])
                    if route_demand(nr) > capacity:
                        continue
                    sn = simulate_fast(nr)
                    if sn is None:
                        continue
                    if sn[0] < cur_d - 1e-7:
                        route, cur_d = nr, sn[0]
                        improved = True
        return route

    def relocate(routes, blocklen=1):
        nR = len(routes)
        oldD = [(simulate_fast(r)[0] if r else 0.0) for r in routes]
        for i in range(nR):
            ri_full = routes[i]
            old_di = oldD[i]
            for pi in range(len(ri_full) - blocklen + 1):
                block = ri_full[pi:pi + blocklen]
                block_demand = 0.0
                for v in block:
                    if is_customer[v]:
                        block_demand += cust_demand[v]
                rest_i = ri_full[:pi] + ri_full[pi + blocklen:]
                sim_i = simulate_fast(rest_i) if rest_i else (0.0, 0.0, battery_cap)
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
                    # Find best insertion position by delta
                    best_pj = -1
                    best_delta = float('inf')
                    for pj in range(len(rj_base) + 1):
                        prev = rj_base[pj - 1] if pj > 0 else 0
                        nxt = rj_base[pj] if pj < len(rj_base) else 0
                        b0 = block[0]
                        b1 = block[-1]
                        delta = dist[prev, b0] + dist[b1, nxt] - dist[prev, nxt]
                        if delta < best_delta:
                            best_delta = delta
                            best_pj = pj
                    if best_pj == -1:
                        continue
                    rj = rj_base[:best_pj] + list(block) + rj_base[best_pj:]
                    sim_j = simulate_fast(rj)
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
        oldD = [(simulate_fast(r)[0] if r else 0.0) for r in routes]
        for i in range(nR):
            old_di = oldD[i]
            for j in range(i + 1, nR):
                old_dj = oldD[j]
                for pi in range(len(routes[i])):
                    ci = routes[i][pi]
                    if not is_customer[ci]:
                        continue
                    for pj in range(len(routes[j])):
                        cj = routes[j][pj]
                        if not is_customer[cj]:
                            continue
                        if route_demand(routes[i]) - demands[ci] + demands[cj] > capacity:
                            continue
                        if route_demand(routes[j]) - demands[cj] + demands[ci] > capacity:
                            continue
                        ni = list(routes[i])
                        nj = list(routes[j])
                        ni[pi] = cj
                        nj[pj] = ci
                        si = simulate_fast(ni)
                        sj = simulate_fast(nj)
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
            nr = [two_opt(r) for r in routes]
            nd = total_dist(nr)
            if nd < cur - 1e-7:
                routes, cur = nr, nd
                improved = True
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
            nr, imp = relocate(routes, 2)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
            nr, imp = swap(routes)
            if imp:
                nd = total_dist(nr)
                if nd < cur - 1e-7:
                    routes, cur = nr, nd
                    improved = True
            if not improved:
                break
        return routes, cur

    def dp_reinsertion(removed, routes, budget_end):
        """LNS: reinsert removed customers using Dijkstra over (route, position, time, battery)."""
        if not removed:
            return routes, total_dist(routes)
        
        n_removed = len(removed)
        # State: (total_dist, route_idx, pos_idx, time, battery, remaining_mask)
        # We'll use a simpler approach: try all permutations for small n_removed,
        # or greedy insertion with DP-like cost evaluation for larger sets.
        
        if n_removed <= 3:
            # Try all permutations
            from itertools import permutations
            best_routes = None
            best_cost = float('inf')
            for perm in permutations(removed):
                cand = [list(r) for r in routes]
                placed = True
                for c in perm:
                    best_ri = -1
                    best_pos = -1
                    best_delta = float('inf')
                    for ri, r in enumerate(cand):
                        if route_demand(r) + demands[c] > capacity:
                            continue
                        for pos in range(len(r) + 1):
                            prev = r[pos - 1] if pos > 0 else 0
                            nxt = r[pos] if pos < len(r) else 0
                            delta = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                            if delta < best_delta:
                                best_delta = delta
                                best_ri = ri
                                best_pos = pos
                    if best_ri == -1:
                        placed = False
                        break
                    cand[best_ri].insert(best_pos, c)
                if not placed:
                    continue
                # Fix stations and evaluate
                fixed_cand = []
                valid = True
                for r in cand:
                    fixed, cd = best_station_fix(r)
                    if fixed is None:
                        valid = False
                        break
                    fixed_cand.append(fixed)
                if not valid:
                    continue
                cost = total_dist(fixed_cand)
                if cost < best_cost:
                    best_cost = cost
                    best_routes = fixed_cand
            if best_routes is not None:
                return best_routes, best_cost
            return routes, total_dist(routes)
        else:
            # Greedy reinsertion with multiple restarts
            best_routes = None
            best_cost = total_dist(routes)
            for attempt in range(8):
                if time.time() > budget_end:
                    break
                cand = [list(r) for r in routes]
                rem = list(removed)
                if attempt > 0:
                    random.Random(attempt * 1000 + 42).shuffle(rem)
                else:
                    rem.sort(key=lambda c: dist[0, c])
                placed_all = True
                for c in rem:
                    best_ri = -1
                    best_pos = -1
                    best_delta = float('inf')
                    for ri, r in enumerate(cand):
                        if route_demand(r) + demands[c] > capacity:
                            continue
                        for pos in range(len(r) + 1):
                            prev = r[pos - 1] if pos > 0 else 0
                            nxt = r[pos] if pos < len(r) else 0
                            delta = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                            if delta < best_delta:
                                best_delta = delta
                                best_ri = ri
                                best_pos = pos
                    if best_ri == -1:
                        placed_all = False
                        break
                    cand[best_ri].insert(best_pos, c)
                if not placed_all:
                    continue
                fixed_cand = []
                valid = True
                for r in cand:
                    fixed, cd = best_station_fix(r)
                    if fixed is None:
                        valid = False
                        break
                    fixed_cand.append(fixed)
                if not valid:
                    continue
                cost = total_dist(fixed_cand)
                if cost < best_cost:
                    best_cost = cost
                    best_routes = fixed_cand
            if best_routes is not None:
                return best_routes, best_cost
            return routes, total_dist(routes)

    def lns(routes, budget_end):
        """Large Neighborhood Search: remove random subset, reinsert optimally."""
        cur = total_dist(routes)
        if cur == float("inf"):
            return routes, cur
        rng = random.Random(999)
        while time.time() < budget_end:
            n_routes = len(routes)
            # Choose which routes to perturb
            n_remove = rng.randint(3, min(8, n_customers))
            # Pick customers to remove from existing routes
            all_cust_in_routes = []
            for r in routes:
                for v in r:
                    if is_customer[v]:
                        all_cust_in_routes.append(v)
            if len(all_cust_in_routes) < n_remove:
                break
            removed = set(rng.sample(all_cust_in_routes, n_remove))
            # Remove them from routes
            new_routes = []
            for r in routes:
                nr = [v for v in r if v not in removed]
                if nr:
                    new_routes.append(nr)
            removed_list = list(removed)
            # Reinsert
            result_routes, result_cost = dp_reinsertion(removed_list, new_routes, budget_end)
            if result_cost < cur - 1e-7:
                routes = result_routes
                cur = result_cost
                # Polish after LNS improvement
                routes = [two_opt(r) for r in routes]
                cur = total_dist(routes)
        return routes, cur

    # Build diverse candidates
    candidates = []
    order_angle = sorted(customers, key=lambda v: math.atan2(coords[v, 1] - coords[0, 1],
                                                            coords[v, 0] - coords[0, 0]))
    candidates.append(greedy_construct(order_angle))
    order_due = sorted(customers, key=lambda v: due[v])
    candidates.append(greedy_construct(order_due))
    order_dist = sorted(customers, key=lambda v: dist[0, v])
    candidates.append(greedy_construct(order_dist))
    rnd = random.Random(12345)
    for _ in range(4):
        o = customers[:]
        rnd.shuffle(o)
        candidates.append(greedy_construct(o))

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

    for d0, cand in scored:
        if time.time() > deadline - 12.0:
            break
        r, dd = deep_search(cand, deadline - 10.0)
        if dd < global_d - 1e-9:
            global_best, global_d = r, dd

    extra = random.Random(777)
    k = 0
    while time.time() < deadline - 25.0 and k < 40:
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
        r, dd = deep_search(cand, deadline - 15.0)
        if dd < global_d - 1e-9:
            global_best, global_d = r, dd
        k += 1

    # LNS phase on the best solution
    if global_best is not None and time.time() < deadline - 10.0:
        global_best, global_d = lns(global_best, deadline - 8.0)

    # Final high-precision polish pass
    if global_best is not None and time.time() < deadline - 3.0:
        polish_end = deadline - 2.0
        while time.time() < polish_end:
            improved = False
            # More aggressive 2-opt
            nr = [two_opt(r, max_iter=300) for r in global_best]
            nd = total_dist(nr)
            if nd < global_d - 1e-7:
                global_best, global_d = nr, nd
                improved = True
            # Or-opt blocks of 3
            for ri, r in enumerate(global_best):
                n = len(r)
                if n < 4:
                    continue
                for _ in range(20):
                    imp2 = False
                    for i in range(n - 2):
                        block = r[i:i+3]
                        rest = r[:i] + r[i+3:]
                        for j in range(len(rest) + 1):
                            if j <= i - 1 and j >= i - 1:
                                continue
                            nr2 = rest[:j] + block + rest[j:]
                            if route_demand(nr2) > capacity:
                                continue
                            sn = simulate_fast(nr2)
                            if sn is None:
                                continue
                            old_sim = simulate_fast(r)
                            if old_sim is None:
                                break
                            if sn[0] < old_sim[0] - 1e-7:
                                global_best[ri] = nr2
                                imp2 = True
                                improved = True
                                break
                        if imp2:
                            break
                    if not imp2:
                        break
            if not improved:
                break

    routes = global_best

    def all_feasible(rs):
        served = []
        for r in rs:
            s = simulate_fast(r)
            if s is None:
                return False
            for v in r:
                if is_customer[v]:
                    served.append(v)
        return len(served) == len(customers) and len(set(served)) == len(customers)

    if routes is None or not all_feasible(routes):
        # Fallback: build feasible solution one customer at a time
        routes = []
        for c in customers:
            fixed, cd = best_station_fix([c])
            if fixed is None:
                fixed = [c]
            routes.append(fixed)

    return routes
# EVOLVE-BLOCK-END

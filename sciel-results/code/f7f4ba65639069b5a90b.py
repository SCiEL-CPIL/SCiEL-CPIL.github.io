"""EV routing with time windows and battery; clustered construction + fast local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from math import sqrt, inf

def plan(instance):
    t0 = time.time()
    deadline = t0 + 138.0  # safety margin before 150s kill

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    N = len(coords)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    cust = list(range(1, n_customers + 1))
    station_set = set(stations)
    is_customer = np.zeros(N, dtype=bool)
    is_customer[1:n_customers + 1] = True

    # Fast simulation: given a sequence of customer/station nodes (no depot),
    # check feasibility and return (ok, total_dist, full_route_with_stations_inserted)
    def simulate_route(cust_seq):
        """cust_seq is a list of customer node ids only. Returns (ok, total_dist, full_route)."""
        if not cust_seq:
            return True, 0.0, []
        
        total_demand = sum(demands[v] for v in cust_seq)
        if total_demand > capacity:
            return False, 0.0, []
        
        t = 0.0
        bat = battery_cap
        d_total = 0.0
        full_route = []
        prev = 0
        
        for v in cust_seq:
            dd = dist[prev, v]
            if bat >= dd - 1e-9:
                t += dd / speed
                bat -= dd
                d_total += dd
            else:
                # Need to charge: find best station
                best_s = -1
                best_cost = np.inf
                for s in stations:
                    d1 = dist[prev, s]
                    if d1 > bat + 1e-9:
                        continue
                    d2 = dist[s, v]
                    if d2 > battery_cap + 1e-9:
                        continue
                    cost = d1 + d2
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s
                if best_s == -1:
                    return False, 0.0, []
                d1 = dist[prev, best_s]
                t += d1 / speed
                bat -= d1
                d_total += d1
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
                full_route.append(best_s)
                d2 = dist[best_s, v]
                t += d2 / speed
                bat -= d2
                d_total += d2
                prev = best_s
            
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, []
            t += service[v]
            full_route.append(v)
            prev = v
        
        # Return to depot
        dd = dist[prev, 0]
        if bat >= dd - 1e-9:
            t += dd / speed
            bat -= dd
            d_total += dd
        else:
            best_s = -1
            best_cost = np.inf
            for s in stations:
                d1 = dist[prev, s]
                if d1 > bat + 1e-9:
                    continue
                d2 = dist[s, 0]
                if d2 > battery_cap + 1e-9:
                    continue
                cost = d1 + d2
                if cost < best_cost:
                    best_cost = cost
                    best_s = s
            if best_s == -1:
                return False, 0.0, []
            d1 = dist[prev, best_s]
            t += d1 / speed
            bat -= d1
            d_total += d1
            charge_time = (battery_cap - bat) * recharge_rate
            t += charge_time
            bat = battery_cap
            full_route.append(best_s)
            d2 = dist[best_s, 0]
            t += d2 / speed
            bat -= d2
            d_total += d2
        
        if t > horizon + 1e-9:
            return False, 0.0, []
        
        return True, d_total, full_route

    # K-means clustering
    def _kmeans(points, k, seed=42, iters=50):
        rng = np.random.default_rng(seed)
        n = len(points)
        if k >= n:
            return list(range(n))
        pts = np.array(points, dtype=float)
        idx = rng.choice(n, size=k, replace=False)
        centers = pts[idx].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(d, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                m = labels == c
                if m.any():
                    centers[c] = pts[m].mean(axis=0)
        return labels.tolist()

    # Construction: cluster + nearest neighbor within cluster + cheapest insertion
    def construct_routes():
        customers = list(range(1, n_customers + 1))
        # Sort by due time for tie-breaking in insertion
        customers.sort(key=lambda v: due[v])
        
        # Determine number of routes: aim for ~6-10 customers per route
        k = max(2, min(6, n_customers // 5))
        pts = [coords[v] for v in customers]
        labels = _kmeans(pts, k, seed=42)
        
        clusters = [[] for _ in range(k)]
        for i, v in enumerate(customers):
            clusters[labels[i]].append(v)
        
        routes = []
        remaining = set(cust)
        
        for cl in clusters:
            if not cl:
                continue
            # Sort cluster by nearest neighbor from depot
            route = []
            prev = 0
            cl_remaining = list(cl)
            cl_remaining.sort(key=lambda v: dist[0, v])
            
            for v in cl_remaining:
                if v not in remaining:
                    continue
                # Try to insert v into route at best position
                best_pos = -1
                best_d = np.inf
                for pos in range(len(route) + 1):
                    new_seq = route[:pos] + [v] + route[pos:]
                    ok, d, _ = simulate_route(new_seq)
                    if ok and d < best_d:
                        best_d = d
                        best_pos = pos
                
                if best_pos >= 0:
                    route = route[:best_pos] + [v] + route[best_pos:]
                    remaining.discard(v)
                else:
                    # Can't insert into this route; start new route
                    if route:
                        routes.append(route)
                        route = []
                    ok, d, full = simulate_route([v])
                    if ok:
                        route = [v]
                        remaining.discard(v)
                    # else: leave for later
        
        if route:
            routes.append(route)
        
        # Handle any remaining customers
        for v in list(remaining):
            placed = False
            for i, r in enumerate(routes):
                for pos in range(len(r) + 1):
                    new_seq = r[:pos] + [v] + r[pos:]
                    ok, d, _ = simulate_route(new_seq)
                    if ok:
                        routes[i] = new_seq
                        remaining.discard(v)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                routes.append([v])
                remaining.discard(v)
        
        return routes

    # Route distance (customer-only sequence, for comparison)
    def get_route_dist(cust_seq):
        if not cust_seq:
            return 0.0
        d = 0.0
        prev = 0
        for v in cust_seq:
            d += dist[prev, v]
            prev = v
        d += dist[prev, 0]
        return d

    def extract_custs(full_route):
        return [v for v in full_route if is_customer[v]]

    # ---------- Build initial solution ----------
    routes = construct_routes()
    
    # Convert to customer-only sequences for local search
    cust_routes = [extract_custs(r) for r in routes]
    
    # Verify and get distances
    valid_routes = []
    route_dists = []
    for cr in cust_routes:
        ok, d, full = simulate_route(cr)
        if ok:
            valid_routes.append(full)
            route_dists.append(d)
        else:
            # Fallback: split into individual customers
            for v in cr:
                ok2, d2, full2 = simulate_route([v])
                if ok2:
                    valid_routes.append(full2)
                    route_dists.append(d2)
    
    routes = valid_routes
    total_d = sum(route_dists)
    
    # ---------- Local Search ----------
    improved = True
    no_improve_count = 0
    max_no_improve = 3
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            break
        
        # 2-opt intra-route
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri_custs = extract_custs(routes[i])
            n = len(ri_custs)
            if n < 2:
                continue
            found = False
            for a in range(n - 1):
                if found:
                    break
                for b in range(a + 1, n):
                    new_seq = ri_custs[:a] + ri_custs[a:b+1][::-1] + ri_custs[b+1:]
                    ok, d, new_route = simulate_route(new_seq)
                    if ok and d < route_dists[i] - 1e-9:
                        old_d = route_dists[i]
                        routes[i] = new_route
                        route_dists[i] = d
                        total_d += d - old_d
                        improved = True
                        found = True
                        break
            if found:
                no_improve_count = 0
                break
        if improved:
            continue
        
        # Relocate
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri_custs = extract_custs(routes[i])
            if len(ri_custs) <= 1:
                continue
            for j in range(len(ri_custs)):
                v = ri_custs[j]
                src_without = ri_custs[:j] + ri_custs[j+1:]
                ok_s, d_s, src_route = simulate_route(src_without)
                if not ok_s:
                    continue
                for k in range(len(routes)):
                    if k == i:
                        continue
                    rk_custs = extract_custs(routes[k])
                    for pos in range(len(rk_custs) + 1):
                        new_tgt = rk_custs[:pos] + [v] + rk_custs[pos:]
                        ok_t, d_t, tgt_route = simulate_route(new_tgt)
                        if not ok_t:
                            continue
                        new_pair = d_s + d_t
                        old_pair = route_dists[i] + route_dists[k]
                        if new_pair < old_pair - 1e-9:
                            routes[i] = src_route
                            routes[k] = tgt_route
                            route_dists[i] = d_s
                            route_dists[k] = d_t
                            total_d += new_pair - old_pair
                            improved = True
                            no_improve_count = 0
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        
        # Swap
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri_custs = extract_custs(routes[i])
            for k in range(i + 1, len(routes)):
                rk_custs = extract_custs(routes[k])
                found = False
                for a in range(len(ri_custs)):
                    if found:
                        break
                    for b in range(len(rk_custs)):
                        vi = ri_custs[a]
                        vk = rk_custs[b]
                        new_ri = list(ri_custs)
                        new_rk = list(rk_custs)
                        new_ri[a] = vk
                        new_rk[b] = vi
                        ok_i, d_i, route_i = simulate_route(new_ri)
                        ok_k, d_k, route_k = simulate_route(new_rk)
                        if ok_i and ok_k and d_i + d_k < route_dists[i] + route_dists[k] - 1e-9:
                            old_sum = route_dists[i] + route_dists[k]
                            routes[i] = route_i
                            routes[k] = route_k
                            route_dists[i] = d_i
                            route_dists[k] = d_k
                            total_d += d_i + d_k - old_sum
                            improved = True
                            no_improve_count = 0
                            found = True
                            break
                if found:
                    break
            if improved:
                break
        if improved:
            continue
        
        # 2-opt inter-route (connect two routes)
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri_custs = extract_custs(routes[i])
            for k in range(i + 1, len(routes)):
                rk_custs = extract_custs(routes[k])
                # Try merging: take suffix of i + prefix of k
                for a in range(len(ri_custs)):
                    for b in range(len(rk_custs)):
                        new_seq = ri_custs[:a] + rk_custs[b:]
                        new_seq2 = rk_custs[:b] + ri_custs[a:]
                        ok1, d1, r1 = simulate_route(new_seq)
                        ok2, d2, r2 = simulate_route(new_seq2)
                        if ok1 and ok2:
                            old_sum = route_dists[i] + route_dists[k]
                            if d1 + d2 < old_sum - 1e-9:
                                routes[i] = r1
                                routes[k] = r2
                                route_dists[i] = d1
                                route_dists[k] = d2
                                total_d += d1 + d2 - old_sum
                                improved = True
                                no_improve_count = 0
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

    # Final validation
    all_customers = set(cust)
    covered = set()
    for r in routes:
        for v in r:
            if is_customer[v]:
                covered.add(v)
    
    # Add any missing customers as individual routes
    for v in all_customers - covered:
        ok, d, full = simulate_route([v])
        if ok:
            routes.append(full)
    
    return routes
# EVOLVE-BLOCK-END

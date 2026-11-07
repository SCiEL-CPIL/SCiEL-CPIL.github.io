"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start = time.time()
    deadline = start + 145.0  # safety margin vs 150s wall-clock

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
    # Precompute Euclidean distances.
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(d2)

    customers = list(range(1, n_customers + 1))

    # ---------- Fast feasibility check ----------
    def simulate(route):
        """Return (ok, total_dist, end_time). Battery/time windows/capacity/horizon."""
        load = demands[route].sum() if route else 0.0
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

    # ---------- Route distance helper ----------
    def route_dist(route):
        if not route:
            return 0.0
        t = dist[0, route[0]]
        for i in range(len(route) - 1):
            t += dist[route[i], route[i + 1]]
        t += dist[route[-1], 0]
        return t

    # ---------- Fast insertion cost estimation ----------
    def insertion_cost(route, pos, v):
        """Estimate extra distance if v is inserted at pos in route."""
        if not route:
            return dist[0, v] + dist[v, 0]
        if pos == 0:
            return dist[v, route[0]] + dist[0, v] - dist[0, route[0]]
        elif pos == len(route):
            return dist[route[-1], v] + dist[v, 0] - dist[route[-1], 0]
        else:
            return dist[route[pos-1], v] + dist[v, route[pos]] - dist[route[pos-1], route[pos]]

    # ---------- Greedy construction with nearest-neighbor seeding ----------
    def construct(k_routes, seed):
        rng = random.Random(seed)
        
        # Sort customers by due date for time-window awareness
        sorted_customers = sorted(customers, key=lambda v: (due[v], ready[v]))
        
        routes = [[] for _ in range(k_routes)]
        route_loads = [0.0] * k_routes
        remaining = list(sorted_customers)
        
        for v in remaining:
            best_r = -1
            best_pos = -1
            best_cost = float('inf')
            
            for r in range(k_routes):
                if route_loads[r] + demands[v] > capacity + 1e-9:
                    continue
                route = routes[r]
                L = len(route)
                
                # Try all positions, but use insertion_cost for fast filtering
                for pos in range(L + 1):
                    ic = insertion_cost(route, pos, v)
                    if ic >= best_cost:
                        continue
                    # Full feasibility check only for promising candidates
                    new_route = route[:pos] + [v] + route[pos:]
                    ok, tot, _ = simulate(new_route)
                    if ok:
                        # Use actual total distance for comparison
                        actual_cost = tot
                        if actual_cost < best_cost:
                            best_cost = actual_cost
                            best_r = r
                            best_pos = pos
                            best_new_route = new_route
            
            if best_r >= 0:
                routes[best_r] = best_new_route
                route_loads[best_r] += demands[v]
            else:
                # Start a new route
                routes.append([v])
                route_loads.append(demands[v])
        
        # Drop empty routes
        routes = [r for r in routes if r]
        return routes

    # ---------- 2-opt with fast distance delta ----------
    def two_opt(route, max_iter=5000):
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

    # ---------- Or-opt: move a segment of length 1-3 ----------
    def or_opt(route, max_iter=2000):
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
            for seg_len in (1, 2, 3):
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

    # ---------- Cross-route 2-opt (swap segments between routes) ----------
    def cross_route_opt(routes, max_iter=100):
        """Try swapping single customers between routes."""
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n_routes = len(routes)
            for r1 in range(n_routes):
                for r2 in range(r1 + 1, n_routes):
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for i in range(len(route1)):
                        v1 = route1[i]
                        for j in range(len(route2)):
                            v2 = route2[j]
                            # Check capacity
                            if demands[v1] + sum(demands[x] for x in route2) - demands[v2] > capacity + 1e-9:
                                continue
                            if demands[v2] + sum(demands[x] for x in route1) - demands[v1] > capacity + 1e-9:
                                continue
                            # Try swap
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

    # ---------- Total distance ----------
    def total_distance(routes):
        return sum(route_dist(r) for r in routes)

    # Choose initial number of routes based on total demand.
    total_demand = demands[customers].sum()
    min_routes = max(1, int(math.ceil(total_demand / capacity)))
    base_k = min_routes + 1

    best_routes = None
    best_score = float('inf')

    # Try a few route counts, keep best.
    k_candidates = sorted(set([min_routes, base_k, base_k + 1, base_k + 2]))
    
    for k in k_candidates:
        if time.time() > deadline - 10.0:
            break
        for seed in range(8):
            if time.time() > deadline - 10.0:
                break
            routes = construct(k, seed)
            # Local search: 2-opt then or-opt on each route
            for r in range(len(routes)):
                if time.time() > deadline - 10.0:
                    break
                routes[r] = two_opt(routes[r])
                routes[r] = or_opt(routes[r])
            # Cross-route optimization
            if time.time() < deadline - 15.0:
                routes = cross_route_opt(routes)
            score = total_distance(routes)
            if score < best_score:
                best_score = score
                best_routes = [list(r) for r in routes]

    # Fallback: ensure every customer served
    if best_routes is None:
        best_routes = [[v] for v in customers]
    else:
        served = set()
        for r in best_routes:
            for v in r:
                served.add(v)
        missing = [v for v in customers if v not in served]
        for v in missing:
            best_routes.append([v])

    return best_routes
# EVOLVE-BLOCK-END

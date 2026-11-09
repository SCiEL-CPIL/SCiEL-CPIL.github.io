"""EV routing with time windows and battery; construction + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def plan(instance):
    t0 = time.time()
    deadline = t0 + 142.0  # safety margin before 150s kill

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    N = len(coords)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    cust = list(range(1, n_customers + 1))

    def check_route(route):
        """Check route validity, return (ok, total_distance, total_time)."""
        if not route:
            return True, 0.0, 0.0
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0, 0.0
        t = 0.0
        bat = float(battery)
        d = 0.0
        prev = 0
        for v in route:
            dd = dist[prev, v]
            if bat < dd - 1e-9:
                return False, 0.0, 0.0
            bat -= dd
            t += dd / speed
            d += dd
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0
            t += service[v]
            prev = v
        # return to depot
        dd = dist[prev, 0]
        if bat < dd - 1e-9:
            return False, 0.0, 0.0
        bat -= dd
        t += dd / speed
        d += dd
        if t > horizon + 1e-9:
            return False, 0.0, 0.0
        return True, d, t

    def try_insert(route, v, pos):
        newroute = route[:pos] + [v] + route[pos:]
        ok, d, t = check_route(newroute)
        if not ok:
            return None
        return newroute, d

    # ---------- Construction ----------
    routes = []
    remaining = list(cust)
    while remaining:
        route = []
        while True:
            if time.time() > deadline:
                break
            best = None
            best_key = None
            for v in remaining:
                for pos in range(len(route) + 1):
                    res = try_insert(route, v, pos)
                    if res is None:
                        continue
                    newroute, d = res
                    key = (d,)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (v, newroute)
            if best is None:
                break
            v, newroute = best
            route = newroute
            remaining.remove(v)
        if route:
            routes.append(route)

    # ---------- Local search with cached route info ----------
    def get_route_info(route):
        """Cache-friendly route check."""
        if not route:
            return True, 0.0, 0.0
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0, 0.0
        t = 0.0
        bat = float(battery)
        d = 0.0
        prev = 0
        for v in route:
            dd = dist[prev, v]
            if bat < dd - 1e-9:
                return False, 0.0, 0.0
            bat -= dd
            t += dd / speed
            d += dd
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0
            t += service[v]
            prev = v
        dd = dist[prev, 0]
        if bat < dd - 1e-9:
            return False, 0.0, 0.0
        bat -= dd
        t += dd / speed
        d += dd
        if t > horizon + 1e-9:
            return False, 0.0, 0.0
        return True, d, t

    def total_dist(routes):
        s = 0.0
        for r in routes:
            ok, d, _ = get_route_info(r)
            if ok:
                s += d
        return s

    best_total = total_dist(routes)
    best_routes = [list(r) for r in routes]

    # Precompute distance for each route
    def compute_all_route_dists(routes):
        dists = []
        for r in routes:
            ok, d, _ = get_route_info(r)
            dists.append(d if ok else float('inf'))
        return dists

    improved = True
    while improved and time.time() < deadline:
        improved = False
        route_dists = compute_all_route_dists(routes)

        # Relocate: move one customer from route i to route k
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri = routes[i]
            if len(ri) <= 1:
                continue
            for j in range(len(ri)):
                v = ri[j]
                src_without = ri[:j] + ri[j+1:]
                ok_s, d_s, _ = get_route_info(src_without)
                if not ok_s:
                    continue
                best_gain = 1e-9
                best_target = None
                best_pos = None
                best_new_tgt = None
                for k in range(len(routes)):
                    if k == i:
                        continue
                    tgt = routes[k]
                    for pos in range(len(tgt) + 1):
                        new_tgt = tgt[:pos] + [v] + tgt[pos:]
                        ok_t, d_t, _ = get_route_info(new_tgt)
                        if not ok_t:
                            continue
                        gain = (d_t + d_s) - (route_dists[k] + route_dists[i])
                        if -gain > best_gain:
                            best_gain = -gain
                            best_target = k
                            best_pos = pos
                            best_new_tgt = new_tgt
                if best_target is not None:
                    routes[i] = src_without
                    routes[best_target] = best_new_tgt
                    if not routes[i]:
                        routes.pop(i)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        # Swap: exchange customers between two routes
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            for k in range(i + 1, len(routes)):
                ri = routes[i]
                rk = routes[k]
                found_swap = False
                for a in range(len(ri)):
                    if found_swap:
                        break
                    for b in range(len(rk)):
                        vi = ri[a]
                        vk = rk[b]
                        new_ri = list(ri)
                        new_rk = list(rk)
                        new_ri[a] = vk
                        new_rk[b] = vi
                        ok_i, d_i, _ = get_route_info(new_ri)
                        ok_k, d_k, _ = get_route_info(new_rk)
                        if not (ok_i and ok_k):
                            continue
                        if d_i + d_k < route_dists[i] + route_dists[k] - 1e-9:
                            routes[i] = new_ri
                            routes[k] = new_rk
                            improved = True
                            found_swap = True
                            break
                if found_swap:
                    break
            if improved:
                break
        if improved:
            continue

        # 2-opt intra route
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri = routes[i]
            n = len(ri)
            if n < 2:
                continue
            found = False
            for a in range(n - 1):
                if found:
                    break
                for b in range(a + 1, n):
                    new_ri = ri[:a] + ri[a:b+1][::-1] + ri[b+1:]
                    ok, d, _ = get_route_info(new_ri)
                    if not ok:
                        continue
                    if d < route_dists[i] - 1e-9:
                        routes[i] = new_ri
                        ri = new_ri
                        n = len(ri)
                        improved = True
                        found = True
                        break
            if found:
                break
        if improved:
            continue

    cur = total_dist(routes)
    if cur < best_total:
        best_total = cur
        best_routes = [list(r) for r in routes]

    return best_routes
# EVOLVE-BLOCK-END

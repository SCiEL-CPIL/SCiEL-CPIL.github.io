"""EV routing with time windows and battery; construction + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def plan(instance):
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin before 150s kill

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
        """Check route feasibility. Returns (ok, total_distance, total_time)."""
        if not route:
            return True, 0.0, 0.0
        total_demand = demands[route].sum()
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

    def route_distance(route):
        """Compute total Euclidean distance of a route (depot legs included)."""
        if not route:
            return 0.0
        d = dist[0, route[0]]
        for i in range(len(route) - 1):
            d += dist[route[i], route[i + 1]]
        d += dist[route[-1], 0]
        return float(d)

    def incremental_route_distance(route, i, j):
        """Compute distance change when reversing segment [i..j] in route.
        Returns the new total distance."""
        if not route or i >= j:
            return route_distance(route)
        n = len(route)
        # Old edges affected: (i-1, i) and (j, j+1)
        # New edges: (i-1, j) and (i, j+1)
        d = route_distance(route)
        old1 = dist[route[i - 1], route[i]] if i > 0 else dist[0, route[i]]
        old2 = dist[route[j], route[j + 1]] if j < n - 1 else dist[route[j], 0]
        new1 = dist[route[i - 1], route[j]] if i > 0 else dist[0, route[j]]
        new2 = dist[route[i], route[j + 1]] if j < n - 1 else dist[route[i], 0]
        return d - old1 - old2 + new1 + new2

    # ---------- Clustering-based Construction ----------
    # Use k-means style spatial clustering to partition customers into groups
    # Each group becomes a route; we try multiple k values and pick best.

    def kmeans_cluster(points, k, max_iter=50, seed=42):
        """Simple k-means on 2D points. Returns cluster labels."""
        rng = np.random.RandomState(seed)
        n = len(points)
        if k >= n:
            return list(range(n))
        # Initialize: pick k random points as centroids
        idx = rng.choice(n, k, replace=False)
        centroids = points[idx].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # Assign to nearest centroid
            d2 = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            new_labels = d2.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # Update centroids
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    centroids[c] = points[mask].mean(axis=0)
        return labels.tolist()

    def construct_routes_for_k(k):
        """Build routes using k-means clustering with k clusters."""
        if k <= 0:
            k = 1
        if k > n_customers:
            k = n_customers

        cust_coords = coords[1:n_customers + 1]
        labels = kmeans_cluster(cust_coords, k)

        # Group customers by cluster
        clusters = [[] for _ in range(k)]
        for i, lab in enumerate(labels):
            clusters[lab].append(i + 1)  # customer id = index + 1

        routes = []
        for cluster in clusters:
            if not cluster:
                continue
            # Sort by angle from depot (sweep heuristic)
            depot_pos = coords[0]
            angles = [np.arctan2(coords[c][1] - depot_pos[1], coords[c][0] - depot_pos[0]) for c in cluster]
            order = np.argsort(angles)
            route = [cluster[i] for i in order]

            # Check feasibility; if infeasible, try splitting
            ok, d, t = check_route(route)
            if ok:
                routes.append(route)
            else:
                # Try to fix by splitting into two sub-routes
                mid = len(route) // 2
                r1 = route[:mid]
                r2 = route[mid:]
                ok1, _, _ = check_route(r1)
                ok2, _, _ = check_route(r2)
                if ok1:
                    routes.append(r1)
                if ok2:
                    routes.append(r2)
                if not ok1 and not ok2:
                    # Fallback: add one by one
                    for c in route:
                        routes.append([c])

        return routes

    # Try multiple k values and pick the one with best total distance
    best_routes = None
    best_dist = float('inf')
    for k in range(2, 10):
        if time.time() > deadline:
            break
        routes = construct_routes_for_k(k)
        total_d = sum(route_distance(r) for r in routes)
        if total_d < best_dist:
            best_dist = total_d
            best_routes = [list(r) for r in routes]

    if best_routes is None:
        # Fallback: one customer per route
        best_routes = [[c] for c in cust]
        best_dist = sum(route_distance(r) for r in best_routes)

    routes = [list(r) for r in best_routes]

    # ---------- Local Search ----------
    def total_dist():
        return sum(route_distance(r) for r in routes)

    cur_total = total_dist()
    best_total = cur_total
    best_routes_ls = [list(r) for r in routes]

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # --- 2-opt intra-route (fast with incremental distance) ---
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri = routes[i]
            n = len(ri)
            if n < 2:
                continue
            found = False
            for a in range(n - 1):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n):
                    # Fast distance estimate before full check
                    new_d = incremental_route_distance(ri, a, b)
                    old_d = route_distance(ri)
                    if new_d >= old_d - 1e-9:
                        continue
                    new_ri = ri[:a] + ri[a:b + 1][::-1] + ri[b + 1:]
                    ok, _, _ = check_route(new_ri)
                    if not ok:
                        continue
                    routes[i] = new_ri
                    ri = new_ri
                    n = len(ri)
                    improved = True
                    found = True
                    break
                if found:
                    break
            if found:
                break
        if improved:
            continue

        # --- Relocate ---
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            for j in range(len(routes[i])):
                v = routes[i][j]
                src = routes[i]
                src_without = src[:j] + src[j + 1:]
                ok_s, d_s_new, _ = check_route(src_without)
                if not ok_s:
                    continue
                d_s_old = route_distance(src)
                best_gain = 1e-9
                best_target = None
                best_pos = None
                for k in range(len(routes)):
                    if k == i:
                        continue
                    tgt = routes[k]
                    d_t_old = route_distance(tgt)
                    for pos in range(len(tgt) + 1):
                        new_tgt = tgt[:pos] + [v] + tgt[pos:]
                        ok_t, d_t_new, _ = check_route(new_tgt)
                        if not ok_t:
                            continue
                        gain = (d_t_new + d_s_new) - (d_t_old + d_s_old)
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

        # --- Swap ---
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            for k in range(i + 1, len(routes)):
                ri = routes[i]
                rk = routes[k]
                d_i_old = route_distance(ri)
                d_k_old = route_distance(rk)
                for a in range(len(ri)):
                    if time.time() > deadline:
                        break
                    for b in range(len(rk)):
                        vi = ri[a]
                        vk = rk[b]
                        new_ri = list(ri)
                        new_rk = list(rk)
                        new_ri[a] = vk
                        new_rk[b] = vi
                        ok_i, d_i_new, _ = check_route(new_ri)
                        if not ok_i:
                            continue
                        ok_k, d_k_new, _ = check_route(new_rk)
                        if not ok_k:
                            continue
                        if d_i_new + d_k_new < d_i_old + d_k_old - 1e-9:
                            routes[i] = new_ri
                            routes[k] = new_rk
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

    cur = total_dist()
    if cur < best_total:
        best_total = cur
        best_routes_ls = [list(r) for r in routes]

    return best_routes_ls
# EVOLVE-BLOCK-END

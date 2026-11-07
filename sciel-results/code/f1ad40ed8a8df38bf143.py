"""EV routing: parallel multi-start ILS with vectorized feasibility and inter-route Or-Opt."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import sqrt

WALL_LIMIT = 140.0  # safety margin under 150s kill


def _dist_matrix(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            d[j, i] = d[i, j]
    return d


def _eval_route(route, D, demands, ready, due, service, stations_set,
                capacity, bat_max, speed, rrate, horizon, n_cust):
    """Return (valid, dist, time, end_battery)."""
    if not route:
        return True, 0.0, 0.0, bat_max
    total_dem = 0
    for n in route:
        if 1 <= n <= n_cust:
            total_dem += demands[n]
    if total_dem > capacity:
        return False, 0.0, 0.0, 0.0

    t = 0.0
    bat = bat_max
    dist = 0.0
    cur = 0
    for n in route:
        d = D[cur, n]
        bat -= d
        if bat < -1e-9:
            return False, 0.0, 0.0, 0.0
        t += d / speed
        dist += d
        cur = n
        if n in stations_set:
            t += (bat_max - bat) * rrate
            bat = bat_max
        else:
            if t < ready[n]:
                t = ready[n]
            if t > due[n] + 1e-9:
                return False, 0.0, 0.0, 0.0
            t += service[n]
    # back to depot
    d = D[cur, 0]
    bat -= d
    if bat < -1e-9:
        return False, 0.0, 0.0, 0.0
    t += d / speed
    dist += d
    if t > horizon + 1e-9:
        return False, 0.0, 0.0, 0.0
    return True, dist, t, bat


def _eval_solution(routes, D, demands, ready, due, service, stations_set,
                   capacity, bat_max, speed, rrate, horizon, n_cust):
    """Return (valid, total_dist)."""
    visited = set()
    total_dist = 0.0
    for route in routes:
        for n in route:
            if 1 <= n <= n_cust:
                if n in visited:
                    return False, 0.0
                visited.add(n)
        valid, dist, _, _ = _eval_route(route, D, demands, ready, due, service,
                                         stations_set, capacity, bat_max, speed,
                                         rrate, horizon, n_cust)
        if not valid:
            return False, 0.0
        total_dist += dist
    if len(visited) != n_cust:
        return False, 0.0
    return True, total_dist


def _construct_nn(remaining, D, demands, ready, due, service, stations_set,
                  capacity, bat_max, speed, rrate, horizon, n_cust, start=0):
    """Nearest-neighbor construction for one route starting from depot."""
    route = []
    t = 0.0
    bat = bat_max
    cur = start
    total_dem = 0
    rem = list(remaining)
    random.shuffle(rem)

    while rem:
        best = -1
        best_d = float('inf')
        for idx, cand in enumerate(rem):
            if 1 <= cand <= n_cust and total_dem + demands[cand] > capacity:
                continue
            d = D[cur, cand]
            nb = bat - d
            if nb < -1e-9:
                continue
            nt = t + d / speed
            if cand in stations_set:
                nt += (bat_max - nb) * rrate
            else:
                nt = max(nt, ready[cand])
                if nt > due[cand] + 1e-9:
                    continue
            if d < best_d:
                best_d = d
                best = idx
        if best < 0:
            break
        node = rem.pop(best)
        d = D[cur, node]
        bat -= d
        t += d / speed
        cur = node
        if node in stations_set:
            t += (bat_max - bat) * rrate
            bat = bat_max
        else:
            t = max(t, ready[node])
            t += service[node]
            if 1 <= node <= n_cust:
                total_dem += demands[node]
        route.append(node)

    # check return
    d = D[cur, 0]
    if bat - d < -1e-9:
        return route, remaining  # can't return, keep as is (will be invalid)
    return route, [n for n in remaining if n not in set(route)]


def _construct_all(customers, D, demands, ready, due, service, stations_set,
                   capacity, bat_max, speed, rrate, horizon, n_cust, seed=0):
    """Build full solution with NN, multiple routes."""
    rng = random.Random(seed)
    rem = list(customers)
    rng.shuffle(rem)
    routes = []
    while rem:
        route, rem = _construct_nn(rem, D, demands, ready, due, service,
                                    stations_set, capacity, bat_max, speed,
                                    rrate, horizon, n_cust)
        if route:
            routes.append(route)
        else:
            # fallback: one customer per route
            for c in rem:
                routes.append([c])
            rem = []
            break
    return routes


def _two_opt(route, D, demands, ready, due, service, stations_set,
             capacity, bat_max, speed, rrate, horizon, n_cust, max_passes=5):
    """2-opt on a single route (customers only, stations stay in place)."""
    if len(route) < 2:
        return route
    best = list(route)
    best_valid, best_dist, _, _ = _eval_route(best, D, demands, ready, due, service,
                                               stations_set, capacity, bat_max, speed,
                                               rrate, horizon, n_cust)
    if not best_valid:
        return best
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j+1][::-1] + best[j+1:]
                v, d, _, _ = _eval_route(cand, D, demands, ready, due, service,
                                          stations_set, capacity, bat_max, speed,
                                          rrate, horizon, n_cust)
                if v and d < best_dist - 1e-9:
                    best = cand
                    best_dist = d
                    improved = True
    return best


def _or_opt_inter(routes, D, demands, ready, due, service, stations_set,
                  capacity, bat_max, speed, rrate, horizon, n_cust, max_iters=30):
    """Move 1-2 consecutive customers from one route to another (inter-route)."""
    best_routes = [list(r) for r in routes]
    best_valid, best_dist = _eval_solution(best_routes, D, demands, ready, due, service,
                                            stations_set, capacity, bat_max, speed,
                                            rrate, horizon, n_cust)
    if not best_valid:
        return best_routes
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(len(best_routes)):
            for j in range(len(best_routes)):
                if i == j:
                    continue
                ri = best_routes[i]
                if len(ri) < 1:
                    continue
                # try moving 1 or 2 customers from ri to position in rj
                for seg_len in [1, 2]:
                    for pos in range(len(ri) - seg_len + 1):
                        seg = ri[pos:pos+seg_len]
                        # remove from ri
                        new_ri = ri[:pos] + ri[pos+seg_len:]
                        # try all insertion positions in rj
                        for ins_pos in range(len(best_routes[j]) + 1):
                            new_rj = best_routes[j][:ins_pos] + seg + best_routes[j][ins_pos:]
                            # check capacity
                            dem_rj = sum(demands[n] for n in new_rj if 1 <= n <= n_cust)
                            if dem_rj > capacity:
                                continue
                            # build candidate solution
                            cand_routes = [list(r) for r in best_routes]
                            cand_routes[i] = new_ri
                            cand_routes[j] = new_rj
                            # remove empty routes
                            cand_routes = [r for r in cand_routes if r]
                            v, d = _eval_solution(cand_routes, D, demands, ready, due, service,
                                                   stations_set, capacity, bat_max, speed,
                                                   rrate, horizon, n_cust)
                            if v and d < best_dist - 1e-9:
                                best_routes = cand_routes
                                best_dist = d
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best_routes


def _ils_perturbation(routes, D, demands, ready, due, service, stations_set,
                      capacity, bat_max, speed, rrate, horizon, n_cust, rng):
    """Perturb: swap two customers between routes or reverse a segment."""
    routes = [list(r) for r in routes]
    op = rng.random()
    if op < 0.5 and len(routes) >= 2:
        # swap two customers from different routes
        ri = rng.randrange(len(routes))
        rj = rng.randrange(len(routes))
        if ri == rj:
            rj = (rj + 1) % len(routes)
        if len(routes[ri]) < 1 or len(routes[rj]) < 1:
            return routes
        pi = rng.randrange(len(routes[ri]))
        pj = rng.randrange(len(routes[rj]))
        a = routes[ri][pi]
        b = routes[rj][pj]
        routes[ri][pi] = b
        routes[rj][pj] = a
    elif op < 0.8:
        # reverse a segment in a random route
        if len(routes) == 0:
            return routes
        ri = rng.randrange(len(routes))
        if len(routes[ri]) >= 2:
            i = rng.randrange(len(routes[ri]))
            j = rng.randrange(len(routes[ri]))
            if i > j:
                i, j = j, i
            routes[ri][i:j+1] = routes[ri][i:j+1][::-1]
    else:
        # remove a customer and reinsert in random route
        if len(routes) == 0:
            return routes
        # find a customer
        all_cust = [n for r in routes for n in r if 1 <= n <= n_cust]
        if not all_cust:
            return routes
        c = rng.choice(all_cust)
        # remove from its route
        for ri in range(len(routes)):
            if c in routes[ri]:
                routes[ri].remove(c)
                break
        # insert into random route
        rj = rng.randrange(len(routes))
        if routes[rj]:
            ins = rng.randrange(len(routes[rj]) + 1)
            routes[rj].insert(ins, c)
        else:
            routes[rj] = [c]
    # remove empty routes
    routes = [r for r in routes if r]
    return routes


def _local_search(routes, D, demands, ready, due, service, stations_set,
                  capacity, bat_max, speed, rrate, horizon, n_cust, deadline):
    """Apply 2-opt per route + inter-route or-opt until deadline."""
    routes = [list(r) for r in routes]
    # 2-opt each route
    for i in range(len(routes)):
        if time.time() > deadline:
            break
        routes[i] = _two_opt(routes[i], D, demands, ready, due, service,
                              stations_set, capacity, bat_max, speed, rrate,
                              horizon, n_cust, max_passes=3)
    # inter-route or-opt
    if time.time() < deadline:
        routes = _or_opt_inter(routes, D, demands, ready, due, service,
                                stations_set, capacity, bat_max, speed, rrate,
                                horizon, n_cust, max_iters=15)
    return routes


def plan(instance):
    t_start = time.time()
    deadline = t_start + WALL_LIMIT

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    bat_max = instance["battery"]
    speed = instance["speed"]
    rrate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_cust = instance["n_customers"]
    stations_set = set(instance["stations"])
    customers = list(range(1, n_cust + 1))

    D = _dist_matrix(coords)

    # Multi-start ILS
    best_routes = None
    best_dist = float('inf')
    n_starts = 20
    per_start_budget = (deadline - time.time()) / n_starts

    for s in range(n_starts):
        t_now = time.time()
        if t_now > deadline - 2.0:
            break

        # Construct
        routes = _construct_all(customers, D, demands, ready, due, service,
                                 stations_set, capacity, bat_max, speed, rrate,
                                 horizon, n_cust, seed=s * 137 + 42)

        # Validate
        valid, dist = _eval_solution(routes, D, demands, ready, due, service,
                                      stations_set, capacity, bat_max, speed,
                                      rrate, horizon, n_cust)
        if valid:
            if dist < best_dist:
                best_dist = dist
                best_routes = [list(r) for r in routes]

        # ILS loop for this start
        rng = random.Random(s * 997 + 7)
        current = [list(r) for r in routes]
        cur_valid, cur_dist = _eval_solution(current, D, demands, ready, due, service,
                                              stations_set, capacity, bat_max, speed,
                                              rrate, horizon, n_cust)
        if not cur_valid:
            continue

        # Local search
        ls_deadline = min(deadline - 1.0, t_now + per_start_budget * 0.5)
        current = _local_search(current, D, demands, ready, due, service,
                                 stations_set, capacity, bat_max, speed, rrate,
                                 horizon, n_cust, ls_deadline)
        cur_valid, cur_dist = _eval_solution(current, D, demands, ready, due, service,
                                              stations_set, capacity, bat_max, speed,
                                              rrate, horizon, n_cust)
        if cur_valid and cur_dist < best_dist:
            best_dist = cur_dist
            best_routes = [list(r) for r in current]

        # ILS iterations
        ils_deadline = min(deadline - 1.0, t_now + per_start_budget)
        iters = 0
        while time.time() < ils_deadline and iters < 50:
            iters += 1
            perturbed = _ils_perturbation(current, D, demands, ready, due, service,
                                           stations_set, capacity, bat_max, speed,
                                           rrate, horizon, n_cust, rng)
            # quick local search
            ls_d = min(ils_deadline, time.time() + 0.5)
            improved = _local_search(perturbed, D, demands, ready, due, service,
                                      stations_set, capacity, bat_max, speed, rrate,
                                      horizon, n_cust, ls_d)
            v, d = _eval_solution(improved, D, demands, ready, due, service,
                                   stations_set, capacity, bat_max, speed,
                                   rrate, horizon, n_cust)
            if v and d <= cur_dist + 1e-9:
                current = improved
                cur_dist = d
                if d < best_dist:
                    best_dist = d
                    best_routes = [list(r) for r in current]
            else:
                # accept with small probability (simulated annealing style)
                if v and rng.random() < 0.1:
                    current = improved
                    cur_dist = d

    if best_routes is None:
        # Fallback: one customer per route
        best_routes = [[c] for c in customers]

    # Final validation
    valid, total_d = _eval_solution(best_routes, D, demands, ready, due, service,
                                     stations_set, capacity, bat_max, speed,
                                     rrate, horizon, n_cust)
    if not valid:
        best_routes = [[c] for c in customers]

    return best_routes
# EVOLVE-BLOCK-END

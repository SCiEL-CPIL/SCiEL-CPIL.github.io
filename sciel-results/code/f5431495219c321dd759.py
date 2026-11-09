"""EV routing with time windows + charging stations.
Constructive NN heuristic + 2-opt/relocate/or-opt local search, with a hard wall-clock guard.
"""
# EVOLVE-BLOCK-START
import time
import math
import random

_T0 = time.time()
_LIMIT = 135.0  # safety margin under the 150s kill
_EPS = 1e-9


def _dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _build(instance):
    coords = instance["coords"]
    n = len(coords)
    dmat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        row = dmat[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            v = math.hypot(xi - xj, yi - yj)
            row[j] = v
            dmat[j][i] = v
    return dmat


def _check_route(route, dmat, C):
    """Return (feasible, distance). route = list of customer/station ids (depot implicit)."""
    ready = C["ready"]; due = C["due"]; service = C["service"]
    demand = C["demand"]; stations = C["stations"]
    cap = C["capacity"]; bat = C["battery"]; speed = C["speed"]
    rate = C["recharge_rate"]; horizon = C["horizon"]
    n = C["n"]

    t = 0.0
    e = bat
    load = 0
    dist = 0.0
    prev = 0
    for node in route:
        d = dmat[prev][node]
        t += d / speed
        e -= d
        dist += d
        if node in stations:
            if t > horizon:
                return False, dist
            wait = (bat - e) * rate
            t += wait
            e = bat
        else:
            if e < -_EPS:
                return False, dist
            load += demand[node]
            if load > cap + _EPS:
                return False, dist
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + _EPS:
                return False, dist
            t += service[node]
        prev = node
    d = dmat[prev][0]
    t += d / speed
    e -= d
    dist += d
    if e < -_EPS:
        return False, dist
    if t > horizon + _EPS:
        return False, dist
    return True, dist


def _construct(instance, dmat, C):
    """Nearest-neighbour construction: repeatedly open a route and fill with feasible customers."""
    customers = list(range(1, C["n_customers"] + 1))
    routes = []
    while customers:
        # pick the customer with smallest due as seed (most constrained)
        seed = min(customers, key=lambda v: C["due"][v])
        customers.remove(seed)
        route = [seed]
        cur = seed
        load = C["demand"][seed]
        # walk forward adding nearest feasible customer
        while True:
            best = None
            best_d = None
            for v in customers:
                if load + C["demand"][v] > C["capacity"] + _EPS:
                    continue
                cand = route + [v]
                ok, _ = _check_route(cand, dmat, C)
                if ok:
                    dd = dmat[cur][v]
                    if best_d is None or dd < best_d:
                        best_d = dd
                        best = v
            if best is None:
                break
            customers.remove(best)
            route.append(best)
            cur = best
            load += C["demand"][best]
        routes.append(route)
    return routes


def _total_distance(routes, dmat):
    tot = 0.0
    for r in routes:
        if not r:
            continue
        prev = 0
        for node in r:
            tot += dmat[prev][node]
            prev = node
        tot += dmat[prev][0]
    return tot


def _relocate(routes, dmat, C):
    """Move a single node to a better position. Returns True if improved."""
    n_routes = len(routes)
    for i in range(n_routes):
        r = routes[i]
        for a in range(len(r)):
            node = r[a]
            # try inserting into every other route at every position
            for j in range(n_routes):
                if i == j:
                    continue
                rj = routes[j]
                for pos in range(len(rj) + 1):
                    cand_j = rj[:pos] + [node] + rj[pos:]
                    cand_i = r[:a] + r[a + 1:]
                    ok_i, _ = _check_route(cand_i, dmat, C)
                    ok_j, _ = _check_route(cand_j, dmat, C)
                    if ok_i and ok_j:
                        # distance change
                        old = 0.0
                        # recompute both affected routes fully (cheap enough)
                        new_tot = 0.0
                        for rr in (cand_i, cand_j):
                            if rr:
                                p = 0
                                for nd in rr:
                                    new_tot += dmat[p][nd]
                                    p = nd
                                new_tot += dmat[p][0]
                        # old contribution
                        old_tot = 0.0
                        for rr in (r, rj):
                            if rr:
                                p = 0
                                for nd in rr:
                                    old_tot += dmat[p][nd]
                                    p = nd
                                old_tot += dmat[p][0]
                        if new_tot < old_tot - _EPS:
                            routes[i] = cand_i
                            routes[j] = cand_j
                            return True
    return False


def _or_opt(routes, dmat, C):
    """Move a short segment (1-3 nodes) to a better position."""
    n_routes = len(routes)
    for i in range(n_routes):
        r = routes[i]
        L = len(r)
        for a in range(L):
            for seglen in (1, 2, 3):
                if a + seglen > L:
                    continue
                seg = r[a:a + seglen]
                cand_i = r[:a] + r[a + seglen:]
                for j in range(n_routes):
                    if i == j:
                        continue
                    rj = routes[j]
                    for pos in range(len(rj) + 1):
                        cand_j = rj[:pos] + seg + rj[pos:]
                        ok_i, _ = _check_route(cand_i, dmat, C)
                        ok_j, _ = _check_route(cand_j, dmat, C)
                        if ok_i and ok_j:
                            new_tot = 0.0
                            for rr in (cand_i, cand_j):
                                if rr:
                                    p = 0
                                    for nd in rr:
                                        new_tot += dmat[p][nd]
                                        p = nd
                                    new_tot += dmat[p][0]
                            old_tot = 0.0
                            for rr in (r, rj):
                                if rr:
                                    p = 0
                                    for nd in rr:
                                        old_tot += dmat[p][nd]
                                        p = nd
                                    old_tot += dmat[p][0]
                            if new_tot < old_tot - _EPS:
                                routes[i] = cand_i
                                routes[j] = cand_j
                                return True
    return False


def _two_opt_within(routes, dmat, C):
    """2-opt reversal within a single route."""
    improved = False
    for i in range(len(routes)):
        r = routes[i]
        L = len(r)
        for a in range(L - 1):
            for b in range(a + 1, L):
                cand = r[:a] + r[a:b + 1][::-1] + r[b + 1:]
                ok, _ = _check_route(cand, dmat, C)
                if ok:
                    old = 0.0
                    p = 0
                    for nd in r:
                        old += dmat[p][nd]
                        p = nd
                    old += dmat[p][0]
                    new = 0.0
                    p = 0
                    for nd in cand:
                        new += dmat[p][nd]
                        p = nd
                    new += dmat[p][0]
                    if new < old - _EPS:
                        routes[i] = cand
                        improved = True
    return improved


def _local_search(routes, dmat, C, deadline):
    best = _total_distance(routes, dmat)
    improved = True
    while improved and (time.time() - _T0) < deadline:
        improved = False
        if _two_opt_within(routes, dmat, C):
            improved = True
        if _relocate(routes, dmat, C):
            improved = True
        if _or_opt(routes, dmat, C):
            improved = True
        # re-evaluate total
        cur = _total_distance(routes, dmat)
        if cur < best:
            best = cur
    return best


def _merge_empty(routes):
    out = [r for r in routes if r]
    return out


def plan(instance):
    global _T0
    _T0 = time.time()
    random.seed(0)

    coords = instance["coords"]
    n = len(coords)
    C = {
        "coords": coords,
        "n": n,
        "ready": instance["ready"],
        "due": instance["due"],
        "service": instance["service"],
        "demand": instance["demand"],
        "stations": set(instance["stations"]),
        "capacity": instance["capacity"],
        "battery": instance["battery"],
        "speed": instance["speed"],
        "recharge_rate": instance["recharge_rate"],
        "horizon": instance["horizon"],
        "n_customers": instance["n_customers"],
    }
    dmat = _build(instance)

    # Fallback: one customer per route (always feasible).
    customers = list(range(1, C["n_customers"] + 1))
    customers.sort(key=lambda v: C["due"][v])
    fallback = [[v] for v in customers]

    deadline = _T0 + _LIMIT - 8.0  # leave margin for final validation

    # Constructive + local search, with a few random restarts if time allows.
    best_routes = None
    best_score = float("inf")

    # First attempt: deterministic construction
    routes = _construct(instance, dmat, C)
    score = _local_search(routes, dmat, C, deadline)
    best_routes = [list(r) for r in routes]
    best_score = score

    # Randomized restarts: shuffle customer order for construction
    restarts = 6
    for _ in range(restarts):
        if (time.time() - _T0) > deadline:
            break
        routes = _construct(instance, dmat, C)
        # randomize route order slightly by shuffling within a copy
        score = _local_search(routes, dmat, C, deadline)
        if score < best_score:
            best_score = score
            best_routes = [list(r) for r in routes]

    # Final validation; if invalid, fall back to the safe one-per-route solution.
    if best_routes is None:
        best_routes = fallback
    for r in best_routes:
        ok, _ = _check_route(r, dmat, C)
        if not ok:
            best_routes = fallback
            break

    # Guard against running over the limit.
    if (time.time() - _T0) > _LIMIT - 2.0:
        return fallback

    return best_routes
# EVOLVE-BLOCK-END

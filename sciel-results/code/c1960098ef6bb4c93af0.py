import time
import numpy as np
import random

_DEADLINE = 118.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _dist(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        dx = coords[i][0] - coords
        dy = coords[i][1] - coords
        D[i] = np.sqrt(dx * dx + dy * dy)
    return D


def _check_route(route, inst, D, cap, bat, speed, rate, horizon, service, ready, due, demand):
    t = 0.0
    b = float(bat)
    load = 0.0
    prev = 0
    for node in route:
        d = D[prev][node]
        b -= d
        t += d / speed
        if b < -1e-6:
            return False
        if node >= 1 and node <= inst["n_customers"]:
            t = max(t, ready[node])
            if t > due[node] + 1e-6:
                return False
            load += demand[node]
            t += service[node]
            if load > cap + 1e-6:
                return False
        elif node > inst["n_customers"]:
            need = (b - bat) * rate
            if need < -1e-6:
                need = 0.0
            t += need
            b = float(bat)
        prev = node
    d = D[prev][0]
    b -= d
    t += d / speed
    if b < -1e-6:
        return False
    if t > horizon + 1e-6:
        return False
    return True


def _route_cost(route, D):
    if not route:
        return 0.0
    return float(D[0][route[0]] + D[route[-1]][0] + sum(D[route[i]][route[i+1]] for i in range(len(route)-1)))


def _insert_cost(route, pos, node, D):
    if not route:
        return float(D[0][node] + D[node][0])
    a = 0 if pos == 0 else route[pos-1]
    b = route[pos] if pos < len(route) else 0
    return float(D[a][node] + D[node][b] - D[a][b])


def _merge_cost(r1, r2, D):
    if not r1:
        return _route_cost(r2, D)
    if not r2:
        return _route_cost(r1, D)
    c1 = r1 + r2
    c2 = r1 + list(reversed(r2))
    return min(_route_cost(c1, D), _route_cost(c2, D))


def _merge_routes(r1, r2, D):
    if not r1:
        return r2
    if not r2:
        return r1
    c1 = r1 + r2
    c2 = r1 + list(reversed(r2))
    if _route_cost(c1, D) <= _route_cost(c2, D):
        return c1
    return c2


def _feasible(r, inst, D, cap, bat, speed, rate, horizon, service, ready, due, demand):
    return _check_route(r, inst, D, cap, bat, speed, rate, horizon, service, ready, due, demand)


def _solve_instance(inst, seed):
    random.seed(seed)
    coords = inst["coords"]
    D = _dist(coords)
    nc = inst["n_customers"]
    stations = inst["stations"]
    cap = inst["capacity"]
    bat = inst["battery"]
    speed = inst["speed"]
    rate = inst["recharge_rate"]
    horizon = inst["horizon"]
    service = inst["service"]
    ready = inst["ready"]
    due = inst["due"]
    demand = inst["demands"]

    customers = list(range(1, nc + 1))

    # --- Construction: cheapest insertion ---
    routes = []
    placed = set()
    remaining = list(customers)
    # sort by due for tie-break / initial
    while remaining:
        # pick the customer with the smallest min insertion cost
        best = None
        for c in remaining:
            if c in placed:
                continue
            if not routes:
                cost = float(D[0][c] + D[c][0])
                ridx = 0
                pos = 0
            else:
                cost = float("inf")
                ridx = 0
                pos = 0
                for i, r in enumerate(routes):
                    # capacity check
                    load = sum(demand[x] for x in r)
                    if load + demand[c] > cap + 1e-6:
                        continue
                    for p in range(len(r) + 1):
                        cc = _insert_cost(r, p, c, D)
                        if cc < cost:
                            cost = cc
                            ridx = i
                            pos = p
            # feasibility of candidate
            if not routes:
                cand = [c]
            else:
                cand = routes[ridx][:pos] + [c] + routes[ridx][pos:]
            if _feasible(cand, inst, D, cap, bat, speed, rate, horizon, service, ready, due, demand):
                if best is None or cost < best[0]:
                    best = (cost, c, ridx, pos)
        if best is None:
            # fallback: place each remaining as its own route
            for c in remaining:
                if c not in placed:
                    routes.append([c])
                    placed.add(c)
            break
        cost, c, ridx, pos = best
        if not routes:
            routes.append([c])
        else:
            routes[ridx] = routes[ridx][:pos] + [c] + routes[ridx][pos:]
        placed.add(c)
        remaining.remove(c)

    # --- Local search ---
    def current_cost():
        return sum(_route_cost(r, D) for r in routes)

    best_routes = [list(r) for r in routes]
    best_cost = current_cost()

    def try_routes(new_routes):
        nonlocal best_routes, best_cost
        if any(len(r) == 0 for r in new_routes):
            new_routes = [r for r in new_routes if len(r) > 0]
        # feasibility
        for r in new_routes:
            if not _feasible(r, inst, D, cap, bat, speed, rate, horizon, service, ready, due, demand):
                return
        # all customers exactly once
        flat = [x for r in new_routes for x in r]
        if sorted(flat) != sorted(customers):
            return
        c = current_cost()
        if c < best_cost - 1e-9:
            best_cost = c
            best_routes = [list(r) for r in new_routes]

    def relocate():
        for i in range(len(routes)):
            for p in range(len(routes[i])):
                node = routes[i][p]
                rsrc = routes[i][:p] + routes[i][p+1:]
                for j in range(len(routes)):
                    rdst = routes[j]
                    for q in range(len(rdst) + 1):
                        new = rdst[:q] + [node] + rdst[q:]
                        nr = list(routes)
                        nr[i] = rsrc
                        nr[j] = new
                        try_routes(nr)

    def two_opt(r):
        n = len(r)
        if n < 2:
            return r
        best = r
        bestc = _route_cost(r, D)
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = r[:i] + list(reversed(r[i:j+1])) + r[j+1:]
                cc = _route_cost(cand, D)
                if cc < bestc - 1e-9:
                    bestc = cc
                    best = cand
        return best

    def or_opt():
        for i in range(len(routes)):
            r = routes[i]
            for a in range(len(r)):
                for b in range(a + 1, min(a + 3, len(r) + 1)):
                    seg = r[a:b]
                    rsrc = r[:a] + r[b:]
                    for j in range(len(routes)):
                        rdst = routes[j]
                        for q in range(len(rdst) + 1):
                            new = rdst[:q] + seg + rdst[q:]
                            nr = list(routes)
                            nr[i] = rsrc
                            nr[j] = new
                            try_routes(nr)

    def cross_exchange():
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                r1, r2 = routes[i], routes[j]
                for p in range(len(r1) + 1):
                    for q in range(len(r2) + 1):
                        nr1 = r1[:p] + r2[q:]
                        nr2 = r2[:q] + r1[p:]
                        nr = list(routes)
                        nr[i] = nr1
                        nr[j] = nr2
                        try_routes(nr)

    def merge():
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if _merge_cost(routes[i], routes[j], D) < best_cost - 1e-9:
                    nr = list(routes)
                    nr[i] = _merge_routes(routes[i], routes[j], D)
                    nr[j] = []
                    try_routes(nr)

    def station_insert():
        # try inserting a station into each route to improve feasibility headroom (rarely helps cost)
        pass

    moves = [relocate, two_opt_all, or_opt, cross_exchange, merge]

    def two_opt_all():
        for i in range(len(routes)):
            routes[i] = two_opt(routes[i])

    no_improve = 0
    while _elapsed() < _DEADLINE:
        improved = False
        for mv in moves:
            if _elapsed() > _DEADLINE:
                break
            before = best_cost
            mv()
            if best_cost < before - 1e-9:
                improved = True
        if not improved:
            no_improve += 1
            if no_improve >= 3:
                break
        else:
            no_improve = 0
        # sync routes to best
        routes = [list(r) for r in best_routes]

    return best_routes


def plan(instance):
    global _START
    _START = time.time()
    rng = np.random.default_rng()
    seed = int(rng.integers(0, 2**31 - 1))
    routes = _solve_instance(instance, seed)
    return routes

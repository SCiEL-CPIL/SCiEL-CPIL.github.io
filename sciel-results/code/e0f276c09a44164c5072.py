# EVOLVE-BLOCK-START
import numpy as np
import time, math, random

_DEADLINE = 145.0

def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _feasible(route, coords, demand, ready, due, service, capacity, battery_cap,
              speed, recharge_rate, horizon, stations):
    t = 0.0; bat = float(battery_cap); load = 0
    prev = 0
    for v in route:
        d = _dist(coords[prev], coords[v])
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return False
        if v in stations:
            t += (bat) * recharge_rate
            bat = float(battery_cap)
        else:
            if t > due[v] + 1e-9:
                return False
            if t < ready[v]:
                t = ready[v]
            t += service[v]
            load += demand[v]
            if load > capacity + 1e-9:
                return False
        prev = v
    d = _dist(coords[prev], coords[0])
    t += d / speed
    bat -= d
    if bat < -1e-9:
        return False
    if t > horizon + 1e-9:
        return False
    return True

def _route_dist(route, coords):
    if not route:
        return 0.0
    s = _dist(coords[0], coords[route[0]])
    for i in range(len(route)-1):
        s += _dist(coords[route[i]], coords[route[i+1]])
    s += _dist(coords[route[-1]], coords[0])
    return s

def _construct(coords, demand, ready, due, service, capacity, battery_cap,
               speed, recharge_rate, horizon, stations):
    n = len(coords)
    custs = list(range(1, n))
    custs.sort(key=lambda v: due[v])
    routes = []
    unassigned = set(custs)
    while unassigned:
        start = min(unassigned, key=lambda v: due[v])
        route = [start]
        unassigned.discard(start)
        load = demand[start]
        changed = True
        while changed:
            changed = False
            best_v = None; best_pos = None; best_gain = 1e18
            for v in unassigned:
                if load + demand[v] > capacity:
                    continue
                for pos in range(len(route)+1):
                    cand = route[:pos] + [v] + route[pos:]
                    if _feasible(cand, coords, demand, ready, due, service,
                                 capacity, battery_cap, speed, recharge_rate,
                                 horizon, stations):
                        gain = _route_dist(cand, coords) - _route_dist(route, coords)
                        if gain < best_gain:
                            best_gain = gain
                            best_v = v
                            best_pos = pos
            if best_v is not None:
                route = route[:best_pos] + [best_v] + route[best_pos:]
                unassigned.discard(best_v)
                load += demand[best_v]
                changed = True
        routes.append(route)
    return routes

def _try_2opt(route, coords, demand, ready, due, service, capacity, battery_cap,
              speed, recharge_rate, horizon, stations, cur_dist):
    n = len(route)
    if n < 2:
        return route, cur_dist
    best = route; best_d = cur_dist
    for i in range(n-1):
        for j in range(i+1, n):
            new = route[:i] + route[i:j+1][::-1] + route[j+1:]
            nd = _route_dist(new, coords)
            if nd < best_d - 1e-9 and _feasible(new, coords, demand, ready, due,
                                                service, capacity, battery_cap,
                                                speed, recharge_rate, horizon, stations):
                best = new; best_d = nd
    return best, best_d

def _try_relocate(route, coords, demand, ready, due, service, capacity, battery_cap,
                  speed, recharge_rate, horizon, stations, cur_dist):
    n = len(route)
    if n < 2:
        return route, cur_dist
    best = route; best_d = cur_dist
    for i in range(n):
        v = route[i]
        base = route[:i] + route[i+1:]
        for j in range(len(base)):
            new = base[:j] + [v] + base[j:]
            nd = _route_dist(new, coords)
            if nd < best_d - 1e-9 and _feasible(new, coords, demand, ready, due,
                                                service, capacity, battery_cap,
                                                speed, recharge_rate, horizon, stations):
                best = new; best_d = nd
    return best, best_d

def _try_oropt(route, coords, demand, ready, due, service, capacity, battery_cap,
               speed, recharge_rate, horizon, stations, cur_dist):
    n = len(route)
    best = route; best_d = cur_dist
    for L in (1,2,3):
        for i in range(n-L+1):
            seg = route[i:i+L]
            base = route[:i] + route[i+L:]
            for j in range(len(base)+1):
                new = base[:j] + seg + base[j:]
                if new == route:
                    continue
                nd = _route_dist(new, coords)
                if nd < best_d - 1e-9 and _feasible(new, coords, demand, ready, due,
                                                    service, capacity, battery_cap,
                                                    speed, recharge_rate, horizon, stations):
                    best = new; best_d = nd
    return best, best_d

def _try_cross(routes, coords, demand, ready, due, service, capacity, battery_cap,
               speed, recharge_rate, horizon, stations):
    best_routes = routes
    best_total = sum(_route_dist(r, coords) for r in routes)
    for a in range(len(routes)):
        for b in range(len(routes)):
            if a == b:
                continue
            ra = routes[a]; rb = routes[b]
            if len(ra) < 2 or len(rb) < 2:
                continue
            for i in range(1, len(ra)):
                for j in range(1, len(rb)):
                    newa = ra[:i] + rb[j:]
                    newb = rb[:j] + ra[i:]
                    if not newa or not newb:
                        continue
                    if _feasible(newa, coords, demand, ready, due, service,
                                 capacity, battery_cap, speed, recharge_rate,
                                 horizon, stations) and \
                       _feasible(newb, coords, demand, ready, due, service,
                                 capacity, battery_cap, speed, recharge_rate,
                                 horizon, stations):
                        nt = _route_dist(newa, coords) + _route_dist(newb, coords)
                        if nt < best_total - 1e-9:
                            best_total = nt
                            new_routes = list(routes)
                            new_routes[a] = newa
                            new_routes[b] = newb
                            best_routes = new_routes
    return best_routes

def _local_search(routes, coords, demand, ready, due, service, capacity, battery_cap,
                  speed, recharge_rate, horizon, stations, deadline):
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(routes)):
            if time.time() >= deadline:
                return routes
            r = routes[idx]
            if not r:
                continue
            d0 = _route_dist(r, coords)
            r, d1 = _try_2opt(r, coords, demand, ready, due, service, capacity,
                              battery_cap, speed, recharge_rate, horizon, stations, d0)
            if d1 < d0 - 1e-9:
                routes[idx] = r; improved = True
                continue
            r, d1 = _try_oropt(r, coords, demand, ready, due, service, capacity,
                               battery_cap, speed, recharge_rate, horizon, stations, d0)
            if d1 < d0 - 1e-9:
                routes[idx] = r; improved = True
                continue
            r, d1 = _try_relocate(r, coords, demand, ready, due, service, capacity,
                                  battery_cap, speed, recharge_rate, horizon, stations, d0)
            if d1 < d0 - 1e-9:
                routes[idx] = r; improved = True
        if time.time() >= deadline:
            break
        nr = _try_cross(routes, coords, demand, ready, due, service, capacity,
                        battery_cap, speed, recharge_rate, horizon, stations)
        if nr != routes:
            routes = nr; improved = True
    return routes

def _solve_one(instance, deadline):
    coords = instance["coords"]
    demand = instance["demand"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    routes = _construct(coords, demand, ready, due, service, capacity, battery_cap,
                        speed, recharge_rate, horizon, stations)
    routes = _local_search(routes, coords, demand, ready, due, service, capacity,
                           battery_cap, speed, recharge_rate, horizon, stations,
                           deadline)
    return routes

def plan(instance):
    t0 = time.time()
    deadline = t0 + _DEADLINE
    routes = _solve_one(instance, deadline)
    return routes
# EVOLVE-BLOCK-END

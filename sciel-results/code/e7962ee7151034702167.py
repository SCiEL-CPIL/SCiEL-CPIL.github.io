"""Improved EV routing: multi-customer route construction with feasibility checks."""
# EVOLVE-BLOCK-START
import numpy as np


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def plan(instance):
    coords = np.asarray(instance["coords"], dtype=float)
    n = instance["n_customers"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    depot = 0

    remaining = list(range(1, n + 1))
    # priority: earlier due first
    remaining.sort(key=lambda v: (due[v], v))

    routes = []

    def try_add(route_nodes, cur, time, batt, load, candidates):
        """Try to append best feasible customer to route. Return (node, new_time, new_batt) or None."""
        best = None
        for v in candidates:
            d = _dist(coords[cur], coords[v])
            t_arr = time + d / speed
            b_arr = batt - d
            if b_arr < -1e-9:
                continue
            t = max(t_arr, ready[v])
            if t > due[v] + 1e-9:
                continue
            t_serv = t + service[v]
            # check return to depot from v (battery and horizon)
            d_ret = _dist(coords[v], coords[depot])
            b_ret = b_arr - d_ret
            if b_ret < -1e-9:
                continue
            t_ret = t_serv + d_ret / speed
            if t_ret > horizon + 1e-9:
                continue
            # capacity
            if load + demands[v] > capacity + 1e-9:
                continue
            # prefer nearest (smallest d)
            if best is None or d < best[0]:
                best = (d, v, t_serv, b_arr)
        return best

    while remaining:
        route_nodes = []
        cur = depot
        time = 0.0
        batt = float(battery)
        load = 0
        candidates = list(remaining)
        while True:
            best = try_add(route_nodes, cur, time, batt, load, candidates)
            if best is None:
                break
            d, v, t_serv, b_arr = best
            route_nodes.append(v)
            candidates.remove(v)
            remaining.remove(v)
            cur = v
            time = t_serv
            batt = b_arr
            load += demands[v]
        if route_nodes:
            routes.append(route_nodes)

    return routes
# EVOLVE-BLOCK-END

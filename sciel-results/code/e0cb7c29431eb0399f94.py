"""EV routing with time windows: constructive + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time

# Global deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 12.0  # seconds before hard limit

def _check_deadline():
    if _DEADLINE is not None and time.time() > _DEADLINE - _DEADLINE_MARGIN:
        return True
    return False


def _dist_matrix(coords):
    n = len(coords)
    m = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            m[i, j] = d
            m[j, i] = d
    return m


def _route_info(route, D, demands, ready, due, service,
                battery_cap, recharge_rate, horizon, speed, station_set):
    """Simulate a route; return (total_dist, feasible, total_time)."""
    t = 0.0
    bat = battery_cap
    load = 0.0
    total_d = 0.0
    prev = 0  # depot
    for node in route:
        d = D[prev, node]
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return total_d + d, False, t
        total_d += d
        if node in station_set:
            charge = (battery_cap - bat) * recharge_rate
            t += charge
            bat = battery_cap
        else:
            # customer
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return total_d, False, t
            t += service[node]
            load += demands[node]
        prev = node
    # return to depot
    d = D[prev, 0]
    t += d / speed
    bat -= d
    if bat < -1e-9:
        return total_d + d, False, t
    total_d += d
    if t > horizon + 1e-9:
        return total_d, False, t
    if load > 100 + 1e-9:
        return total_d, False, t
    return total_d, True, t


def _initial_solution(instance):
    coords = np.array(instance["coords"])
    n = len(coords)
    D = _dist_matrix(coords)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    cap = instance["capacity"]
    bat_cap = instance["battery"]
    rr = instance["recharge_rate"]
    horizon = instance["horizon"]
    speed = instance["speed"]
    stations = set(instance["stations"])
    n_cust = instance["n_customers"]

    # sort customers by due time
    cust_order = sorted(range(1, n_cust + 1), key=lambda v: due[v])

    routes = []
    unassigned = set(cust_order)

    # Greedy insertion: build routes by inserting customers into existing routes
    # or opening new ones, respecting battery/time/capacity.
    while unassigned:
        # try to insert each unassigned customer into an existing route
        best_key = None
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')
        for ridx, route in enumerate(routes):
            for pos in range(len(route) + 1):
                # find candidate customer
                for c in list(unassigned):
                    new_route = route[:pos] + [c] + route[pos:]
                    td, feas, _ = _route_info(new_route, D, demands, ready, due,
                                              service, bat_cap, rr, horizon,
                                              speed, stations)
                    if feas:
                        # cost = increase in total distance
                        # approximate: D[prev,c]+D[c,next] - D[prev,next]
                        prev_node = route[pos - 1] if pos > 0 else 0
                        next_node = route[pos] if pos < len(route) else 0
                        inc = D[prev_node, c] + D[c, next_node] - D[prev_node, next_node]
                        if inc < best_cost:
                            best_cost = inc
                            best_route_idx = ridx
                            best_pos = pos
                            best_key = c
        if best_route_idx >= 0:
            routes[best_route_idx].insert(best_pos, best_key)
            unassigned.discard(best_key)
        else:
            # open new route with the customer with earliest due
            c = min(unassigned, key=lambda v: due[v])
            routes.append([c])
            unassigned.discard(c)

    return routes, D, demands, ready, due, service, bat_cap, rr, horizon, speed, stations


def _local_search(routes, D, demands, ready, due, service,
                  bat_cap, rr, horizon, speed, stations, deadline):
    """Battery-aware local search with relocate, swap, 2-opt, or-opt."""
    n_routes = len(routes)
    best_total = sum(_route_info(r, D, demands, ready, due, service,
                                 bat_cap, rr, horizon, speed, stations)[0]
                     for r in routes)
    no_improve = 0
    max_no_improve = 50

    while time.time() < deadline:
        improved = False
        # Relocate
        for i in range(n_routes):
            if time.time() > deadline:
                break
            ri = list(routes[i])
            for pos in range(len(ri)):
                c = ri[pos]
                ri2 = ri[:pos] + ri[pos + 1:]
                for j in range(n_routes):
                    if i == j:
                        continue
                    rj = list(routes[j])
                    for pos2 in range(len(rj) + 1):
                        new_j = rj[:pos2] + [c] + rj[pos2:]
                        td_i, f_i, _ = _route_info(ri2, D, demands, ready, due,
                                                   service, bat_cap, rr, horizon,
                                                   speed, stations)
                        td_j, f_j, _ = _route_info(new_j, D, demands, ready, due,
                                                   service, bat_cap, rr, horizon,
                                                   speed, stations)
                        if f_i and f_j:
                            new_total = td_i + td_j
                            if new_total < best_total - 1e-9:
                                routes[i] = ri2
                                routes[j] = new_j
                                best_total = new_total
                                improved = True
                                no_improve = 0
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
        for i in range(n_routes):
            if time.time() > deadline:
                break
            ri = list(routes[i])
            for pos in range(len(ri)):
                c1 = ri[pos]
                for j in range(i + 1, n_routes):
                    rj = list(routes[j])
                    for pos2 in range(len(rj)):
                        c2 = rj[pos2]
                        new_i = ri[:pos] + [c2] + ri[pos + 1:]
                        new_j = rj[:pos2] + [c1] + rj[pos2 + 1:]
                        td_i, f_i, _ = _route_info(new_i, D, demands, ready, due,
                                                   service, bat_cap, rr, horizon,
                                                   speed, stations)
                        td_j, f_j, _ = _route_info(new_j, D, demands, ready, due,
                                                   service, bat_cap, rr, horizon,
                                                   speed, stations)
                        if f_i and f_j:
                            new_total = td_i + td_j
                            if new_total < best_total - 1e-9:
                                routes[i] = new_i
                                routes[j] = new_j
                                best_total = new_total
                                improved = True
                                no_improve = 0
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # 2-opt within each route
        for i in range(n_routes):
            if time.time() > deadline:
                break
            ri = routes[i]
            if len(ri) < 2:
                continue
            for a in range(len(ri) - 1):
                for b in range(a + 1, len(ri)):
                    new_ri = ri[:a] + ri[a:b + 1][::-1] + ri[b + 1:]
                    td_new, f_new, _ = _route_info(new_ri, D, demands, ready, due,
                                                   service, bat_cap, rr, horizon,
                                                   speed, stations)
                    td_old, f_old, _ = _route_info(ri, D, demands, ready, due,
                                                   service, bat_cap, rr, horizon,
                                                   speed, stations)
                    if f_new and (not f_old or td_new < td_old - 1e-9):
                        routes[i] = new_ri
                        best_total = sum(_route_info(r, D, demands, ready, due,
                                                     service, bat_cap, rr, horizon,
                                                     speed, stations)[0]
                                         for r in routes)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # Or-opt: move a segment of 1-3 nodes
        for i in range(n_routes):
            if time.time() > deadline:
                break
            ri = routes[i]
            for seg_len in [1, 2, 3]:
                if len(ri) < seg_len:
                    continue
                for a in range(len(ri) - seg_len + 1):
                    segment = ri[a:a + seg_len]
                    ri2 = ri[:a] + ri[a + seg_len:]
                    for j in range(n_routes):
                        if i == j:
                            continue
                        rj = list(routes[j])
                        for pos2 in range(len(rj) + 1):
                            new_j = rj[:pos2] + segment + rj[pos2:]
                            td_i, f_i, _ = _route_info(ri2, D, demands, ready, due,
                                                       service, bat_cap, rr, horizon,
                                                       speed, stations)
                            td_j, f_j, _ = _route_info(new_j, D, demands, ready, due,
                                                       service, bat_cap, rr, horizon,
                                                       speed, stations)
                            if f_i and f_j:
                                new_total = td_i + td_j
                                if new_total < best_total - 1e-9:
                                    routes[i] = ri2
                                    routes[j] = new_j
                                    best_total = new_total
                                    improved = True
                                    no_improve = 0
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        no_improve += 1
        if no_improve > max_no_improve:
            break

    return routes, best_total


def plan(instance):
    global _DEADLINE
    _DEADLINE = time.time() + 150.0 - 5.0  # 145s effective

    # Solve single instance
    routes, D, demands, ready, due, service, bat_cap, rr, horizon, speed, stations = _initial_solution(instance)

    # Local search
    deadline = _DEADLINE - _DEADLINE_MARGIN
    routes, _ = _local_search(routes, D, demands, ready, due, service,
                              bat_cap, rr, horizon, speed, stations, deadline)

    # Verify all customers are served
    served = set()
    for r in routes:
        for v in r:
            if v > 0 and v <= instance["n_customers"]:
                served.add(v)
    expected = set(range(1, instance["n_customers"] + 1))
    if served != expected:
        # Fallback: one customer per route
        routes = [[v] for v in range(1, instance["n_customers"] + 1)]

    return routes
# EVOLVE-BLOCK-END

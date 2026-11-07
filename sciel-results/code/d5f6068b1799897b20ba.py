"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
            dist[i, j] = d
            dist[j, i] = d

    # Precompute feasible visit order for a single customer: check if depot->c->depot is feasible
    def is_single_feasible(c):
        # depot -> c
        d1 = dist[0, c]
        if d1 > battery_cap:
            return False
        t1 = d1 / speed
        # at customer
        t_arr = t1
        t_start = max(t_arr, ready[c])
        if t_start > due[c]:
            return False
        t_after_service = t_start + service[c]
        # c -> depot
        d2 = dist[c, 0]
        if d1 + d2 > battery_cap:
            return False
        t_return = t_after_service + d2 / speed
        if t_return > horizon:
            return False
        return True

    # Check if a route (list of node ids, no depot) is feasible
    def check_route(route):
        if not route:
            return True, 0.0
        total_dist = 0.0
        t = 0.0
        bat = battery_cap
        # depot -> first
        for i, node in enumerate(route):
            prev = 0 if i == 0 else route[i - 1]
            d = dist[prev, node]
            total_dist += d
            bat -= d
            if bat < 0:
                return False, total_dist
            t += d / speed
            if node in stations:
                # recharge
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
            else:
                # customer
                t = max(t, ready[node])
                if t > due[node]:
                    return False, total_dist
                t += service[node]
        # last -> depot
        d = dist[route[-1], 0]
        total_dist += d
        bat -= d
        if bat < 0:
            return False, total_dist
        t += d / speed
        if t > horizon:
            return False, total_dist
        return True, total_dist

    # Compute total distance of a list of routes
    def total_distance(routes):
        total = 0.0
        for r in routes:
            if not r:
                continue
            total += dist[0, r[0]]
            for i in range(len(r) - 1):
                total += dist[r[i], r[i + 1]]
            total += dist[r[-1], 0]
        return total

    # Check total demand
    def route_demand(route):
        return sum(demands[n] for n in route if n not in stations)

    # Initial construction: one customer per route
    customers = list(range(1, n_customers + 1))
    # Filter out infeasible singles? Shouldn't happen, but safety
    routes = []
    for c in customers:
        if is_single_feasible(c):
            routes.append([c])
        else:
            # Fallback: still add, but this will be invalid. 
            # Given constraints, singles should be feasible.
            routes.append([c])

    # Sort customers by due time for better merging?
    # Clarke-Wright Savings
    # Calculate savings for merging two routes [a] and [b]: saving = dist(0,a) + dist(0,b) - dist(a,b)
    # But we must maintain time windows and battery.
    
    # Let's use a simpler greedy merging:
    # Sort customers by angle or due time.
    # Try to insert each customer into an existing route if feasible.
    
    # First, sort customers by due time (earliest first)
    customers.sort(key=lambda c: due[c])
    
    # Start with empty routes
    routes = []
    for c in customers:
        best_route = None
        best_pos = -1
        best_cost = float('inf')
        
        # Try inserting into existing routes
        for i, r in enumerate(routes):
            for j in range(len(r) + 1):
                new_route = r[:j] + [c] + r[j:]
                # Check demand
                if route_demand(new_route) > capacity:
                    continue
                # Check feasibility
                feas, d = check_route(new_route)
                if feas:
                    # Cost is the increase in distance
                    old_d = total_distance([r])
                    new_d = total_distance([new_route])
                    cost = new_d - old_d
                    if cost < best_cost:
                        best_cost = cost
                        best_route = i
                        best_pos = j
        
        if best_route is not None:
            r = routes[best_route]
            r.insert(best_pos, c)
        else:
            routes.append([c])

    # Local Search: 2-opt and Or-opt
    def try_2opt(routes):
        improved = False
        for i in range(len(routes)):
            r = routes[i]
            if len(r) < 2:
                continue
            n = len(r)
            for a in range(n - 1):
                for b in range(a + 1, n):
                    # Reverse r[a..b]
                    new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                    if new_r == r:
                        continue
                    feas, d_new = check_route(new_r)
                    if not feas:
                        continue
                    d_old = total_distance([r])
                    if d_new < d_old - 1e-9:
                        routes[i] = new_r
                        improved = True
        return improved

    def try_oropt(routes):
        improved = False
        for i in range(len(routes)):
            r = routes[i]
            n = len(r)
            for length in [1, 2, 3]:
                for a in range(n):
                    if a + length > n:
                        continue
                    segment = r[a:a+length]
                    rest = r[:a] + r[a+length:]
                    for b in range(len(rest) + 1):
                        new_r = rest[:b] + segment + rest[b:]
                        if new_r == r:
                            continue
                        feas, d_new = check_route(new_r)
                        if not feas:
                            continue
                        d_old = total_distance([r])
                        if d_new < d_old - 1e-9:
                            routes[i] = new_r
                            improved = True
                            # Restart loop for this route
                            break
                    if improved:
                        break
                if improved:
                    break
        return improved

    # Inter-route: move a customer from one route to another
    def try_reassign(routes):
        improved = False
        for i in range(len(routes)):
            r_i = routes[i]
            for k in range(len(r_i)):
                c = r_i[k]
                # Remove c
                r_i_new = r_i[:k] + r_i[k+1:]
                feas_i, d_i = check_route(r_i_new)
                if not feas_i:
                    continue
                # Try inserting into other routes
                for j in range(len(routes)):
                    if i == j:
                        continue
                    r_j = routes[j]
                    for pos in range(len(r_j) + 1):
                        r_j_new = r_j[:pos] + [c] + r_j[pos:]
                        if route_demand(r_j_new) > capacity:
                            continue
                        feas_j, d_j = check_route(r_j_new)
                        if not feas_j:
                            continue
                        # Check total distance change
                        d_old = total_distance([r_i, r_j])
                        d_new = total_distance([r_i_new, r_j_new])
                        if d_new < d_old - 1e-9:
                            routes[i] = r_i_new
                            routes[j] = r_j_new
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        return improved

    # Run local search until no improvement or deadline
    while time.time() < deadline:
        imp = try_2opt(routes)
        if imp:
            continue
        imp = try_oropt(routes)
        if imp:
            continue
        imp = try_reassign(routes)
        if imp:
            continue
        break

    # Remove empty routes
    routes = [r for r in routes if r]
    
    return routes
# EVOLVE-BLOCK-END

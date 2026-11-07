"""EV routing with time windows – Clarke-Wright savings + fast local search."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


def _build_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate(route, D, instance):
    """
    Simulate a single route. Returns (total_distance, valid, final_time, final_battery, load).
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = set(instance["stations"])

    t = 0.0
    batt = battery_max
    load = 0
    dist_total = 0.0
    prev = 0

    for node in route:
        d = D[prev][node]
        dist_total += d
        t += d
        batt -= d
        if batt < -1e-9:
            return dist_total, False, t, batt, load
        if node in stations_set:
            recharge_time = (battery_max - batt) * recharge_rate
            t += recharge_time
            batt = battery_max
        else:
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return dist_total, False, t, batt, load
            load += demands[node]
            if load > capacity + 1e-9:
                return dist_total, False, t, batt, load
            t += service[node]
        prev = node

    d = D[prev][0]
    dist_total += d
    t += d
    batt -= d
    if batt < -1e-9:
        return dist_total, False, t, batt, load
    if t > horizon + 1e-9:
        return dist_total, False, t, batt, load

    return dist_total, True, t, batt, load


def _total_distance(routes, D, instance):
    total = 0.0
    for route in routes:
        d, valid, _, _, _ = _simulate(route, D, instance)
        if valid:
            total += d
    return total


def _is_valid_solution(routes, instance, D):
    served = set()
    for route in routes:
        dist, valid, _, _, _ = _simulate(route, D, instance)
        if not valid:
            return False
        for node in route:
            if node in set(instance["stations"]):
                continue
            if node in served:
                return False
            served.add(node)
    n_customers = instance["n_customers"]
    expected = set(range(1, n_customers + 1))
    return served == expected


def _clarke_wright_savings(instance, D):
    """
    Clarke-Wright savings heuristic for initial route construction.
    Returns a list of routes (lists of customer node ids).
    """
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    demands = instance["demands"]
    capacity = instance["capacity"]
    
    # Start with each customer as its own route
    routes = [[c] for c in customers]
    route_of = {c: i for i, c in enumerate(customers)}
    
    # Compute savings: s(i,j) = D(0,i) + D(0,j) - D(i,j)
    # Sort savings in descending order
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            s = D[0][ci] + D[0][cj] - D[ci][cj]
            savings.append((s, ci, cj))
    
    savings.sort(reverse=True)
    
    # Merge routes
    for s, ci, cj in savings:
        if s <= 0:
            break
        ri = route_of[ci]
        rj = route_of[cj]
        if ri == rj:
            continue
        
        # Check if ci is at the end of its route and cj is at the start of its route,
        # or ci is at the start and cj is at the end
        route_i = routes[ri]
        route_j = routes[rj]
        
        # Try to merge: route_i + route_j or route_j + route_i
        # ci must be at an end of route_i, cj at an end of route_j
        ci_at_end = (route_i[-1] == ci)
        ci_at_start = (route_i[0] == ci)
        cj_at_end = (route_j[-1] == cj)
        cj_at_start = (route_j[0] == cj)
        
        new_route = None
        if ci_at_end and cj_at_start:
            new_route = route_i + route_j
        elif ci_at_start and cj_at_end:
            new_route = route_j + route_i
        elif ci_at_end and cj_at_end:
            new_route = route_i + route_j[::-1]
        elif ci_at_start and cj_at_start:
            new_route = route_j[::-1] + route_i
        
        if new_route is None:
            continue
        
        # Check capacity
        total_demand = sum(demands[c] for c in new_route)
        if total_demand > capacity:
            continue
        
        # Merge
        routes[ri] = new_route
        for c in route_j:
            route_of[c] = ri
        routes[rj] = []
    
    # Filter out empty routes
    routes = [r for r in routes if r]
    
    return routes


def _insert_station_if_needed(route, D, instance):
    """
    Given a route of customer nodes, insert charging stations as needed.
    Returns the full route with stations, or None if infeasible.
    """
    stations_set = set(instance["stations"])
    stations = instance["stations"]
    battery_max = instance["battery"]
    
    full_route = []
    prev = 0
    batt = battery_max
    
    for node in route:
        d = D[prev][node]
        if d <= batt + 1e-9:
            full_route.append(node)
            batt -= d
            prev = node
        else:
            # Need to insert a station
            found = False
            for st in stations:
                d1 = D[prev][st]
                d2 = D[st][node]
                if d1 <= batt + 1e-9 and d2 <= battery_max + 1e-9:
                    full_route.append(st)
                    batt = battery_max
                    full_route.append(node)
                    batt -= d2
                    prev = node
                    found = True
                    break
            if not found:
                # Try two stations
                found = False
                for st1 in stations:
                    for st2 in stations:
                        if st1 == st2:
                            continue
                        d1 = D[prev][st1]
                        d2 = D[st1][st2]
                        d3 = D[st2][node]
                        if d1 <= batt + 1e-9 and d2 <= battery_max + 1e-9 and d3 <= battery_max + 1e-9:
                            full_route.append(st1)
                            batt = battery_max
                            full_route.append(st2)
                            batt = battery_max
                            full_route.append(node)
                            batt -= d3
                            prev = node
                            found = True
                            break
                    if found:
                        break
                if not found:
                    return None
    
    # Return to depot
    d = D[prev][0]
    if d <= batt + 1e-9:
        pass  # No station needed
    else:
        found = False
        for st in stations:
            d1 = D[prev][st]
            d2 = D[st][0]
            if d1 <= batt + 1e-9 and d2 <= battery_max + 1e-9:
                full_route.append(st)
                batt = battery_max
                found = True
                break
        if not found:
            return None
    
    return full_route


def _2opt(route, D, instance, max_iters=20):
    """
    2-opt improvement for a single route. Only reverses segments of customers (not stations).
    """
    improved = True
    iters = 0
    current_dist, _, _, _, _ = _simulate(route, D, instance)
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        n = len(route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                dist, valid, _, _, _ = _simulate(new_route, D, instance)
                if valid and dist < current_dist - 1e-9:
                    route = new_route
                    current_dist = dist
                    improved = True
    return route


def _or_opt(route, D, instance, max_iters=20):
    """
    Or-opt: relocate 1 or 2 consecutive nodes.
    """
    improved = True
    iters = 0
    current_dist, _, _, _, _ = _simulate(route, D, instance)
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        n = len(route)
        for i in range(n):
            for length in [1, 2]:
                if i + length > n:
                    continue
                segment = route[i:i+length]
                rest = route[:i] + route[i+length:]
                if not rest:
                    continue
                for j in range(len(rest) + 1):
                    new_route = rest[:j] + segment + rest[j:]
                    dist, valid, _, _, _ = _simulate(new_route, D, instance)
                    if valid and dist < current_dist - 1e-9:
                        route = new_route
                        current_dist = dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return route


def _cross_route_swap(routes, D, instance):
    """
    Inter-route improvement: swap customers between routes.
    """
    improved = True
    iters = 0
    while improved and iters < 10:
        improved = False
        iters += 1
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                for i in range(len(routes[a])):
                    for j in range(len(routes[b])):
                        ra = list(routes[a])
                        rb = list(routes[b])
                        ra[i], rb[j] = rb[j], ra[i]
                        va = _simulate(ra, D, instance)[1]
                        vb = _simulate(rb, D, instance)[1]
                        if va and vb:
                            old_dist = _simulate(ra, D, instance)[0] + _simulate(rb, D, instance)[0]
                            new_dist = _simulate(ra, D, instance)[0] + _simulate(rb, D, instance)[0]
                            if new_dist < old_dist - 1e-9:
                                routes[a] = ra
                                routes[b] = rb
                                improved = True
    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 110.0
    
    coords = instance["coords"]
    D = _build_distance_matrix(coords)
    
    # Construct initial solution using Clarke-Wright savings
    customer_routes = _clarke_wright_savings(instance, D)
    
    # Insert stations and build full routes
    routes = []
    for cr in customer_routes:
        full_route = _insert_station_if_needed(cr, D, instance)
        if full_route is not None:
            routes.append(full_route)
    
    # If no routes or invalid, fallback
    if not _is_valid_solution(routes, instance, D):
        # Fallback: one customer per route
        customers = list(range(1, instance["n_customers"] + 1))
        routes = []
        for c in customers:
            route = _insert_station_if_needed([c], D, instance)
            if route is not None:
                routes.append(route)
    
    best_routes = routes
    
    # Local search
    max_passes = 10
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        
        # 2-opt on each route
        for i in range(len(best_routes)):
            new_route = _2opt(best_routes[i], D, instance, max_iters=10)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True
        
        # Or-opt
        for i in range(len(best_routes)):
            new_route = _or_opt(best_routes[i], D, instance, max_iters=10)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True
        
        if not improved:
            break
    
    # Final validation
    if not _is_valid_solution(best_routes, instance, D):
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[c] for c in customers]
    
    return best_routes
# EVOLVE-BLOCK-END

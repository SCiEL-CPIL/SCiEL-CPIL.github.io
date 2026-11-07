import math
import time
import random
import numpy as np

_EPS = 1e-9

def _precompute(instance):
    """Precompute all matrices and lookup arrays for fast route evaluation."""
    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = n_cust + len(stations) + 1
    cust_ids = list(range(1, n_cust + 1))
    station_ids = set(stations)
    
    cx = coords[:, 0]
    cy = coords[:, 1]
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
            dist[i, j] = d
            dist[j, i] = d
    
    # Lookup arrays: 0 = depot, 1..n_cust = customers, rest = stations
    is_customer = np.zeros(n_nodes, dtype=bool)
    for c in cust_ids:
        is_customer[c] = True
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
    
    return {
        "coords": coords,
        "demands": demands,
        "ready": ready,
        "due": due,
        "service": service,
        "n_cust": n_cust,
        "stations": stations,
        "capacity": capacity,
        "battery_max": battery_max,
        "speed": speed,
        "recharge_rate": recharge_rate,
        "horizon": horizon,
        "n_nodes": n_nodes,
        "cust_ids": cust_ids,
        "station_ids": station_ids,
        "dist": dist,
        "is_customer": is_customer,
        "is_station": is_station,
    }

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin for 150s kill

    P = _precompute(instance)
    coords = P["coords"]
    demands = P["demands"]
    ready = P["ready"]
    due = P["due"]
    service = P["service"]
    n_cust = P["n_cust"]
    stations = P["stations"]
    capacity = P["capacity"]
    battery_max = P["battery_max"]
    speed = P["speed"]
    recharge_rate = P["recharge_rate"]
    horizon = P["horizon"]
    n_nodes = P["n_nodes"]
    cust_ids = P["cust_ids"]
    station_ids = P["station_ids"]
    dist = P["dist"]
    is_customer = P["is_customer"]
    is_station = P["is_station"]

    def check_route(route):
        """
        Returns (total_dist, valid, final_time, final_battery)
        """
        if not route:
            return 0.0, True, 0.0, battery_max
        
        total_dist = 0.0
        t = 0.0
        b = battery_max
        load = 0.0
        prev = 0
        
        # Check capacity
        for node in route:
            if is_customer[node]:
                load += demands[node]
        if load > capacity + _EPS:
            return float('inf'), False, 0.0, 0.0

        for node in route:
            d = dist[prev, node]
            total_dist += d
            t += d / speed
            b -= d
            
            if b < -_EPS:
                return float('inf'), False, 0.0, 0.0
            
            if is_station[node]:
                # Recharge
                t += (battery_max - b) * recharge_rate
                b = battery_max
            elif is_customer[node]:
                t = max(t, ready[node])
                if t > due[node] + _EPS:
                    return float('inf'), False, 0.0, 0.0
                t += service[node]
            
            prev = node
            
        # Return to depot
        d = dist[prev, 0]
        total_dist += d
        t += d / speed
        b -= d
        
        if b < -_EPS:
            return float('inf'), False, 0.0, 0.0
            
        if t > horizon + _EPS:
            return float('inf'), False, 0.0, 0.0
            
        return total_dist, True, t, b

    def try_insert_station(route):
        """
        If route is infeasible due to battery, try inserting a station.
        Returns (new_route, new_cost) or (None, inf)
        """
        total_dist, valid, t, b = check_route(route)
        if valid:
            return route, total_dist
        
        best_route = None
        best_cost = float('inf')
        
        # Identify where battery drops below 0 or time windows are violated?
        # Simple heuristic: try inserting each station at each position
        for s in stations:
            for i in range(len(route) + 1):
                cand = route[:i] + [s] + route[i:]
                c, v, _, _ = check_route(cand)
                if v and c < best_cost:
                    best_cost = c
                    best_route = cand
                    
        if best_route:
            return best_route, best_cost
        return None, float('inf')

    # --- Construction: Clustering + Greedy Insertion ---
    
    # Sort customers by angle from depot for initial clustering
    depot_x, depot_y = coords[0, 0], coords[0, 1]
    angles = []
    for c in cust_ids:
        dx = coords[c, 0] - depot_x
        dy = coords[c, 1] - depot_y
        angles.append((math.atan2(dy, dx), c))
    angles.sort(key=lambda x: x[0])
    
    # Initial routes: group customers by angle sector
    n_routes = 4
    routes = [[] for _ in range(n_routes)]
    for idx, (_, c) in enumerate(angles):
        routes[idx % n_routes].append(c)
    
    # Greedy insertion: try to insert each customer into best feasible route
    # First, build initial routes with nearest-neighbor heuristic per cluster
    for ri in range(n_routes):
        if not routes[ri]:
            continue
        # Sort by nearest neighbor from depot
        remaining = list(routes[ri])
        current_route = []
        prev = 0
        while remaining:
            best_c = min(remaining, key=lambda c: dist[prev, c])
            current_route.append(best_c)
            remaining.remove(best_c)
            prev = best_c
        routes[ri] = current_route
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    # Greedy insertion of customers to minimize total distance
    # For each customer, find the route and position that gives minimum additional cost
    unassigned = list(cust_ids)
    for c in unassigned:
        best_r = -1
        best_pos = -1
        best_cost = float('inf')
        for ri in range(len(routes)):
            route = routes[ri]
            for pos in range(len(route) + 1):
                cand = route[:pos] + [c] + route[pos:]
                cost, valid, _, _ = check_route(cand)
                if valid and cost < best_cost:
                    best_cost = cost
                    best_r = ri
                    best_pos = pos
        if best_r >= 0:
            routes[best_r].insert(best_pos, c)
        else:
            # Create new route
            routes.append([c])
    
    # Ensure all customers are assigned
    assigned = set()
    for r in routes:
        for c in r:
            assigned.add(c)
    for c in cust_ids:
        if c not in assigned:
            routes.append([c])
    
    # --- Local Search ---
    
    # Helper to get route cost
    def get_route_cost(route):
        return check_route(route)[0]
    
    def _two_opt(route):
        """Apply 2-opt until no improvement, return (new_route, improved)."""
        n = len(route)
        if n < 2:
            return route, False
        
        best_route = list(route)
        best_cost, valid, _, _ = check_route(best_route)
        if not valid:
            return best_route, False
        
        improved = False
        max_passes = 5
        for _ in range(max_passes):
            pass_improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                    c, v, _, _ = check_route(cand)
                    if v and c < best_cost - 1e-7:
                        best_cost = c
                        best_route = cand
                        pass_improved = True
                        improved = True
            if not pass_improved:
                break
        
        return best_route, improved
    
    def _or_opt(route):
        """Apply OR-opt (move 1-2 consecutive nodes) until no improvement."""
        n = len(route)
        if n < 3:
            return route, False
        
        best_route = list(route)
        best_cost, valid, _, _ = check_route(best_route)
        if not valid:
            return best_route, False
        
        improved = False
        max_passes = 5
        for _ in range(max_passes):
            pass_improved = False
            for seg_len in [1, 2]:
                for i in range(n - seg_len + 1):
                    seg = best_route[i:i+seg_len]
                    rest = best_route[:i] + best_route[i+seg_len:]
                    for j in range(len(rest) + 1):
                        cand = rest[:j] + seg + rest[j:]
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost - 1e-7:
                            best_cost = c
                            best_route = cand
                            pass_improved = True
                            improved = True
                            break
                    if pass_improved:
                        break
                if pass_improved:
                    break
            if not pass_improved:
                break
        
        return best_route, improved
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        
        # 1. Intra-route 2-Opt and OR-opt
        for ri in range(len(routes)):
            if time.time() > DEADLINE: break
            route = routes[ri]
            if len(route) < 2: continue
            
            # Try 2-opt
            new_route, imp = _two_opt(route)
            if imp:
                routes[ri] = new_route
                improved = True
            
            # Try OR-opt on the (possibly improved) route
            new_route, imp = _or_opt(routes[ri])
            if imp:
                routes[ri] = new_route
                improved = True
        
        # 2. Inter-route Relocate
        if time.time() > DEADLINE: break
        for ri in range(len(routes)):
            if time.time() > DEADLINE: break
            for ci in range(len(routes[ri])):
                if time.time() > DEADLINE: break
                cust = routes[ri][ci]
                # Remove from ri
                temp_from = routes[ri][:]
                del temp_from[ci]
                c_from, v_from, _, _ = check_route(temp_from)
                if not v_from:
                    continue
                
                for rj in range(len(routes)):
                    if rj == ri: continue
                    if time.time() > DEADLINE: break
                    for cj in range(len(routes[rj]) + 1):
                        if time.time() > DEADLINE: break
                        cand_to = routes[rj][:cj] + [cust] + routes[rj][cj:]
                        c_to, v_to, _, _ = check_route(cand_to)
                        if not v_to:
                            continue
                        
                        # Check capacity
                        load_to = sum(demands[node] for node in cand_to if is_customer[node])
                        if load_to > capacity + _EPS:
                            continue
                        
                        old_cost = c_from + get_route_cost(routes[rj])
                        new_cost = c_from + c_to
                        if new_cost < old_cost - 1e-7:
                            routes[ri] = temp_from
                            routes[rj] = cand_to
                            improved = True
                            # Restart relocation search
                            ri = -1
                            break
                    if ri == -1: break
                if ri == -1: break
            if ri == -1: break
        
        # Remove empty routes
        routes = [r for r in routes if r]
        
        # 3. Inter-route Swap
        if time.time() > DEADLINE: break
        swapped = False
        for ri in range(len(routes)):
            if swapped or time.time() > DEADLINE: break
            for rj in range(ri + 1, len(routes)):
                if swapped or time.time() > DEADLINE: break
                for ci in range(len(routes[ri])):
                    if swapped or time.time() > DEADLINE: break
                    cust1 = routes[ri][ci]
                    for cj in range(len(routes[rj])):
                        if swapped or time.time() > DEADLINE: break
                        cust2 = routes[rj][cj]
                        
                        # Capacity check
                        load_i = sum(demands[node] for node in routes[ri] if is_customer[node])
                        load_j = sum(demands[node] for node in routes[rj] if is_customer[node])
                        new_load_i = load_i - demands[cust1] + demands[cust2]
                        new_load_j = load_j - demands[cust2] + demands[cust1]
                        
                        if new_load_i > capacity + _EPS or new_load_j > capacity + _EPS:
                            continue
                            
                        new_route_i = routes[ri][:]
                        new_route_i[ci] = cust2
                        new_route_j = routes[rj][:]
                        new_route_j[cj] = cust1
                        
                        c_i, v_i, _, _ = check_route(new_route_i)
                        c_j, v_j, _, _ = check_route(new_route_j)
                        
                        if v_i and v_j:
                            old_c_i = get_route_cost(routes[ri])
                            old_c_j = get_route_cost(routes[rj])
                            if old_c_i + old_c_j > c_i + c_j + 1e-7:
                                routes[ri] = new_route_i
                                routes[rj] = new_route_j
                                improved = True
                                swapped = True
                                break
                if swapped: break
            if swapped: break

    # Final validation and fallback
    total_dist = 0.0
    served = set()
    valid = True
    for r in routes:
        c, v, _, _ = check_route(r)
        if not v:
            valid = False
            break
        total_dist += c
        for node in r:
            if node in served:
                valid = False
                break
            served.add(node)
        if not valid:
            break
            
    if not valid or served != set(range(1, n_cust + 1)):
        # Fallback: one customer per route
        customers = list(range(1, n_cust + 1))
        customers.sort(key=lambda v: due[v])
        routes = [[v] for v in customers]
        # Ensure validity of fallback
        for i in range(len(routes)):
            if not check_route(routes[i])[1]:
                rep, _ = try_insert_station(routes[i])
                if rep:
                    routes[i] = rep

    return routes

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    n_customers = instance["n_customers"]
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute pairwise distances using NumPy
    n_nodes = len(coords)
    coords_arr = np.array(coords, dtype=float)
    diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    station_set = set(stations)

    def build_route_from_sequence(cust_seq):
        """
        Given a sequence of customers, build a valid route by inserting stations as needed.
        Returns the route (list of node ids) or None if infeasible.
        """
        if not cust_seq:
            return []
        
        route = []
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        
        for cust in cust_seq:
            d_direct = dist[prev][cust]
            best_t = float('inf')
            best_bat = -1
            best_st = -1
            best_direct = False

            # Option 1: Direct
            if bat >= d_direct - 1e-9:
                t_arr = t + d_direct / speed
                if t_arr < ready[cust]:
                    t_arr = ready[cust]
                if t_arr <= due[cust] + 1e-9:
                    t_new = t_arr + service[cust]
                    bat_new = bat - d_direct
                    load_new = load + demands[cust]
                    if load_new <= capacity:
                        if t_new < best_t:
                            best_t = t_new
                            best_bat = bat_new
                            best_st = -1
                            best_direct = True

            # Option 2: Via Station
            for st in stations:
                d_to_st = dist[prev][st]
                if bat < d_to_st - 1e-9:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                deficit = battery_cap - bat_at_st
                recharge_time = deficit * recharge_rate
                t_st += recharge_time
                bat_st = battery_cap
                d_st_to_cust = dist[st][cust]
                if bat_st < d_st_to_cust - 1e-9:
                    continue
                t_cust = t_st + d_st_to_cust / speed
                if t_cust < ready[cust]:
                    t_cust = ready[cust]
                if t_cust > due[cust] + 1e-9:
                    continue
                t_new = t_cust + service[cust]
                bat_new = bat_st - d_st_to_cust
                load_new = load + demands[cust]
                if load_new <= capacity:
                    if t_new < best_t:
                        best_t = t_new
                        best_bat = bat_new
                        best_st = st
                        best_direct = False

            if best_st == -1 and not best_direct:
                return None
            
            if best_direct:
                route.append(cust)
            else:
                route.append(best_st)
                route.append(cust)
            
            t = best_t
            bat = best_bat
            load += demands[cust]
            prev = cust
        
        # Check return to depot
        d_return = dist[prev][0]
        best_t = float('inf')
        best_st = -1
        best_direct = False

        if bat >= d_return - 1e-9:
            t_return = t + d_return / speed
            if t_return <= horizon + 1e-9:
                best_t = t_return
                best_direct = True

        for st in stations:
            d_to_st = dist[prev][st]
            if bat < d_to_st - 1e-9:
                continue
            t_st = t + d_to_st / speed
            bat_at_st = bat - d_to_st
            deficit = battery_cap - bat_at_st
            recharge_time = deficit * recharge_rate
            t_st += recharge_time
            bat_st = battery_cap
            d_st_to_depot = dist[st][0]
            if bat_st < d_st_to_depot - 1e-9:
                continue
            t_depot = t_st + d_st_to_depot / speed
            if t_depot <= horizon + 1e-9:
                if t_depot < best_t:
                    best_t = t_depot
                    best_st = st
                    best_direct = False

        if best_st == -1 and not best_direct:
            return None
        
        if not best_direct:
            route.append(best_st)
        
        return route

    def route_distance(route):
        if not route:
            return 0.0
        total = 0.0
        prev = 0
        for node in route:
            total += dist[prev][node]
            prev = node
        total += dist[prev][0]
        return total

    def validate_route(route):
        if not route:
            return True
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        for node in route:
            d = dist[prev][node]
            if bat < d - 1e-9:
                return False
            t += d / speed
            bat -= d
            load += demands[node]
            if load > capacity:
                return False
            if 1 <= node <= n_customers:
                if t < ready[node] - 1e-9:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return False
                t += service[node]
            elif node in station_set:
                deficit = battery_cap - bat
                t += max(0, deficit * recharge_rate)
                bat = battery_cap
            prev = node
        d = dist[prev][0]
        if bat < d - 1e-9:
            return False
        t += d / speed
        if t > horizon + 1e-9:
            return False
        return True

    def get_cust_seq(route):
        return [c for c in route if 1 <= c <= n_customers]

    # Savings-based construction
    customers = list(range(1, n_customers + 1))
    
    # Compute savings for each pair of customers
    # Savings(i,j) = dist[0][i] + dist[0][j] - dist[i][j]
    savings = {}
    for i in customers:
        for j in customers:
            if i != j:
                savings[(i,j)] = dist[0][i] + dist[0][j] - dist[i][j]
    
    # Sort by savings (descending)
    sorted_savings = sorted(savings.keys(), key=lambda x: savings[x], reverse=True)
    
    # Initialize each customer as its own route
    routes = {}
    for c in customers:
        routes[c] = [c]
    
    # Track which customers are in which route
    route_of = {c: c for c in customers}
    route_load = {c: demands[c] for c in customers}
    route_start = {c: c for c in customers}
    route_end = {c: c for c in customers}
    
    # Try to merge routes based on savings
    for i, j in sorted_savings:
        if time.time() > deadline - 20.0:
            break
        
        ri = route_of[i]
        rj = route_of[j]
        
        if ri == rj:
            continue
        
        # Check if i is at end of ri and j is at start of rj, or vice versa
        if route_end[ri] != i or route_start[rj] != j:
            continue
        if route_end[rj] != j or route_start[ri] != i:
            continue
        
        # Check capacity
        if route_load[ri] + route_load[rj] > capacity:
            continue
        
        # Try merging ri and rj
        seq_i = routes[ri]
        seq_j = routes[rj]
        merged_seq = seq_i + seq_j
        
        # Only merge if i is last in ri and j is first in rj
        if not (seq_i[-1] == i and seq_j[0] == j):
            continue
        
        merged_route = build_route_from_sequence(merged_seq)
        if merged_route is not None and validate_route(merged_route):
            # Perform merge
            routes[ri] = merged_seq
            route_end[ri] = route_end[rj]
            route_start[ri] = route_start[ri]
            route_load[ri] += route_load[rj]
            
            for c in seq_j:
                route_of[c] = ri
            
            del routes[rj]
            del route_load[rj]
            del route_start[rj]
            del route_end[rj]
    
    # Convert to list of routes
    final_routes = []
    for key in routes:
        cust_seq = routes[key]
        route = build_route_from_sequence(cust_seq)
        if route is not None and validate_route(route):
            final_routes.append(route)
    
    # Add any missing customers
    covered = set()
    for r in final_routes:
        for node in r:
            if 1 <= node <= n_customers:
                covered.add(node)
    
    for cust in range(1, n_customers + 1):
        if cust not in covered:
            single_route = build_route_from_sequence([cust])
            if single_route is not None and validate_route(single_route):
                final_routes.append(single_route)
                covered.add(cust)
            else:
                final_routes.append([cust])
                covered.add(cust)
    
    routes = final_routes
    
    # Local search: Relocate and 2-opt with time budget
    if time.time() < deadline - 10.0:
        # Relocate: Move single customers between routes
        improved = True
        max_relocate_iters = 5
        while improved and time.time() < deadline - 5.0:
            improved = False
            for _ in range(max_relocate_iters):
                if time.time() > deadline - 5.0:
                    break
                for i in range(len(routes)):
                    if time.time() > deadline - 5.0:
                        break
                    cust_seq_i = get_cust_seq(routes[i])
                    for j in range(len(cust_seq_i)):
                        if time.time() > deadline - 5.0:
                            break
                        cust = cust_seq_i[j]
                        
                        # Try moving to another route
                        for k in range(len(routes)):
                            if k == i:
                                continue
                            if time.time() > deadline - 5.0:
                                break
                            cust_seq_k = get_cust_seq(routes[k])
                            for pos in range(len(cust_seq_k) + 1):
                                new_seq_i = cust_seq_i[:j] + cust_seq_i[j+1:]
                                new_seq_k = cust_seq_k[:pos] + [cust] + cust_seq_k[pos:]
                                
                                r_i = build_route_from_sequence(new_seq_i)
                                r_k = build_route_from_sequence(new_seq_k)
                                
                                if r_i is None or r_k is None:
                                    continue
                                if not validate_route(r_i) or not validate_route(r_k):
                                    continue
                                
                                old_dist = route_distance(routes[i]) + route_distance(routes[k])
                                new_dist = route_distance(r_i) + route_distance(r_k)
                                
                                if new_dist < old_dist - 1e-6:
                                    routes[i] = r_i
                                    routes[k] = r_k
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        # 2-opt within each route
        if time.time() < deadline - 5.0:
            for i in range(len(routes)):
                if time.time() > deadline - 5.0:
                    break
                cust_seq = get_cust_seq(routes[i])
                if len(cust_seq) < 2:
                    continue
                
                best_seq = cust_seq[:]
                best_route = build_route_from_sequence(cust_seq)
                if best_route is None:
                    continue
                best_dist = route_distance(best_route)
                
                n = len(best_seq)
                for a in range(n - 1):
                    if time.time() > deadline - 5.0:
                        break
                    for b in range(a + 1, n):
                        if time.time() > deadline - 5.0:
                            break
                        new_seq = best_seq[:a] + best_seq[a:b+1][::-1] + best_seq[b+1:]
                        new_route = build_route_from_sequence(new_seq)
                        if new_route is not None and validate_route(new_route):
                            d = route_distance(new_route)
                            if d < best_dist - 1e-6:
                                best_dist = d
                                best_seq = new_seq[:]
                                best_route = new_route
                
                if best_route is not None and best_dist < route_distance(routes[i]) - 1e-6:
                    routes[i] = best_route

    # Final validation
    valid_routes = []
    covered = set()
    for r in routes:
        if validate_route(r):
            valid_routes.append(r)
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)

    # Add any missing customers
    for cust in range(1, n_customers + 1):
        if cust not in covered:
            single_route = build_route_from_sequence([cust])
            if single_route is not None:
                valid_routes.append(single_route)
            else:
                valid_routes.append([cust])
            covered.add(cust)

    return valid_routes
# EVOLVE-BLOCK-END

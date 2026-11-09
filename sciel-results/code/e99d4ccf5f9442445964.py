# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def _expand_route(customer_route, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations):
    """
    Takes a list of customer IDs and returns a list of node IDs (including stations) 
    that forms a feasible route, or None if infeasible.
    Also returns the total distance for scoring.
    """
    if not customer_route:
        return [], 0.0
        
    route = []
    current_node = 0
    bat = battery_cap
    total_time = 0.0
    total_dist = 0.0
    
    for c in customer_route:
        # Travel to c, inserting stations as needed
        while True:
            d = dist[current_node, c]
            if bat >= d:
                break
            # Need to charge
            best_s = None
            best_d_s = float('inf')
            for s in stations:
                d_to_s = dist[current_node, s]
                if bat >= d_to_s:
                    if dist[s, c] <= battery_cap:
                        if d_to_s < best_d_s:
                            best_d_s = d_to_s
                            best_s = s
            
            if best_s is None:
                return None, 0.0
            
            d_to_s = dist[current_node, best_s]
            travel_time = d_to_s / speed
            total_time += travel_time
            bat -= d_to_s
            total_dist += d_to_s
            
            charge_time = (battery_cap - bat) * recharge_rate
            total_time += charge_time
            bat = battery_cap
            
            route.append(best_s)
            current_node = best_s
            
        d = dist[current_node, c]
        travel_time = d / speed
        total_time += travel_time
        bat -= d
        total_dist += d
        
        arr_time = total_time
        start_service = max(arr_time, ready[c])
        if start_service > due[c]:
            return None, 0.0
        
        total_time = start_service + service[c]
        route.append(c)
        current_node = c
        
    # Return to depot
    while True:
        d = dist[current_node, 0]
        if bat >= d:
            break
        best_s = None
        best_d_s = float('inf')
        for s in stations:
            d_to_s = dist[current_node, s]
            if bat >= d_to_s:
                if dist[s, 0] <= battery_cap:
                    if d_to_s < best_d_s:
                        best_d_s = d_to_s
                        best_s = s
        
        if best_s is None:
            return None, 0.0
            
        d_to_s = dist[current_node, best_s]
        travel_time = d_to_s / speed
        total_time += travel_time
        bat -= d_to_s
        total_dist += d_to_s
        
        charge_time = (battery_cap - bat) * recharge_rate
        total_time += charge_time
        bat = battery_cap
        
        route.append(best_s)
        current_node = best_s
        
    d = dist[current_node, 0]
    travel_time = d / speed
    total_time += travel_time
    bat -= d
    total_dist += d
    
    if total_time > horizon:
        return None, 0.0
        
    return route, total_dist

def _solve_instance(instance, time_limit=100):
    coords = np.array(instance["coords"])
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distances
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = sqrt((coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2)
            dist[i, j] = d
            dist[j, i] = d
    
    customers = list(range(1, n_customers + 1))
    routes = []
    unserved = set(customers)
    start_time = time.time()
    
    # Phase 1: Build initial routes using savings heuristic
    # Calculate savings for each pair
    savings = {}
    for i in customers:
        for j in customers:
            if i != j:
                savings[(i, j)] = dist[0, i] + dist[0, j] - dist[i, j]
    
    # Sort pairs by savings (descending)
    sorted_pairs = sorted(savings.keys(), key=lambda x: savings[x], reverse=True)
    
    # Initialize: each customer in its own route
    # route[i] = list of customers in route i, or None if route is full
    active_routes = {c: [c] for c in customers}
    route_demand = {c: demands[c] for c in customers}
    
    # Merge routes based on savings
    for (i, j) in sorted_pairs:
        if time.time() - start_time > time_limit * 0.3:
            break
        if i not in active_routes or j not in active_routes:
            continue
        if i == j:
            continue
        
        ri = i  # route containing i
        rj = j   # route containing j
        
        # Find which routes i and j belong to
        ri = None
        rj = None
        for r, custs in active_routes.items():
            if i in custs:
                ri = r
            if j in custs:
                rj = r
            if ri is not None and rj is not None:
                break
        
        if ri is None or rj is None or ri == rj:
            continue
        
        # Check if we can merge: i should be at end of ri, j at start of rj (or vice versa)
        ri_custs = active_routes[ri]
        rj_custs = active_routes[rj]
        
        # Try merging: append rj to ri or vice versa
        # Only merge if i is last in ri and j is first in rj, or i is first and j is last
        merged = None
        if ri_custs[-1] == i and rj_custs[0] == j:
            new_custs = ri_custs + rj_custs
            if sum(demands[c] for c in new_custs) <= capacity:
                expanded, r_dist = _expand_route(new_custs, dist, ready, due, service, battery, speed, 
                                                   recharge_rate, horizon, stations)
                if expanded is not None:
                    merged = new_custs
        elif ri_custs[0] == i and rj_custs[-1] == j:
            new_custs = rj_custs + ri_custs
            if sum(demands[c] for c in new_custs) <= capacity:
                expanded, r_dist = _expand_route(new_custs, dist, ready, due, service, battery, speed, 
                                                   recharge_rate, horizon, stations)
                if expanded is not None:
                    merged = new_custs
        
        if merged is not None:
            # Merge rj into ri
            # Remove rj
            del active_routes[rj]
            active_routes[ri] = merged
            # Update route_demand
            route_demand[ri] = sum(demands[c] for c in merged)
            # Remove demand for rj
            if rj in route_demand:
                del route_demand[rj]
    
    # Convert to list of routes
    for r, custs in active_routes.items():
        expanded, _ = _expand_route(custs, dist, ready, due, service, battery, speed, 
                                     recharge_rate, horizon, stations)
        if expanded is not None:
            routes.append(expanded)
        else:
            # Fallback: serve individually
            for c in custs:
                exp_single, _ = _expand_route([c], dist, ready, due, service, battery, speed, 
                                              recharge_rate, horizon, stations)
                if exp_single is not None:
                    routes.append(exp_single)
                else:
                    routes.append([c])
    
    # Ensure all customers are served
    served = set()
    for route in routes:
        for node in route:
            if 1 <= node <= n_customers:
                served.add(node)
    
    for c in customers:
        if c not in served:
            expanded, _ = _expand_route([c], dist, ready, due, service, battery, speed, 
                                         recharge_rate, horizon, stations)
            if expanded is not None:
                routes.append(expanded)
            else:
                routes.append([c])
    
    # Phase 2: Optimize with 2-opt and relocate
    def extract_customers(route):
        return [node for node in route if 1 <= node <= n_customers]
    
    def two_opt_customers(customer_seq, max_iters=5):
        if len(customer_seq) < 2:
            return customer_seq
        
        best_seq = list(customer_seq)
        best_expanded, best_dist = _expand_route(best_seq, dist, ready, due, service, battery, speed, 
                                                  recharge_rate, horizon, stations)
        if best_expanded is None:
            return customer_seq
        
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            for i in range(len(best_seq) - 1):
                for j in range(i + 1, len(best_seq)):
                    new_seq = best_seq[:i] + best_seq[i:j+1][::-1] + best_seq[j+1:]
                    new_expanded, new_dist = _expand_route(new_seq, dist, ready, due, service, battery, speed, 
                                                           recharge_rate, horizon, stations)
                    if new_expanded is not None and new_dist < best_dist - 1e-9:
                        best_seq = new_seq
                        best_expanded = new_expanded
                        best_dist = new_dist
                        improved = True
        return best_seq
    
    # 2-opt on each route
    optimized_routes = []
    for route in routes:
        cust_seq = extract_customers(route)
        opt_seq = two_opt_customers(cust_seq, max_iters=5)
        opt_expanded, _ = _expand_route(opt_seq, dist, ready, due, service, battery, speed, 
                                        recharge_rate, horizon, stations)
        if opt_expanded is not None:
            optimized_routes.append(opt_expanded)
        else:
            optimized_routes.append(route)
    
    # Relocate moves: try moving customers between routes
    def total_route_distance(route):
        _, d = _expand_route(extract_customers(route), dist, ready, due, service, battery, speed, 
                              recharge_rate, horizon, stations)
        return d
    
    improved_global = True
    global_iter = 0
    while improved_global and global_iter < 3 and time.time() - start_time < time_limit * 0.8:
        improved_global = False
        global_iter += 1
        
        for ri in range(len(optimized_routes)):
            custs_i = extract_customers(optimized_routes[ri])
            if len(custs_i) < 2:
                continue
            
            for ci, c in enumerate(custs_i):
                # Try removing c from route i
                new_custs_i = custs_i[:ci] + custs_i[ci+1:]
                exp_i, dist_i = _expand_route(new_custs_i, dist, ready, due, service, battery, speed, 
                                               recharge_rate, horizon, stations)
                if exp_i is None:
                    continue
                
                # Try inserting c into each other route
                for rj in range(len(optimized_routes)):
                    if rj == ri:
                        continue
                    custs_j = extract_customers(optimized_routes[rj])
                    
                    # Try all insertion positions
                    best_pos = None
                    best_dist_j = float('inf')
                    for pos in range(len(custs_j) + 1):
                        new_custs_j = custs_j[:pos] + [c] + custs_j[pos:]
                        if sum(demands[x] for x in new_custs_j) > capacity:
                            continue
                        exp_j, dist_j = _expand_route(new_custs_j, dist, ready, due, service, battery, speed, 
                                                       recharge_rate, horizon, stations)
                        if exp_j is not None and dist_j < best_dist_j:
                            best_dist_j = dist_j
                            best_pos = pos
                            best_custs_j = new_custs_j
                    
                    if best_pos is not None:
                        # Check total improvement
                        old_total = dist_i + total_route_distance(optimized_routes[rj])
                        new_total = dist_i + best_dist_j
                        if new_total < old_total - 1e-9:
                            # Perform the move
                            optimized_routes[ri] = exp_i
                            optimized_routes[rj] = _expand_route(best_custs_j, dist, ready, due, service, battery, speed, 
                                                                  recharge_rate, horizon, stations)[0]
                            improved_global = True
                            break
                    if improved_global:
                        break
                if improved_global:
                    break
            if improved_global:
                break
    
    return optimized_routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145
    
    routes = _solve_instance(instance, time_limit=100)
    
    return routes
# EVOLVE-BLOCK-END

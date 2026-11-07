"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

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
        Optimized to pick the best station (minimizing time) if multiple are feasible.
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

    # Compute total distance of a route
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

    # --- Construction Phase ---
    # We will try a few different construction heuristics and keep the best result.
    
    def construct_greedy_time():
        """Original greedy: start with most urgent, add nearest feasible."""
        customers = list(range(1, n_customers + 1))
        customers.sort(key=lambda v: due[v])
        
        routes = []
        remaining = set(customers)
        
        while remaining and time.time() < deadline - 10.0:
            best_route = None
            best_used = set()
            
            # Try starting from the most urgent customer
            start_cust = min(remaining, key=lambda v: due[v])
            
            route_seq = [start_cust]
            used = {start_cust}
            current_load = demands[start_cust]
            
            while True:
                best_next = None
                best_metric = float('inf')
                
                for c in remaining - used:
                    if current_load + demands[c] > capacity:
                        continue
                    new_seq = route_seq + [c]
                    test_route = build_route_from_sequence(new_seq)
                    if test_route is not None:
                        d = route_distance(test_route)
                        if d < best_metric:
                            best_metric = d
                            best_next = c
                
                if best_next is None:
                    break
                route_seq.append(best_next)
                used.add(best_next)
                current_load += demands[best_next]
            
            final_route = build_route_from_sequence(route_seq)
            if final_route is not None and validate_route(final_route):
                best_route = final_route
                best_used = used
            
            if best_route is None:
                single_route = build_route_from_sequence([start_cust])
                if single_route is not None and validate_route(single_route):
                    best_route = single_route
                    best_used = {start_cust}
                else:
                    best_route = [start_cust]
                    best_used = {start_cust}
            
            routes.append(best_route)
            remaining -= best_used
        
        for c in remaining:
            single_route = build_route_from_sequence([c])
            if single_route is not None:
                routes.append(single_route)
            else:
                routes.append([c])
        
        return routes

    def construct_sweep():
        """Sweep heuristic: sort by angle, pack routes."""
        customers = list(range(1, n_customers + 1))
        depot = 0
        angles = []
        for c in customers:
            dx = coords_arr[c][0] - coords_arr[depot][0]
            dy = coords_arr[c][1] - coords_arr[depot][1]
            ang = math.atan2(dy, dx)
            angles.append((ang, c))
        angles.sort(key=lambda x: x[0])
        sorted_custs = [c for _, c in angles]
        
        routes = []
        unvisited = set(customers)
        
        # Greedy packing
        for c in sorted_custs:
            if c not in unvisited:
                continue
            
            best_route_idx = -1
            best_pos = -1
            best_delta = float('inf')
            
            for i, r in enumerate(routes):
                cust_seq = get_cust_seq(r)
                # Try inserting at each position
                for pos in range(len(cust_seq) + 1):
                    new_seq = cust_seq[:pos] + [c] + cust_seq[pos:]
                    if sum(demands[x] for x in new_seq) > capacity:
                        continue
                    new_route = build_route_from_sequence(new_seq)
                    if new_route is None:
                        continue
                    # Calculate delta cost
                    old_cost = route_distance(r)
                    new_cost = route_distance(new_route)
                    delta = new_cost - old_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_route_idx = i
                        best_pos = pos
                        best_new_route = new_route
            
            if best_route_idx != -1:
                routes[best_route_idx] = best_new_route
                unvisited.remove(c)
            else:
                # Start new route
                single_route = build_route_from_sequence([c])
                if single_route is not None:
                    routes.append(single_route)
                    unvisited.remove(c)
                else:
                    routes.append([c])
                    unvisited.remove(c)
        
        # Handle any remaining (should be none if logic above is correct, but just in case)
        for c in unvisited:
            single_route = build_route_from_sequence([c])
            if single_route is not None:
                routes.append(single_route)
            else:
                routes.append([c])
        
        return routes

    def construct_nearest_neighbor():
        """Nearest neighbor construction."""
        customers = list(range(1, n_customers + 1))
        routes = []
        unvisited = set(customers)
        
        while unvisited:
            # Start with nearest to depot
            start = min(unvisited, key=lambda c: dist[0][c])
            route_seq = [start]
            unvisited.remove(start)
            current_load = demands[start]
            prev = start
            
            while True:
                best_next = None
                best_d = float('inf')
                for c in unvisited:
                    if current_load + demands[c] > capacity:
                        continue
                    d = dist[prev][c]
                    if d < best_d:
                        # Check feasibility quickly
                        test_seq = route_seq + [c]
                        if build_route_from_sequence(test_seq) is not None:
                            best_d = d
                            best_next = c
                
                if best_next is None:
                    break
                
                route_seq.append(best_next)
                unvisited.remove(best_next)
                current_load += demands[best_next]
                prev = best_next
            
            final_route = build_route_from_sequence(route_seq)
            if final_route is not None:
                routes.append(final_route)
            else:
                # Fallback: split if invalid? Just append singletons for safety if this happens
                # But build_route_from_sequence should handle it. If not, we have a bug.
                # For safety, let's just append the sequence and hope validation catches it or we fix it later.
                routes.append(route_seq)
        
        return routes

    # Generate initial solutions
    solutions = []
    
    if time.time() < deadline - 20.0:
        sol1 = construct_greedy_time()
        solutions.append(sol1)
    
    if time.time() < deadline - 20.0:
        sol2 = construct_sweep()
        solutions.append(sol2)
        
    if time.time() < deadline - 20.0:
        sol3 = construct_nearest_neighbor()
        solutions.append(sol3)
        
    # Select best initial solution
    best_routes = None
    best_dist = float('inf')
    for sol in solutions:
        # Validate and calculate distance
        valid = True
        total_d = 0
        covered = set()
        for r in sol:
            if not validate_route(r):
                valid = False
                break
            total_d += route_distance(r)
            for c in get_cust_seq(r):
                if c in covered:
                    valid = False
                    break
                covered.add(c)
            if not valid:
                break
        
        if valid and covered == set(range(1, n_customers + 1)) and total_d < best_dist:
            best_dist = total_d
            best_routes = [r[:] for r in sol]
            
    if best_routes is None:
        # Fallback to simple valid solution
        best_routes = []
        for c in range(1, n_customers + 1):
            best_routes.append([c])
        # Ensure validity, though singletons should be valid
        best_dist = sum(route_distance(r) for r in best_routes)

    # --- Local Search Phase ---
    # We will apply 2-opt and Or-opt (relocate) to the best solution.
    
    current_routes = best_routes
    current_dist = best_dist
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # 1. 2-Opt within routes
        for i in range(len(current_routes)):
            if time.time() > deadline - 5.0:
                break
            cust_seq = get_cust_seq(current_routes[i])
            if len(cust_seq) < 2:
                continue
            
            best_seq = cust_seq[:]
            best_route = build_route_from_sequence(cust_seq)
            if best_route is None:
                continue
            best_d = route_distance(best_route)
            
            # Simple 2-opt
            for a in range(len(best_seq)):
                if time.time() > deadline - 5.0:
                    break
                for b in range(a + 1, len(best_seq)):
                    new_seq = best_seq[:a] + best_seq[a:b+1][::-1] + best_seq[b+1:]
                    new_route = build_route_from_sequence(new_seq)
                    if new_route is not None and validate_route(new_route):
                        d = route_distance(new_route)
                        if d < best_d - 1e-6:
                            best_d = d
                            best_seq = new_seq[:]
                            best_route = new_route
            
            if best_route is not None and best_d < route_distance(current_routes[i]) - 1e-6:
                current_routes[i] = best_route
                current_dist += best_d - route_distance(current_routes[i]) # Update dist roughly
                improved = True
                # Recalculate total dist to be safe
                current_dist = sum(route_distance(r) for r in current_routes)
                break # Restart loop if changed
        
        # 2. Relocate (Or-opt size 1) between routes
        if time.time() < deadline - 5.0:
            for i in range(len(current_routes)):
                if time.time() > deadline - 5.0:
                    break
                cust_seq_i = get_cust_seq(current_routes[i])
                if not cust_seq_i:
                    continue
                for j_idx, cust in enumerate(cust_seq_i):
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Remove cust from route i
                    remaining_seq_i = [c for c in cust_seq_i if c != cust]
                    route_i_new = build_route_from_sequence(remaining_seq_i) if remaining_seq_i else []
                    if route_i_new is None:
                        continue
                    if not validate_route(route_i_new):
                        continue
                    
                    # Try inserting into other routes
                    for k in range(len(current_routes)):
                        if k == i:
                            continue
                        if time.time() > deadline - 5.0:
                            break
                        cust_seq_k = get_cust_seq(current_routes[k])
                        # Try inserting at each position
                        for pos in range(len(cust_seq_k) + 1):
                            new_seq_k = cust_seq_k[:pos] + [cust] + cust_seq_k[pos:]
                            if sum(demands[x] for x in new_seq_k) > capacity:
                                continue
                            route_k_new = build_route_from_sequence(new_seq_k)
                            if route_k_new is None:
                                continue
                            if not validate_route(route_k_new):
                                continue
                            
                            new_total = route_distance(route_i_new) + route_distance(route_k_new)
                            old_total = route_distance(current_routes[i]) + route_distance(current_routes[k])
                            
                            if new_total < old_total - 1e-6:
                                current_routes[i] = route_i_new
                                current_routes[k] = route_k_new
                                current_dist = sum(route_distance(r) for r in current_routes)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    # Final check and cleanup
    # Ensure all customers are covered exactly once
    covered = set()
    final_routes = []
    for r in current_routes:
        if not validate_route(r):
            continue
        custs_in_r = get_cust_seq(r)
        if any(c in covered for c in custs_in_r):
            # Conflict, skip this route? Or try to fix? 
            # For safety, if we have conflicts, we might need to reconstruct.
            # But local search should maintain feasibility.
            pass
        covered.update(custs_in_r)
        final_routes.append(r)
        
    # If we lost customers due to conflicts or errors, re-add them as singletons
    for c in range(1, n_customers + 1):
        if c not in covered:
            single_route = build_route_from_sequence([c])
            if single_route is not None and validate_route(single_route):
                final_routes.append(single_route)
                covered.add(c)
            else:
                # Force add, even if invalid? No, must be valid.
                # If single is invalid, something is wrong with the instance or logic.
                # Try just [c]
                final_routes.append([c])
                covered.add(c)

    return final_routes
# EVOLVE-BLOCK-END

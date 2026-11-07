"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def plan(instance):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit

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
            if bat >= d_direct:
                t_arr = t + d_direct / speed
                if t_arr < ready[cust]:
                    t_arr = ready[cust]
                if t_arr <= due[cust]:
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
                if bat < d_to_st:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                deficit = battery_cap - bat_at_st
                recharge_time = deficit * recharge_rate
                t_st += recharge_time
                bat_st = battery_cap
                d_st_to_cust = dist[st][cust]
                if bat_st < d_st_to_cust:
                    continue
                t_cust = t_st + d_st_to_cust / speed
                if t_cust < ready[cust]:
                    t_cust = ready[cust]
                if t_cust > due[cust]:
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

        if bat >= d_return:
            t_return = t + d_return / speed
            if t_return <= horizon:
                best_t = t_return
                best_direct = True

        for st in stations:
            d_to_st = dist[prev][st]
            if bat < d_to_st:
                continue
            t_st = t + d_to_st / speed
            bat_at_st = bat - d_to_st
            deficit = battery_cap - bat_at_st
            recharge_time = deficit * recharge_rate
            t_st += recharge_time
            bat_st = battery_cap
            d_st_to_depot = dist[st][0]
            if bat_st < d_st_to_depot:
                continue
            t_depot = t_st + d_st_to_depot / speed
            if t_depot <= horizon:
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
            elif node in stations:
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

    # Greedy construction: build routes by adding customers one at a time
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: due[v])

    routes = []
    remaining = set(customers)
    
    while remaining and time.time() < deadline - 5.0:
        best_route = None
        best_dist = float('inf')
        best_used = set()
        
        # Try starting from the most urgent customer
        start_cust = min(remaining, key=lambda v: due[v])
        
        # Build a route starting with this customer using greedy insertion
        route_seq = [start_cust]
        used = {start_cust}
        current_load = demands[start_cust]
        
        # Greedily add the nearest feasible customer
        while True:
            best_next = None
            best_metric = float('inf')
            
            for c in remaining - used:
                if current_load + demands[c] > capacity:
                    continue
                # Check if we can add c to the current sequence
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
        
        # Finalize the route
        final_route = build_route_from_sequence(route_seq)
        if final_route is not None and validate_route(final_route):
            best_route = final_route
            best_used = used
        
        if best_route is None:
            # Fallback: single customer route
            single_route = build_route_from_sequence([start_cust])
            if single_route is not None and validate_route(single_route):
                best_route = single_route
                best_used = {start_cust}
            else:
                # Should not happen
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

    # Local search: Relocate and 2-opt
    if time.time() < deadline - 10.0:
        # Relocate: Move single customers between routes
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(routes)):
                if time.time() > deadline - 5.0: break
                for j in range(len(routes[i])):
                    if time.time() > deadline - 5.0: break
                    cust = routes[i][j]
                    if not (1 <= cust <= n_customers):
                        continue
                    
                    # Try moving to another route
                    for k in range(len(routes)):
                        if k == i:
                            continue
                        # Try inserting at each position in route k
                        for pos in range(len(routes[k]) + 1):
                            new_route_i_seq = [c for c in routes[i] if c != cust]
                            new_route_k_seq = [c for c in routes[k] if 1 <= c <= n_customers]
                            new_route_k_seq.insert(pos, cust)
                            
                            r_i = build_route_from_sequence(new_route_i_seq)
                            r_k = build_route_from_sequence(new_route_k_seq)
                            
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
                if time.time() > deadline - 5.0: break
                cust_seq = [c for c in routes[i] if 1 <= c <= n_customers]
                if len(cust_seq) < 2:
                    continue
                
                best_seq = cust_seq[:]
                best_route = build_route_from_sequence(cust_seq)
                if best_route is None:
                    continue
                best_dist = route_distance(best_route)
                
                for a in range(len(best_seq)):
                    for b in range(a+1, len(best_seq)):
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

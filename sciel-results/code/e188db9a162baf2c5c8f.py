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
                # Can reach directly
                break
            # Need to charge
            # Find a station s reachable from current_node
            found_station = False
            best_s = None
            best_d_s = float('inf')
            for s in stations:
                d_to_s = dist[current_node, s]
                if bat >= d_to_s:
                    # Check if we can reach c from s (with full battery)
                    if dist[s, c] <= battery_cap:
                        if d_to_s < best_d_s:
                            best_d_s = d_to_s
                            best_s = s
            
            if best_s is None:
                return None, 0.0 # Infeasible
            
            # Go to station
            d_to_s = dist[current_node, best_s]
            travel_time = d_to_s / speed
            total_time += travel_time
            bat -= d_to_s
            total_dist += d_to_s
            
            # Charge
            charge_time = (battery_cap - bat) * recharge_rate
            total_time += charge_time
            bat = battery_cap
            
            route.append(best_s)
            current_node = best_s
            
        # Now we can go to c
        d = dist[current_node, c]
        travel_time = d / speed
        total_time += travel_time
        bat -= d
        total_dist += d
        
        # Arrive at c
        arr_time = total_time
        start_service = max(arr_time, ready[c])
        if start_service > due[c]:
            return None, 0.0 # Time window violation
        
        total_time = start_service + service[c]
        route.append(c)
        current_node = c
        
    # Return to depot
    while True:
        d = dist[current_node, 0]
        if bat >= d:
            break
        # Need to charge
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
        
    # Final leg to depot
    d = dist[current_node, 0]
    travel_time = d / speed
    total_time += travel_time
    bat -= d
    total_dist += d
    
    if total_time > horizon:
        return None, 0.0
        
    return route, total_dist

def _solve_instance(instance, time_limit=40):
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
    
    while unserved:
        if time.time() - start_time > time_limit:
            # Emergency: assign remaining to single routes, expand them
            for c in sorted(unserved, key=lambda x: due[x]):
                expanded, _ = _expand_route([c], dist, ready, due, service, battery, speed, 
                                             recharge_rate, horizon, stations)
                if expanded is not None:
                    routes.append(expanded)
                else:
                    # If even single customer is infeasible, this is a critical error
                    # But we must return something.
                    routes.append([c])
            break
            
        # Start a new route
        current_customers = []
        current_demand = 0
        
        while unserved:
            best_customer = None
            best_dist = float('inf')
            
            for c in unserved:
                if current_demand + demands[c] > capacity:
                    continue
                
                # Test adding c to current_route
                new_customers = current_customers + [c]
                expanded, r_dist = _expand_route(new_customers, dist, ready, due, service, battery, speed, 
                                                   recharge_rate, horizon, stations)
                if expanded is not None:
                    if r_dist < best_dist:
                        best_dist = r_dist
                        best_customer = c
            
            if best_customer is None:
                break
            
            current_customers.append(best_customer)
            current_demand += demands[best_customer]
            unserved.discard(best_customer)
        
        if current_customers:
            expanded, _ = _expand_route(current_customers, dist, ready, due, service, battery, speed, 
                                         recharge_rate, horizon, stations)
            if expanded is not None:
                routes.append(expanded)
            else:
                # Should not happen, but handle gracefully
                for c in current_customers:
                    exp_single, _ = _expand_route([c], dist, ready, due, service, battery, speed, 
                                                  recharge_rate, horizon, stations)
                    if exp_single is not None:
                        routes.append(exp_single)
                    else:
                        routes.append([c])
        else:
            break
    
    if unserved:
        for c in sorted(unserved, key=lambda x: due[x]):
            expanded, _ = _expand_route([c], dist, ready, due, service, battery, speed, 
                                         recharge_rate, horizon, stations)
            if expanded is not None:
                routes.append(expanded)
            else:
                routes.append([c])
    
    # Optimize routes with 2-opt on customer sequences
    # We need to extract customer sequences from the expanded routes
    # But optimization is complex with stations. 
    # Given time limits, we will skip complex optimization or do a simple one.
    # Let's try a simple 2-opt on the customer order, re-expanding.
    
    def extract_customers(route):
        return [node for node in route if node not in stations and node != 0]

    def two_opt_customers(customer_seq, max_iters=10):
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

    optimized_routes = []
    for route in routes:
        cust_seq = extract_customers(route)
        opt_seq = two_opt_customers(cust_seq, max_iters=10)
        opt_expanded, _ = _expand_route(opt_seq, dist, ready, due, service, battery, speed, 
                                        recharge_rate, horizon, stations)
        if opt_expanded is not None:
            optimized_routes.append(opt_expanded)
        else:
            # Fallback to original if optimization broke it
            optimized_routes.append(route)
    
    return optimized_routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140
    
    routes = _solve_instance(instance, time_limit=40)
    
    return routes
# EVOLVE-BLOCK-END

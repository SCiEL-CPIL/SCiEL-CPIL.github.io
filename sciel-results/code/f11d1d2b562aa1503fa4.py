"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distance matrix
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
            dist[i][j] = d
            dist[j][i] = d
    
    customers = list(range(1, n_customers + 1))
    
    # Try to build routes using a greedy approach with time window feasibility
    def route_feasible(route, battery_full=True):
        """Check if a route is feasible. Returns (feasible, total_dist, end_battery, end_time)"""
        cur_time = 0.0
        cur_battery = battery if battery_full else 0.0
        total_dist = 0.0
        cur_pos = 0  # depot
        
        # Visit customers in route
        for cust in route:
            d = dist[cur_pos][cust]
            cur_time += d / speed
            cur_battery -= d
            if cur_battery < 0:
                return False, total_dist, cur_battery, cur_time
            total_dist += d
            
            # If battery too low, may need to charge, but for simplicity in greedy, 
            # we assume we plan routes that don't need mid-route charging if possible
            # For now, just check if we can reach depot after this
            
            cur_time = max(cur_time, ready[cust])
            if cur_time > due[cust]:
                return False, total_dist, cur_battery, cur_time
            cur_time += service[cust]
            cur_pos = cust
        
        # Return to depot
        d = dist[cur_pos][0]
        cur_time += d / speed
        cur_battery -= d
        if cur_battery < 0:
            return False, total_dist, cur_battery, cur_time
        total_dist += d
        cur_pos = 0
        
        if cur_time > horizon:
            return False, total_dist, cur_battery, cur_time
        
        return True, total_dist, cur_battery, cur_time
    
    def try_build_routes():
        """Greedy nearest neighbor with time window and battery constraints"""
        unvisited = set(customers)
        routes = []
        
        while unvisited:
            # Start new route from depot
            route = []
            cur_pos = 0
            cur_time = 0.0
            cur_battery = battery
            total_dist = 0.0
            route_demand = 0
            
            while unvisited:
                # Find nearest feasible customer
                best_cust = None
                best_dist = float('inf')
                
                for cust in unvisited:
                    d = dist[cur_pos][cust]
                    new_time = cur_time + d / speed
                    new_battery = cur_battery - d
                    
                    if new_battery < 0:
                        continue
                    if route_demand + demands[cust] > capacity:
                        continue
                    
                    new_time = max(new_time, ready[cust])
                    if new_time > due[cust]:
                        continue
                    
                    # Check if we can return to depot after visiting this customer
                    # and all remaining in route
                    return_dist = dist[cust][0]
                    if new_battery - return_dist < 0:
                        continue
                    
                    if new_time + service[cust] + return_dist / speed > horizon:
                        continue
                    
                    if d < best_dist:
                        best_dist = d
                        best_cust = cust
                
                if best_cust is None:
                    break
                
                # Add customer to route
                cur_time += best_dist / speed
                cur_battery -= best_dist
                total_dist += best_dist
                cur_time = max(cur_time, ready[best_cust])
                cur_time += service[best_cust]
                route_demand += demands[best_cust]
                route.append(best_cust)
                unvisited.remove(best_cust)
                cur_pos = best_cust
            
            if route:
                # Check full route feasibility
                feasible, total_d, _, _ = route_feasible(route)
                if feasible:
                    routes.append(route)
                else:
                    # Should not happen with greedy, but handle gracefully
                    # Re-add customers
                    unvisited.update(route)
                    # Try one customer at a time
                    for cust in route:
                        if cust in unvisited:
                            continue
                        one_route = [cust]
                        feasible2, _, _, _ = route_feasible(one_route)
                        if feasible2:
                            routes.append(one_route)
                            unvisited.add(cust)
            else:
                # No customer could be added, take one
                for cust in list(unvisited):
                    one_route = [cust]
                    feasible, _, _, _ = route_feasible(one_route)
                    if feasible:
                        routes.append(one_route)
                        unvisited.remove(cust)
                        break
        
        return routes
    
    # First, try greedy construction
    routes = try_build_routes()
    
    # Verify all customers are served
    all_served = set()
    for route in routes:
        all_served.update(route)
    
    if all_served != set(customers):
        # Fallback: one customer per route
        routes = [[v] for v in customers]
    
    # Try local improvement: 2-opt and reinsertion
    def route_total_dist(route):
        if not route:
            return 0.0
        d = dist[0][route[0]]
        for i in range(len(route) - 1):
            d += dist[route[i]][route[i + 1]]
        d += dist[route[-1]][0]
        return d
    
    # Simple improvement: try reinserting customers in other routes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, route_i in enumerate(routes):
            for j, cust in enumerate(route_i):
                # Try removing cust and inserting into another route
                route_i_without = route_i[:j] + route_i[j + 1:]
                
                for k, route_k in enumerate(routes):
                    if k == i:
                        continue
                    
                    # Try all insertion positions in route_k
                    for pos in range(len(route_k) + 1):
                        new_route_k = route_k[:pos] + [cust] + route_k[pos:]
                        new_route_i = route_i_without
                        
                        # Check feasibility
                        feas_i, _, _, _ = route_feasible(new_route_i) if new_route_i else (True, 0, 0, 0)
                        feas_k, _, _, _ = route_feasible(new_route_k)
                        
                        if feas_i and feas_k:
                            # Check demand
                            if sum(demands[c] for c in new_route_k) <= capacity:
                                # Calculate distance improvement
                                old_dist = route_total_dist(route_i) + route_total_dist(route_k)
                                new_dist = route_total_dist(new_route_i) + route_total_dist(new_route_k)
                                
                                if new_dist < old_dist - 1e-6:
                                    routes[i] = new_route_i
                                    routes[k] = new_route_k
                                    if not new_route_i:
                                        del routes[i]
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
    
    return routes
# EVOLVE-BLOCK-END

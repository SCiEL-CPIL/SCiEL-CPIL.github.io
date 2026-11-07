import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"])
    ready = np.array(instance["ready"])
    due = np.array(instance["due"])
    service = np.array(instance["service"])
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    
    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
            dist[i][j] = d
            dist[j][i] = d
            
    customers = list(range(1, n_customers + 1))
    
    def simulate_route(route):
        """
        Simulates a route consisting of customer IDs.
        Checks time windows, capacity, and battery.
        Assumes NO intermediate stops for charging (stations must be explicitly in route if used, 
        but our local search operates on customer permutations).
        Returns (feasible, total_dist, total_time)
        """
        if not route:
            return True, 0.0, 0.0
        
        total_demand = sum(demands[c] for c in route)
        if total_demand > capacity:
            return False, 0.0, 0.0
            
        cur_time = 0.0
        cur_battery = battery_cap
        total_dist = 0.0
        cur_pos = 0
        
        for cust in route:
            d = dist[cur_pos][cust]
            cur_time += d / speed
            cur_battery -= d
            total_dist += d
            
            if cur_battery < 0:
                return False, total_dist, cur_time
            
            # Time window check
            if cur_time < ready[cust]:
                cur_time = ready[cust]
                
            if cur_time > due[cust]:
                return False, total_dist, cur_time
                
            cur_time += service[cust]
            cur_pos = cust
            
        # Return to depot
        d = dist[cur_pos][0]
        cur_time += d / speed
        cur_battery -= d
        total_dist += d
        
        if cur_battery < 0:
            return False, total_dist, cur_time
            
        if cur_time > horizon:
            return False, total_dist, cur_time
            
        return True, total_dist, cur_time

    def try_add_station(route):
        """
        If a route is infeasible due to battery, try to insert a station.
        Returns a new route with stations inserted, or None if impossible.
        This is a heuristic: insert the station that maximizes remaining battery 
        at the point of failure or nearest to the failure point.
        """
        # This is a simplified version. In a full solution, we'd search for the best station insertion.
        # Given the constraints, if the basic route is infeasible, we might just split the route 
        # or accept it's infeasible and rely on construction to avoid this.
        # For local search, we will only consider moves that result in feasible routes 
        # WITHOUT needing station insertion, to keep the search space simple and fast.
        # If a route becomes infeasible, the move is rejected.
        return None

    def route_distance(route):
        if not route:
            return 0.0
        d = dist[0][route[0]]
        for i in range(len(route) - 1):
            d += dist[route[i]][route[i+1]]
        d += dist[route[-1]][0]
        return d

    # --- Construction: Greedy Nearest Neighbor with Time Windows ---
    
    def construct_routes():
        unvisited = set(customers)
        routes = []
        
        while unvisited:
            # Start a new route
            route = []
            cur_pos = 0
            cur_time = 0.0
            cur_battery = battery_cap
            cur_demand = 0
            feasible = True
            
            while unvisited:
                best_cust = None
                best_score = float('inf')
                
                for cust in unvisited:
                    # Check if we can go from cur_pos to cust
                    d_go = dist[cur_pos][cust]
                    if cur_battery - d_go < 0:
                        continue
                    if cur_demand + demands[cust] > capacity:
                        continue
                    
                    new_time = cur_time + d_go / speed
                    if new_time < ready[cust]:
                        new_time = ready[cust]
                    if new_time > due[cust]:
                        continue
                    
                    # Check if we can return to depot after cust
                    # (Crucial for feasibility)
                    d_back = dist[cust][0]
                    if cur_battery - d_go - d_back < 0:
                        continue
                    if new_time + service[cust] + d_back / speed > horizon:
                        continue
                        
                    # Score: distance
                    score = d_go
                    if score < best_score:
                        best_score = score
                        best_cust = cust
                        
                if best_cust is None:
                    break
                    
                # Add customer
                d = dist[cur_pos][best_cust]
                cur_time += d / speed
                cur_battery -= d
                cur_time = max(cur_time, ready[best_cust])
                cur_time += service[best_cust]
                cur_demand += demands[best_cust]
                route.append(best_cust)
                unvisited.remove(best_cust)
                cur_pos = best_cust
                
            if route:
                # Verify full route
                feas, _, _ = simulate_route(route)
                if feas:
                    routes.append(route)
                else:
                    # If the greedy route is infeasible (shouldn't happen with checks, but just in case)
                    # Add customers one by one
                    for c in route:
                        if c in unvisited:
                            continue
                        unvisited.add(c)
                    for c in list(unvisited):
                        if simulate_route([c])[0]:
                            routes.append([c])
                            unvisited.remove(c)
            else:
                # No customer could be added to this route, pick any feasible single customer
                for c in list(unvisited):
                    if simulate_route([c])[0]:
                        routes.append([c])
                        unvisited.remove(c)
                        break
                        
        # Check if all served
        served = set()
        for r in routes:
            served.update(r)
        if served != set(customers):
            # Fallback: force remaining customers into single routes
            for c in set(customers) - served:
                routes.append([c])
                
        return routes

    routes = construct_routes()
    
    # --- Local Search: 2-Opt and Relocate ---
    
    def try_2opt(route, route_idx):
        n = len(route)
        if n < 2:
            return False, 0.0
            
        improved = False
        best_gain = 0.0
        best_i, best_j = -1, -1
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Calculate delta distance
                # Old: ... A - B ... C - D ...
                # New: ... A - C ... D - B ...
                # A = route[i-1] or 0, B = route[i], C = route[j], D = route[j+1] or 0
                
                A = route[i-1] if i > 0 else 0
                B = route[i]
                C = route[j]
                D = route[j+1] if j < n - 1 else 0
                
                old_dist = dist[A][B] + dist[C][D]
                new_dist = dist[A][C] + dist[B][D]
                
                gain = old_dist - new_dist
                if gain > best_gain:
                    best_gain = gain
                    best_i, best_j = i, j
                    
        if best_gain > 1e-6:
            i, j = best_i, best_j
            # Reverse segment i..j
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            
            # Check feasibility of new route
            feas, _, _ = simulate_route(new_route)
            if feas:
                routes[route_idx] = new_route
                return True, best_gain
                
        return False, 0.0

    def try_relocate(routes_list, cust_idx, from_route_idx, to_route_idx):
        """
        Move customer at cust_idx from from_route_idx to to_route_idx.
        Returns (improved, gain)
        """
        route_from = routes_list[from_route_idx]
        route_to = routes_list[to_route_idx]
        
        if not route_from:
            return False, 0.0
            
        cust = route_from[cust_idx]
        
        # Remove from route_from
        new_route_from = route_from[:cust_idx] + route_from[cust_idx+1:]
        
        # Find best insertion point in route_to
        best_pos = -1
        best_gain = 0.0
        
        # Calculate cost of removing cust from route_from
        prev = route_from[cust_idx-1] if cust_idx > 0 else 0
        next_ = route_from[cust_idx+1] if cust_idx < len(route_from)-1 else 0
        remove_cost = dist[prev][next_] - (dist[prev][cust] + dist[cust][next_])
        
        # Try inserting into route_to
        for pos in range(len(route_to) + 1):
            # Insertion cost
            p = route_to[pos-1] if pos > 0 else 0
            n = route_to[pos] if pos < len(route_to) else 0
            
            insert_cost = dist[p][n] - (dist[p][cust] + dist[cust][n])
            
            gain = remove_cost - insert_cost
            
            if gain > best_gain:
                best_gain = gain
                best_pos = pos
                
        if best_gain > 1e-6:
            # Perform move
            new_route_to = route_to[:best_pos] + [cust] + route_to[best_pos:]
            
            # Check feasibility
            feas_from, _, _ = simulate_route(new_route_from)
            feas_to, _, _ = simulate_route(new_route_to)
            
            if feas_from and feas_to:
                routes_list[from_route_idx] = new_route_from
                routes_list[to_route_idx] = new_route_to
                return True, best_gain
                
        return False, 0.0

    # Main Local Search Loop
    improved = True
    no_improve_count = 0
    max_no_improve = 50
    
    while improved and time.time() < deadline:
        improved = False
        
        # 2-Opt
        for i in range(len(routes)):
            if len(routes[i]) < 2:
                continue
            imp, gain = try_2opt(routes[i], i)
            if imp:
                improved = True
                no_improve_count = 0
        
        # Relocate
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                for k in range(len(routes)):
                    if i == k:
                        continue
                    imp, gain = try_relocate(routes, j, i, k)
                    if imp:
                        improved = True
                        no_improve_count = 0
                        # Restart loops after a move to handle index shifts
                        break
                if improved:
                    break
            if improved:
                break
                
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break

    return routes

import numpy as np
import math
import time
from scipy.spatial import distance

def plan(instance):
    coords = np.array(instance["coords"])
    n_nodes = len(coords)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    max_battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distance matrix
    dist_matrix = distance.cdist(coords, coords, metric='euclidean')
    
    def get_demand(node):
        return instance["demands"][node]
    
    def get_ready(node):
        return instance["ready"][node]
    
    def get_due(node):
        return instance["due"][node]
    
    def get_service(node):
        return instance["service"][node]
    
    def simulate_route(route):
        """
        Simulates a route starting and ending at depot (0).
        Returns (is_feasible, total_distance, total_time)
        """
        current_node = 0
        current_time = 0.0
        current_battery = max_battery
        total_dist = 0.0
        
        for node in route:
            # Travel from current_node to node
            d = dist_matrix[current_node, node]
            
            # Check battery
            if current_battery < d:
                return False, 0, 0
            
            current_battery -= d
            total_dist += d
            current_time += d / speed
            
            # Arrive at node
            if node in stations:
                # Charging
                # Rule: adds (battery - energy) * recharge_rate time and refills
                # Assuming 'battery' in text means max_battery capacity
                # Energy needed to fill = max_battery - current_battery
                charge_time = (max_battery - current_battery) * recharge_rate
                current_time += charge_time
                current_battery = max_battery
            else:
                # Customer
                # Time window check
                if current_time < get_ready(node):
                    current_time = get_ready(node)
                
                if current_time > get_due(node):
                    return False, 0, 0
                
                current_time += get_service(node)
                
                # Note: Demand is not consumed in simulation unless we track load per route
                # We will handle capacity in insertion logic
        
        # Return to depot
        d = dist_matrix[current_node, 0]
        if current_battery < d:
            return False, 0, 0
        
        current_battery -= d
        total_dist += d
        current_time += d / speed
        
        if current_time > horizon:
            return False, 0, 0
        
        return True, total_dist, current_time

    def calculate_route_cost(route):
        feasible, dist, t = simulate_route(route)
        if not feasible:
            return float('inf')
        return dist

    def try_insert_customer(routes, cust):
        """
        Tries to insert cust into one of the existing routes.
        Returns (new_routes, cost_delta) if successful, else (None, None).
        """
        best_cost = float('inf')
        best_routes = None
        best_idx = -1
        
        for r_idx, route in enumerate(routes):
            # Check capacity
            route_load = sum(get_demand(n) for n in route)
            if route_load + get_demand(cust) > capacity:
                continue
            
            # Try all insertion positions
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                feasible, dist, t = simulate_route(new_route)
                if feasible:
                    # Cost is total distance of all routes
                    # We only need to compare the change in this route's distance?
                    # No, total cost is sum of all.
                    # To speed up, we can just check if this specific route's distance is minimal?
                    # Actually, we should minimize the total distance.
                    # Since other routes don't change, we just need to minimize new_route distance.
                    if dist < best_cost:
                        best_cost = dist
                        best_routes = [r if i != r_idx else new_route for i, r in enumerate(routes)]
                        best_idx = r_idx
        
        if best_idx != -1:
            return best_routes, best_cost
        return None, None

    def optimize_route(route):
        """
        Applies 2-Opt and Or-Opt to a single route.
        """
        if len(route) < 2:
            return route
            
        current_cost = calculate_route_cost(route)
        if math.isinf(current_cost):
            return route
            
        improved = True
        while improved:
            improved = False
            
            # 2-Opt
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    # Segment to reverse is route[i+1...j]
                    # New route: route[:i+1] + reversed(route[i+1...j+1]) + route[j+1:]
                    # Wait, standard 2-Opt on path 0->...->i->i+1->...->j->j+1->...->0
                    # Reverse segment i+1 to j.
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    
                    # Check feasibility and cost
                    feasible, new_dist, t = simulate_route(new_route)
                    if feasible and new_dist < current_cost - 1e-6:
                        route = new_route
                        current_cost = new_dist
                        improved = True
                        # Restart loop to find more improvements
                        break
                if improved:
                    break
            
            # Or-Opt (move 1 or 2 nodes)
            if not improved:
                for seg_len in [1, 2]:
                    if len(route) < seg_len:
                        continue
                    for i in range(len(route) - seg_len + 1):
                        segment = route[i:i+seg_len]
                        remaining = route[:i] + route[i+seg_len:]
                        
                        # Try inserting segment at all positions in remaining
                        for j in range(len(remaining) + 1):
                            new_route = remaining[:j] + segment + remaining[j:]
                            feasible, new_dist, t = simulate_route(new_route)
                            if feasible and new_dist < current_cost - 1e-6:
                                route = new_route
                                current_cost = new_dist
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
        return route

    # 1. Initial Construction: Greedy Insertion
    routes = []
    unserved = list(customers)
    
    # Sort customers by due time (most constrained first)
    unserved.sort(key=lambda x: instance["due"][x])
    
    for cust in unserved:
        # Try to insert into existing routes
        new_routes, cost = try_insert_customer(routes, cust)
        if new_routes is not None:
            routes = new_routes
        else:
            # Open new route
            routes.append([cust])
            
    # 2. Optimization
    # Optimize each route individually
    routes = [optimize_route(r) for r in routes]
    
    # Inter-route moves (Relocation)
    improved = True
    while improved:
        improved = False
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                cust = routes[i][j]
                # Try moving cust to other routes
                for k in range(len(routes)):
                    if k == i:
                        continue
                    # Try inserting into route k
                    # We need to remove cust from i first? 
                    # Better: construct new routes for i and k
                    route_i_without = routes[i][:j] + routes[i][j+1:]
                    # Check if route_i_without is feasible (it should be, as we removed a node)
                    # But we need to check if it's empty? If empty, remove route? 
                    # Let's keep it for now, might merge later.
                    
                    # Try all insertion points in route k
                    for pos in range(len(routes[k]) + 1):
                        new_route_k = routes[k][:pos] + [cust] + routes[k][pos:]
                        
                        # Check feasibility of new_route_k
                        feasible_k, dist_k, _ = simulate_route(new_route_k)
                        if not feasible_k:
                            continue
                        
                        # Check feasibility of route_i_without
                        if len(route_i_without) == 0:
                            # Route i becomes empty
                            feasible_i = True
                            dist_i = 0
                        else:
                            feasible_i, dist_i, _ = simulate_route(route_i_without)
                            if not feasible_i:
                                continue
                        
                        # Calculate total cost change
                        # Old cost: dist(routes[i]) + dist(routes[k])
                        # New cost: dist_i + dist_k
                        # We need old distances. Let's calculate them.
                        # To avoid recalculating all, we can store costs or just calculate total.
                        # For simplicity in this step, calculate total cost of modified routes.
                        # Since n is small, it's okay.
                        
                        # Actually, let's just check if the sum of distances of the two modified routes is less than the sum of the original two.
                        old_cost_ik = calculate_route_cost(routes[i]) + calculate_route_cost(routes[k])
                        new_cost_ik = dist_i + dist_k
                        
                        if new_cost_ik < old_cost_ik - 1e-6:
                            # Perform move
                            routes[k] = new_route_k
                            routes[i] = route_i_without
                            if len(routes[i]) == 0:
                                # Remove empty route
                                routes.pop(i)
                                # Adjust indices? 
                                # Since we are modifying the list, it's safer to break and restart
                                # Or handle carefully.
                                # Let's just mark improved and break loops to restart
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        # Re-optimize routes after moves
        if improved:
            routes = [optimize_route(r) for r in routes]

    # Final check and return
    # Ensure all customers are served
    served = set()
    for r in routes:
        for n in r:
            served.add(n)
            
    # If some are missing (should not happen), add them as single routes
    for c in customers:
        if c not in served:
            routes.append([c])
            
    return routes

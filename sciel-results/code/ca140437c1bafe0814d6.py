"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import distance


def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    coords = np.array(instance["coords"], dtype=float)
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
    
    # Precompute distance matrix
    n_nodes = len(coords)
    dist_matrix = distance.cdist(coords, coords, 'euclidean')
    
    # Customer indices
    customers = list(range(1, n_customers + 1))
    
    # Sort customers by due time for initial ordering
    customers_sorted = sorted(customers, key=lambda v: due[v])
    
    best_routes = None
    best_distance = float('inf')
    
    def compute_route_distance(route):
        """Compute total distance of a route including depot legs."""
        if not route:
            return 0.0
        total = dist_matrix[0, route[0]]
        for i in range(len(route) - 1):
            total += dist_matrix[route[i], route[i + 1]]
        total += dist_matrix[route[-1], 0]
        return total
    
    def is_feasible_route(route):
        """Check if a route is feasible (battery, time windows, capacity)."""
        if not route:
            return True
        
        # Check capacity
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False
        
        # Simulate the route
        current_time = 0.0
        current_battery = battery
        current_node = 0
        
        for v in route:
            d = dist_matrix[current_node, v]
            current_time += d / speed
            current_battery -= d
            
            if current_battery < -1e-9:
                return False
            
            # Time window check
            current_time = max(current_time, ready[v])
            if current_time > due[v] + 1e-9:
                return False
            
            # Service time
            current_time += service[v]
            current_node = v
        
        # Return to depot
        d = dist_matrix[current_node, 0]
        current_time += d / speed
        current_battery -= d
        
        if current_battery < -1e-9:
            return False
        if current_time > horizon + 1e-9:
            return False
        
        return True
    
    def construct_routes_greedy():
        """Construct routes using a greedy nearest-neighbor approach with feasibility checks."""
        remaining = set(customers)
        routes = []
        
        while remaining:
            if time.time() > deadline:
                return None
            
            # Start new route from depot
            route = []
            current_node = 0
            current_time = 0.0
            current_battery = battery
            route_demand = 0
            
            # Try to add customers to this route
            while remaining:
                # Find feasible next customer
                best_cust = None
                best_dist = float('inf')
                
                for v in sorted(remaining, key=lambda x: dist_matrix[current_node, x]):
                    d = dist_matrix[current_node, v]
                    
                    # Check if we can reach v
                    if current_battery - d < -1e-9:
                        continue
                    
                    # Check capacity
                    if route_demand + demands[v] > capacity:
                        continue
                    
                    # Check time window
                    arrival_time = current_time + d / speed
                    arrival_time = max(arrival_time, ready[v])
                    if arrival_time > due[v] + 1e-9:
                        continue
                    
                    best_cust = v
                    best_dist = d
                    break
                
                if best_cust is None:
                    break
                
                # Add customer
                route.append(best_cust)
                remaining.discard(best_cust)
                route_demand += demands[best_cust]
                
                # Update state
                d = dist_matrix[current_node, best_cust]
                current_time += d / speed
                current_time = max(current_time, ready[best_cust])
                current_time += service[best_cust]
                current_battery -= d
                current_node = best_cust
            
            if route:
                routes.append(route)
        
        # Verify all routes are feasible
        for route in routes:
            if not is_feasible_route(route):
                return None
        
        return routes
    
    # Try greedy construction
    routes = construct_routes_greedy()
    if routes is not None:
        best_routes = routes
        best_distance = sum(compute_route_distance(r) for r in routes)
    
    # Try to improve with 2-opt
    if best_routes is not None and time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for route_idx, route in enumerate(best_routes):
                if time.time() > deadline - 5.0:
                    break
                # Try 2-opt within each route
                for i in range(len(route) - 1):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(i + 1, len(route)):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if is_feasible_route(new_route):
                            old_dist = compute_route_distance(route)
                            new_dist = compute_route_distance(new_route)
                            if new_dist < old_dist - 1e-9:
                                best_routes[route_idx] = new_route
                                best_distance = sum(compute_route_distance(r) for r in best_routes)
                                improved = True
    
    # If no valid routes found, fall back to single-customer routes
    if best_routes is None:
        best_routes = [[v] for v in customers_sorted]
    
    return best_routes
# EVOLVE-BLOCK-END

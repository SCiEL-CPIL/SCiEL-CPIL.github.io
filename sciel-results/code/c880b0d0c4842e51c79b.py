"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _euclidean(coords, i, j):
    dx = coords[i, 0] - coords[j, 0]
    dy = coords[i, 1] - coords[j, 1]
    return np.sqrt(dx * dx + dy * dy)


def _build_distance_matrix(coords, n_nodes):
    """Build full Euclidean distance matrix."""
    d = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                d[i, j] = _euclidean(coords, i, j)
    return d


def _is_feasible_route(route, instance, dist_matrix):
    """
    Check if a route is feasible:
    - Battery never below 0
    - Time windows respected
    - Capacity respected
    - Return to depot by horizon
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    # Check capacity
    total_demand = sum(demands[v] for v in route if v not in stations)
    if total_demand > capacity:
        return False
    
    current_time = 0.0
    current_battery = float(battery_cap)
    current_node = 0  # depot
    
    for v in route:
        d = dist_matrix[current_node, v]
        current_time += d / speed
        current_battery -= d
        
        if current_battery < -1e-9:
            return False
        
        if v in stations:
            # Recharge
            energy_needed = battery_cap - current_battery
            recharge_time = energy_needed * recharge_rate
            current_time += recharge_time
            current_battery = float(battery_cap)
        else:
            # Customer
            if current_time < ready[v]:
                current_time = float(ready[v])
            if current_time > due[v] + 1e-9:
                return False
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


def _route_distance(route, dist_matrix):
    """Calculate total Euclidean distance of a route including depot legs."""
    total = dist_matrix[0, route[0]]
    for i in range(len(route) - 1):
        total += dist_matrix[route[i], route[i + 1]]
    total += dist_matrix[route[-1], 0]
    return total


def _solve_one_route(instance, dist_matrix, customers, stations):
    """
    Find a feasible route serving as many customers as possible from the given list.
    Uses a greedy insertion approach.
    """
    if not customers:
        return []
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    station_set = set(stations)
    
    # Sort customers by due time for initial ordering
    sorted_customers = sorted(customers, key=lambda v: due[v])
    
    # Start with the customer with earliest due time
    route = [sorted_customers[0]]
    remaining = set(customers[1:])
    
    # Try to insert customers one by one at the best position
    while remaining:
        best_gain = -1
        best_customer = None
        best_pos = None
        
        for v in list(remaining):
            if demands[v] > capacity - sum(demands[c] for c in route if c not in station_set):
                continue
            
            # Try inserting v at each position in the route
            for pos in range(len(route) + 1):
                candidate_route = route[:pos] + [v] + route[pos:]
                
                # Quick check: is this feasible?
                if _is_feasible_route(candidate_route, instance, dist_matrix):
                    # Calculate distance increase
                    new_dist = _route_distance(candidate_route, dist_matrix)
                    old_dist = _route_distance(route, dist_matrix)
                    gain = new_dist - old_dist
                    # We want to minimize distance, so lower gain is better
                    if best_gain == -1 or gain < best_gain:
                        best_gain = gain
                        best_customer = v
                        best_pos = pos
        
        if best_customer is not None:
            route = route[:best_pos] + [best_customer] + route[best_pos:]
            remaining.remove(best_customer)
        else:
            break
    
    return route


def plan(instance):
    coords = np.array(instance["coords"], dtype=float)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = instance["stations"]
    n_nodes = len(coords)
    
    # Build distance matrix
    dist_matrix = _build_distance_matrix(coords, n_nodes)
    
    # Greedy set cover: repeatedly find a good feasible route, remove those customers
    routes = []
    remaining_customers = set(customers)
    
    # Sort customers by due time for priority
    customer_list = sorted(remaining_customers, key=lambda v: instance["due"][v])
    
    while remaining_customers:
        # Try to build a route starting from the customer with earliest due time
        # Use the greedy insertion method
        route = _solve_one_route(instance, dist_matrix, list(remaining_customers), stations)
        
        if not route:
            # Fallback: serve each remaining customer in its own route
            for v in sorted(remaining_customers, key=lambda v: instance["due"][v]):
                routes.append([v])
                remaining_customers.remove(v)
            break
        
        # Remove served customers from remaining
        for v in route:
            if v in remaining_customers:
                remaining_customers.remove(v)
        
        routes.append(route)
    
    return routes
# EVOLVE-BLOCK-END

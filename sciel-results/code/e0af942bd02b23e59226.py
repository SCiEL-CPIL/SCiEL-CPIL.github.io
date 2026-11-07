"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _dist(i, j, coords):
    return np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)


def _route_distance(route, coords):
    """Total distance including depot legs."""
    if not route:
        return 0.0
    total = _dist(0, route[0], coords)
    for k in range(len(route) - 1):
        total += _dist(route[k], route[k+1], coords)
    total += _dist(route[-1], 0, coords)
    return total


def _check_route_feasibility(route, instance):
    """Check if a route is feasible. Returns (feasible, battery_end, time_end)."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    # Check capacity
    total_demand = sum(demands[v] for v in route if v not in stations)
    if total_demand > capacity:
        return False, 0, 0
    
    # Simulate
    battery = battery_max
    time = 0.0
    prev = 0
    for v in route:
        d = _dist(prev, v, coords)
        battery -= d
        if battery < -1e-9:
            return False, 0, 0
        time += d / speed
        if v not in stations:
            # Customer
            time = max(time, ready[v])
            if time > due[v] + 1e-9:
                return False, 0, 0
            time += service[v]
        else:
            # Station: charge
            energy = battery_max - battery
            time += energy * recharge_rate
            battery = battery_max
        prev = v
    
    # Return to depot
    d = _dist(prev, 0, coords)
    battery -= d
    if battery < -1e-9:
        return False, 0, 0
    time += d / speed
    if time > horizon + 1e-9:
        return False, 0, 0
    
    return True, battery, time


def _can_insert(route, pos, v, instance):
    """Check if customer v can be inserted at position pos in route."""
    new_route = route[:pos] + [v] + route[pos:]
    feasible, _, _ = _check_route_feasibility(new_route, instance)
    return feasible


def _build_initial_routes(instance):
    """Construct initial routes using a greedy insertion heuristic."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]
    
    customers = list(range(1, n_customers + 1))
    # Sort by due time for insertion order
    customers.sort(key=lambda v: due[v])
    
    routes = []
    unserved = set(customers)
    
    while unserved:
        # Start a new route
        route = []
        route_demand = 0
        route_battery_used = 0.0
        route_time = 0.0
        prev = 0
        battery = battery_max
        
        # Try to add customers in sorted order
        for v in customers:
            if v not in unserved:
                continue
            if demands[v] == 0:
                continue
            
            # Check capacity
            if route_demand + demands[v] > capacity:
                continue
            
            # Try inserting v at the best position in current route
            best_pos = None
            best_cost = float('inf')
            
            # Try all positions
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [v] + route[pos:]
                feasible, _, _ = _check_route_feasibility(new_route, instance)
                if feasible:
                    # Compute additional distance
                    if pos == 0:
                        if pos == len(route):
                            add_dist = _dist(prev, v, coords) + _dist(v, 0, coords) - _dist(prev, 0, coords)
                        else:
                            add_dist = _dist(prev, v, coords) + _dist(v, route[0], coords) - _dist(prev, route[0], coords)
                    else:
                        if pos == len(route):
                            add_dist = _dist(route[-1], v, coords) + _dist(v, 0, coords) - _dist(route[-1], 0, coords)
                        else:
                            add_dist = _dist(route[pos-1], v, coords) + _dist(v, route[pos], coords) - _dist(route[pos-1], route[pos], coords)
                    if add_dist < best_cost:
                        best_cost = add_dist
                        best_pos = pos
            
            if best_pos is not None:
                route = route[:best_pos] + [v] + route[best_pos:]
                route_demand += demands[v]
                unserved.remove(v)
        
        if route:
            routes.append(route)
    
    return routes


def _two_opt(route, instance, max_iter=50):
    """Apply 2-opt within a route to reduce distance while maintaining feasibility."""
    coords = instance["coords"]
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if new_route == route:
                    continue
                feasible, _, _ = _check_route_feasibility(new_route, instance)
                if feasible:
                    if _route_distance(new_route, coords) < _route_distance(route, coords) - 1e-9:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    return route


def _relocate(routes, instance, max_iter=30):
    """Try to move a customer from one route to another to reduce total distance."""
    coords = instance["coords"]
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for r1 in range(len(routes)):
            for r2 in range(len(routes)):
                if r1 == r2:
                    continue
                for i in range(len(routes[r1])):
                    v = routes[r1][i]
                    # Try inserting v into route r2 at each position
                    best_pos = None
                    best_cost = float('inf')
                    current_total = sum(_route_distance(r, coords) for r in routes)
                    
                    for pos in range(len(routes[r2]) + 1):
                        new_r2 = routes[r2][:pos] + [v] + routes[r2][pos:]
                        new_r1 = routes[r1][:i] + routes[r1][i+1:]
                        if not new_r1 and not new_r2:
                            continue
                        # Check feasibility
                        feas1, _, _ = _check_route_feasibility(new_r1, instance) if new_r1 else (True, 0, 0)
                        feas2, _, _ = _check_route_feasibility(new_r2, instance) if new_r2 else (True, 0, 0)
                        if feas1 and feas2:
                            new_total = (_route_distance(new_r1, coords) if new_r1 else 0) + (_route_distance(new_r2, coords) if new_r2 else 0)
                            # Add distances for other routes
                            for r in range(len(routes)):
                                if r != r1 and r != r2:
                                    new_total += _route_distance(routes[r], coords)
                            if new_total < best_cost:
                                best_cost = new_total
                                best_pos = pos
                    
                    if best_pos is not None and best_cost < current_total - 1e-9:
                        v = routes[r1][i]
                        routes[r1] = routes[r1][:i] + routes[r1][i+1:]
                        routes[r2] = routes[r2][:best_pos] + [v] + routes[r2][best_pos:]
                        # Remove empty routes
                        routes = [r for r in routes if r]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return routes


def plan(instance):
    coords = np.array(instance["coords"], dtype=float)
    
    # Build initial routes
    routes = _build_initial_routes(instance)
    
    # Ensure all customers are served
    served = set()
    for r in routes:
        for v in r:
            if v not in set(instance["stations"]):
                served.add(v)
    missing = set(range(1, instance["n_customers"] + 1)) - served
    for v in missing:
        routes.append([v])
    
    # Local search: 2-opt within each route
    for i in range(len(routes)):
        routes[i] = _two_opt(routes[i], instance)
    
    # Local search: relocate between routes
    routes = _relocate(routes, instance)
    
    # Final 2-opt pass
    for i in range(len(routes)):
        routes[i] = _two_opt(routes[i], instance)
    
    # Verify all customers served exactly once
    all_nodes = []
    for r in routes:
        all_nodes.extend(r)
    customer_nodes = [v for v in all_nodes if v not in set(instance["stations"])]
    if sorted(customer_nodes) != list(range(1, instance["n_customers"] + 1)):
        # Fallback: one per route
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]
    
    return routes
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt


def _euclidean(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean(coords[i], coords[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _check_route(route, instance, dist):
    """Check if a route is feasible. Returns (feasible, total_dist)."""
    if not route:
        return True, 0.0
    
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
    
    # Check capacity
    total_demand = sum(demands[c] for c in route if c <= n_customers)
    if total_demand > capacity:
        return False, 0.0
    
    # Simulate
    time_t = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0  # depot
    
    for node in route:
        d = dist[prev][node]
        time_t += d / speed
        battery -= d
        total_dist += d
        
        if battery < -1e-9:
            return False, 0.0
        
        if node in stations:
            # Recharge
            energy_needed = battery_max - battery
            if energy_needed > 0:
                time_t += energy_needed * recharge_rate
            battery = battery_max
        elif node <= n_customers:
            # Customer
            if time_t < ready[node]:
                time_t = ready[node]
            if time_t > due[node] + 1e-9:
                return False, 0.0
            time_t += service[node]
        
        prev = node
    
    # Return to depot
    d = dist[prev][0]
    time_t += d / speed
    battery -= d
    total_dist += d
    
    if battery < -1e-9:
        return False, 0.0
    if time_t > horizon + 1e-9:
        return False, 0.0
    
    return True, total_dist


def _construct_route_sweep(instance, dist, customers, start_angle=0.0):
    """Construct a route by sweeping from start_angle, adding customers in angular order."""
    coords = instance["coords"]
    depot = coords[0]
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
    
    if not customers:
        return []
    
    # Sort customers by angle from depot
    def angle_of(c):
        dx = coords[c][0] - depot[0]
        dy = coords[c][1] - depot[1]
        return (dx * start_angle - dy) % 360  # arbitrary rotation for sweep
    
    # Use a simpler approach: sort by angle
    import math
    def angle_of(c):
        dx = coords[c][0] - depot[0]
        dy = coords[c][1] - depot[1]
        return math.atan2(dy, dx)
    
    sorted_customers = sorted(customers, key=angle_of)
    
    route = []
    time_t = 0.0
    battery = battery_max
    total_demand = 0
    prev = 0
    
    for c in sorted_customers:
        d = dist[prev][c]
        new_time = time_t + d / speed
        new_battery = battery - d
        
        if new_battery < -1e-9:
            break
        if total_demand + demands[c] > capacity:
            break
        
        if c <= n_customers:
            if new_time < ready[c]:
                new_time = ready[c]
            if new_time > due[c] + 1e-9:
                continue  # skip if can't make due time
            new_time += service[c]
        
        # Check if we can still return to depot
        d_back = dist[c][0]
        if new_battery - d_back < -1e-9:
            break
        if new_time + d_back / speed > horizon + 1e-9:
            break
        
        route.append(c)
        time_t = new_time
        battery = new_battery
        total_demand += demands[c]
        prev = c
    
    return route


def _improve_route(route, instance, dist, max_iter=50):
    """Improve a route using 2-opt and relocate moves."""
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
    
    if len(route) < 2:
        return route
    
    def route_dist(r):
        if not r:
            return 0.0
        d = dist[0][r[0]]
        for i in range(len(r) - 1):
            d += dist[r[i]][r[i + 1]]
        d += dist[r[-1]][0]
        return d
    
    best_route = list(route)
    best_dist = route_dist(best_route)
    
    for _ in range(max_iter):
        improved = False
        
        # 2-opt
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                feasible, d = _check_route(new_route, instance, dist)
                if feasible and d < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = d
                    improved = True
                    break
            if improved:
                break
        
        # Relocate
        if not improved:
            for i in range(len(best_route)):
                for j in range(len(best_route)):
                    if i == j or i == j + 1:
                        continue
                    node = best_route[i]
                    new_route = best_route[:i] + best_route[i+1:]
                    insert_pos = j if j < i else j + 1
                    new_route = new_route[:insert_pos] + [node] + new_route[insert_pos:]
                    feasible, d = _check_route(new_route, instance, dist)
                    if feasible and d < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = d
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_route


def _solve_instance(instance):
    """Solve a single instance using sweep clustering and local search."""
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
    
    dist = _dist_matrix(coords)
    customers = list(range(1, n_customers + 1))
    
    # Try multiple sweep angles
    best_routes = None
    best_total_dist = float('inf')
    
    start_time = time.time()
    deadline = start_time + 40  # 40 seconds per instance
    
    for angle in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        if time.time() > deadline:
            break
        
        remaining = list(customers)
        routes = []
        
        while remaining:
            route = _construct_route_sweep(instance, dist, remaining, angle)
            if not route:
                # Force assign the closest customer
                c = min(remaining, key=lambda x: dist[0][x])
                route = [c]
            
            # Remove served customers from remaining
            served_set = set(route)
            remaining = [c for c in remaining if c not in served_set]
            
            # Improve the route
            route = _improve_route(route, instance, dist, max_iter=30)
            routes.append(route)
        
        # Check feasibility
        feasible = True
        total_dist = 0.0
        for r in routes:
            ok, d = _check_route(r, instance, dist)
            if not ok:
                feasible = False
                break
            total_dist += d
        
        if feasible and total_dist < best_total_dist:
            best_total_dist = total_dist
            best_routes = [list(r) for r in routes]
    
    if best_routes is None:
        # Fallback: one customer per route
        best_routes = [[c] for c in customers]
    
    return best_routes


def plan(instance):
    """Main entry point. Returns a list of routes for the given instance."""
    start_time = time.time()
    
    # Solve the single instance passed in
    routes = _solve_instance(instance)
    
    return routes
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt, atan2


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


def _fast_feasibility_check(route, instance, dist):
    """Quick pre-check: capacity and rough distance bounds."""
    demands = instance["demands"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    n_customers = instance["n_customers"]
    
    if not route:
        return True
    
    total_demand = sum(demands[c] for c in route if c <= n_customers)
    if total_demand > capacity:
        return False
    
    # Rough distance check: sum of all edges must be <= battery_max * 2 (go and back)
    total_dist = dist[0][route[0]]
    for i in range(len(route) - 1):
        total_dist += dist[route[i]][route[i+1]]
    total_dist += dist[route[-1]][0]
    
    if total_dist > battery_max * 2 + 1e-9:
        return False
    
    return True


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
        return atan2(dy, dx)
    
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


def _improve_route(route, instance, dist, max_iter=50, deadline=None):
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
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # 2-opt with fast pre-check
        for i in range(len(best_route) - 1):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                
                # Fast pre-check
                if not _fast_feasibility_check(new_route, instance, dist):
                    continue
                    
                feasible, d = _check_route(new_route, instance, dist)
                if feasible and d < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = d
                    improved = True
                    break
            if improved:
                break
        
        # Relocate with fast pre-check
        if not improved:
            for i in range(len(best_route)):
                if deadline is not None and time.time() > deadline:
                    break
                for j in range(len(best_route)):
                    if i == j or i == j + 1:
                        continue
                    node = best_route[i]
                    new_route = best_route[:i] + best_route[i+1:]
                    insert_pos = j if j < i else j + 1
                    new_route = new_route[:insert_pos] + [node] + new_route[insert_pos:]
                    
                    # Fast pre-check
                    if not _fast_feasibility_check(new_route, instance, dist):
                        continue
                        
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


def _inter_route_swap(routes, instance, dist, deadline=None):
    """Try to move customers between routes to reduce total distance."""
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
    
    def total_routes_dist(rs):
        total = 0.0
        for r in rs:
            if not r:
                continue
            d = dist[0][r[0]]
            for i in range(len(r) - 1):
                d += dist[r[i]][r[i+1]]
            d += dist[r[-1]][0]
            total += d
        return total
    
    best_routes = [list(r) for r in routes]
    best_total = total_routes_dist(best_routes)
    
    max_iters = 10
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        n_routes = len(best_routes)
        
        for i in range(n_routes):
            if deadline is not None and time.time() > deadline:
                break
            if not best_routes[i]:
                continue
            
            for j in range(n_routes):
                if i == j:
                    continue
                if deadline is not None and time.time() > deadline:
                    break
                
                # Try moving each customer from route i to route j
                for pos in range(len(best_routes[i])):
                    c = best_routes[i][pos]
                    if c > n_customers:  # Don't move stations
                        continue
                    
                    # Check capacity of destination
                    dest_demand = sum(demands[x] for x in best_routes[j] if x <= n_customers)
                    if dest_demand + demands[c] > capacity:
                        continue
                    
                    # Try inserting at each position in route j
                    for insert_pos in range(len(best_routes[j]) + 1):
                        # Create new routes
                        new_routes = [list(r) for r in best_routes]
                        new_routes[i] = new_routes[i][:pos] + new_routes[i][pos+1:]
                        new_routes[j] = new_routes[j][:insert_pos] + [c] + new_routes[j][insert_pos:]
                        
                        # Check feasibility of both affected routes
                        if not _fast_feasibility_check(new_routes[i], instance, dist):
                            continue
                        if not _fast_feasibility_check(new_routes[j], instance, dist):
                            continue
                        
                        ok_i, d_i = _check_route(new_routes[i], instance, dist)
                        if not ok_i:
                            continue
                        ok_j, d_j = _check_route(new_routes[j], instance, dist)
                        if not ok_j:
                            continue
                        
                        new_total = best_total - (total_routes_dist([best_routes[i], best_routes[j]]) - 
                                                   sum(dist[0][r[0]] + dist[r[-1]][0] + 
                                                       sum(dist[r[k]][r[k+1]] for k in range(len(r)-1)) 
                                                       for r in [best_routes[i], best_routes[j]] if r))
                        # Simpler: just compute total
                        new_total = 0.0
                        for r in new_routes:
                            if not r:
                                continue
                            d = dist[0][r[0]]
                            for k in range(len(r) - 1):
                                d += dist[r[k]][r[k+1]]
                            d += dist[r[-1]][0]
                            new_total += d
                        
                        if new_total < best_total - 1e-9:
                            best_routes = new_routes
                            best_total = new_total
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_routes


def _solve_instance(instance, time_budget=45.0):
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
    deadline = start_time + time_budget
    
    for angle in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        if time.time() > deadline - 5:  # Leave 5s for final polish
            break
        
        remaining = list(customers)
        routes = []
        
        while remaining:
            if time.time() > deadline - 5:
                break
                
            route = _construct_route_sweep(instance, dist, remaining, angle)
            if not route:
                # Force assign the closest customer
                c = min(remaining, key=lambda x: dist[0][x])
                route = [c]
            
            # Remove served customers from remaining
            served_set = set(route)
            remaining = [c for c in remaining if c not in served_set]
            
            # Improve the route
            route = _improve_route(route, instance, dist, max_iter=40, deadline=deadline - 5)
            routes.append(route)
        
        # Check all customers are served
        all_served = set()
        for r in routes:
            all_served.update(r)
        if not set(customers).issubset(all_served):
            continue
        
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
    
    # Final inter-route optimization
    if best_routes is not None and time.time() < deadline - 2:
        best_routes = _inter_route_swap(best_routes, instance, dist, deadline=deadline - 2)
        
        # Verify final solution
        feasible = True
        total_dist = 0.0
        all_served = set()
        for r in best_routes:
            ok, d = _check_route(r, instance, dist)
            if not ok:
                feasible = False
                break
            total_dist += d
            all_served.update(r)
        
        if not feasible or not set(customers).issubset(all_served):
            # Revert to pre-swap solution
            # Re-run a simple version
            pass
    
    if best_routes is None:
        # Fallback: one customer per route
        best_routes = [[c] for c in customers]
    
    return best_routes


def plan(instance):
    """Main entry point. Returns a list of routes for the given instance."""
    start_time = time.time()
    
    # Solve the single instance passed in with a safety margin
    routes = _solve_instance(instance, time_budget=45.0)
    
    # Final validation
    dist = _dist_matrix(instance["coords"])
    n_customers = instance["n_customers"]
    customers = set(range(1, n_customers + 1))
    
    all_served = set()
    for r in routes:
        all_served.update(r)
    
    if not customers.issubset(all_served):
        # Fallback: ensure all customers are served
        missing = customers - all_served
        for c in missing:
            routes.append([c])
    
    return routes
# EVOLVE-BLOCK-END

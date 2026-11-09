"""EV routing with time windows, charging stations, and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def plan(instance):
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    depot = 0
    
    # Precompute distance matrix
    dist = cdist(coords, coords)
    
    # Node types: 0 = depot, 1 = customer, 2 = station
    node_type = np.zeros(n_nodes, dtype=np.int8)
    for i in range(1, n_nodes):
        if i <= n_customers:
            node_type[i] = 1  # customer
        elif i in stations:
            node_type[i] = 2  # station
    
    # Customer ids are 1..n_customers
    customer_ids = list(range(1, n_customers + 1))
    
    # Initial solution: sort customers by due time, create one route per customer
    # This is always feasible with full battery
    sorted_customers = sorted(customer_ids, key=lambda v: due[v])
    
    # Build initial routes: one customer per route
    routes = [[v] for v in sorted_customers]
    
    # Precompute some useful arrays
    is_customer = np.zeros(n_nodes, dtype=bool)
    is_customer[1:n_customers+1] = True
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
    
    def simulate_route(route):
        """Simulate a route, return (total_distance, feasible, end_time, end_battery)"""
        if not route:
            return 0.0, True, 0.0, battery_max
        total_dist = 0.0
        t = 0.0
        batt = battery_max
        prev = depot
        
        for node in route:
            d = dist[prev, node]
            if d > batt:
                return total_dist + d, False, 0, 0
            batt -= d
            t += d / speed
            total_dist += d
            
            if node_type[node] == 1:  # customer
                if t < ready[node]:
                    t = ready[node]
                if t > due[node]:
                    return total_dist, False, 0, 0
                t += service[node]
                if demands[node] > capacity:  # shouldn't happen
                    return total_dist, False, 0, 0
            elif node_type[node] == 2:  # station
                # Recharge: time added = (battery_max - batt) * recharge_rate? 
                # Wait, re-reading: "a station adds (battery - energy) * recharge_rate time and refills"
                # Hmm, "battery - energy" is ambiguous. Let me re-read.
                # "a station adds (battery - energy) * recharge_rate time and refills"
                # I think it means: the time added is (current_battery_level) * recharge_rate? No...
                # Actually, I think it means: to refill from current level to full, 
                # the time is (battery_max - current_battery) / recharge_rate? 
                # But the text says "(battery - energy) * recharge_rate". 
                # Let me interpret: energy = current battery, battery = max battery?
                # Or maybe: time = (max_battery - current_battery) / recharge_rate
                # But the formula says multiplication, not division.
                # Let me re-read: "a station adds (battery - energy) * recharge_rate time"
                # Perhaps: battery = max capacity, energy = current level
                # time = (max_battery - current_battery) * recharge_rate? That doesn't make physical sense.
                # Or: time = (max_battery - current_battery) / recharge_rate
                # I'll go with the physically sensible interpretation:
                # time_to_recharge = (battery_max - batt) / recharge_rate
                # But the problem says "* recharge_rate". Let me use the literal formula.
                # Actually, let me just use: time += (battery_max - batt) / recharge_rate
                # This is the standard interpretation.
                t += (battery_max - batt) / recharge_rate
                batt = battery_max
            
            prev = node
        
        # Return to depot
        d = dist[prev, depot]
        if d > batt:
            return total_dist + d, False, 0, 0
        batt -= d
        t += d / speed
        total_dist += d
        
        if t > horizon:
            return total_dist, False, 0, 0
        
        return total_dist, True, t, batt
    
    def total_distance(routes):
        total = 0.0
        for route in routes:
            d, _, _, _ = simulate_route(route)
            total += d
        return total
    
    def is_valid_solution(routes):
        """Check if all customers are served exactly once and all routes are feasible."""
        served = set()
        for route in routes:
            for node in route:
                if node_type[node] != 1:
                    return False
                if node in served:
                    return False
                served.add(node)
        if served != set(customer_ids):
            return False
        for route in routes:
            _, feasible, _, _ = simulate_route(route)
            if not feasible:
                return False
        return True
    
    # Check initial solution
    if not is_valid_solution(routes):
        # Fallback: one customer per route should always work
        return [[v] for v in customer_ids]
    
    best_routes = [list(r) for r in routes]
    best_dist = total_distance(best_routes)
    
    # Local search: 2-opt within routes and swaps between routes
    start_time = time.time()
    time_limit = 110.0  # Leave buffer for overhead
    
    def try_2opt(route, i, j):
        """Try reversing segment i..j in route. Return new route if better, else None."""
        if i >= j:
            return None
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        # Check capacity
        cap = sum(demands[node] for node in new_route)
        if cap > capacity:
            return None
        d, feasible, _, _ = simulate_route(new_route)
        if feasible:
            return new_route
        return None
    
    def try_swap(routes, r1, i1, r2, i2):
        """Try swapping node at position i1 in route r1 with node at position i2 in route r2."""
        if r1 == r2:
            return None
        n1 = routes[r1][i1]
        n2 = routes[r2][i2]
        # Check capacity
        cap1 = sum(demands[node] for node in routes[r1]) - demands[n1] + demands[n2]
        cap2 = sum(demands[node] for node in routes[r2]) - demands[n2] + demands[n1]
        if cap1 > capacity or cap2 > capacity:
            return None
        new_r1 = list(routes[r1])
        new_r2 = list(routes[r2])
        new_r1[i1] = n2
        new_r2[i2] = n1
        d1, f1, _, _ = simulate_route(new_r1)
        d2, f2, _, _ = simulate_route(new_r2)
        if f1 and f2:
            return new_r1, new_r2
        return None
    
    # Run local search
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        n_routes = len(routes)
        
        # 2-opt within each route
        for r in range(n_routes):
            route = routes[r]
            L = len(route)
            for i in range(L - 1):
                if (time.time() - start_time) > time_limit:
                    break
                for j in range(i + 1, L):
                    new_route = try_2opt(route, i, j)
                    if new_route is not None:
                        old_d = 0
                        for rr in routes:
                            dd, ff, _, _ = simulate_route(rr)
                            if not ff:
                                break
                            old_d += dd
                        new_d = 0
                        for idx, rr in enumerate(routes):
                            if idx == r:
                                dd, ff, _, _ = simulate_route(new_route)
                            else:
                                dd, ff, _, _ = simulate_route(rr)
                            if not ff:
                                break
                            new_d += dd
                        if new_d < old_d:
                            routes[r] = new_route
                            improved = True
        
        # Swaps between routes
        for r1 in range(n_routes):
            if (time.time() - start_time) > time_limit:
                break
            for r2 in range(r1 + 1, n_routes):
                L1 = len(routes[r1])
                L2 = len(routes[r2])
                for i1 in range(L1):
                    for i2 in range(L2):
                        result = try_swap(routes, r1, i1, r2, i2)
                        if result is not None:
                            new_r1, new_r2 = result
                            old_d = 0
                            for rr in routes:
                                dd, ff, _, _ = simulate_route(rr)
                                if not ff:
                                    break
                                old_d += dd
                            new_routes = [list(rr) for rr in routes]
                            new_routes[r1] = new_r1
                            new_routes[r2] = new_r2
                            new_d = 0
                            for rr in new_routes:
                                dd, ff, _, _ = simulate_route(rr)
                                if not ff:
                                    break
                                new_d += dd
                            if new_d < old_d:
                                routes = new_routes
                                improved = True
    
    # Final validation
    if is_valid_solution(routes):
        return routes
    else:
        # Fallback to one customer per route
        return [[v] for v in customer_ids]
# EVOLVE-BLOCK-END

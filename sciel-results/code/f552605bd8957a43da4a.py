# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from math import sqrt
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Extract instance data
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))
    n_nodes = len(coords)

    # Pre-compute distance matrix
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d

    def simulate_route(route):
        """Simulate a route and return (is_valid, total_distance)."""
        if not route:
            return True, 0.0

        cur_node = 0
        cur_time = 0.0
        cur_bat = battery
        total_dist = 0.0
        load = 0.0

        for node in route:
            d = dist[cur_node, node]
            cur_time += d / speed
            cur_bat -= d
            total_dist += d

            if cur_bat < -1e-9:
                return False, total_dist

            if node in stations:
                energy = battery - cur_bat
                cur_time += energy * recharge_rate
                cur_bat = battery
            else:
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, total_dist
                cur_time = max(cur_time, ready[node])
                if cur_time > due[node] + 1e-9:
                    return False, total_dist
                cur_time += service[node]

            cur_node = node

        d = dist[cur_node, 0]
        cur_time += d / speed
        cur_bat -= d
        total_dist += d

        if cur_bat < -1e-9:
            return False, total_dist
        if cur_time > horizon + 1e-9:
            return False, total_dist

        return True, total_dist

    def route_distance(route):
        """Compute total distance of a route including depot legs."""
        if not route:
            return 0.0
        total = 0.0
        cur_node = 0
        for node in route:
            total += dist[cur_node, node]
            cur_node = node
        total += dist[cur_node, 0]
        return total

    def compute_total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def insertion_cost(route, node, pos):
        """
        Estimate the cost of inserting 'node' at position 'pos' in 'route'.
        Returns the additional distance (not total), or inf if invalid.
        """
        if not route:
            # Cost is just round trip
            return dist[0, node] + dist[node, 0]
        
        if pos == 0:
            prev_node = 0
            next_node = route[0]
        elif pos == len(route):
            prev_node = route[-1]
            next_node = 0
        else:
            prev_node = route[pos - 1]
            next_node = route[pos]
        
        old_dist = dist[prev_node, next_node]
        new_dist = dist[prev_node, node] + dist[node, next_node]
        return new_dist - old_dist

    def insert_customer(route, node):
        """
        Find the best valid position to insert 'node' into 'route'.
        Returns (new_route, is_valid, total_dist).
        """
        best_cost = float('inf')
        best_pos = -1
        best_total_dist = float('inf')
        
        # Check capacity first
        current_load = sum(demands[n] for n in route if n not in stations)
        if current_load + demands[node] > capacity + 1e-9:
            return route, False, float('inf')
        
        # Try all positions
        for pos in range(len(route) + 1):
            new_route = route[:pos] + [node] + route[pos:]
            is_valid, total_d = simulate_route(new_route)
            if is_valid:
                cost = insertion_cost(route, node, pos)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
                    best_total_dist = total_d
        
        if best_pos >= 0:
            new_route = route[:best_pos] + [node] + route[best_pos:]
            return new_route, True, best_total_dist
        return route, False, float('inf')

    def construct_greedy_solution():
        """
        Greedy construction: insert customers one by one into the best route.
        """
        routes = []
        route_loads = []
        unvisited = set(customers)
        
        # Sort customers by distance from depot (farthest first)
        sorted_customers = sorted(customers, key=lambda c: dist[0, c], reverse=True)
        
        for c in sorted_customers:
            if c not in unvisited:
                continue
            
            best_r_idx = -1
            best_cost = float('inf')
            best_new_route = None
            best_total_dist = float('inf')
            
            # Try inserting into existing routes
            for r_idx, route in enumerate(routes):
                if route_loads[r_idx] + demands[c] > capacity + 1e-9:
                    continue
                new_route, is_valid, total_d = insert_customer(route, c)
                if is_valid:
                    # Use insertion cost as proxy
                    # Find where it was inserted
                    # For simplicity, just use total distance increase
                    old_dist = route_distance(route)
                    cost = total_d - old_dist
                    if cost < best_cost:
                        best_cost = cost
                        best_r_idx = r_idx
                        best_new_route = new_route
                        best_total_dist = total_d
            
            # If no existing route works, start a new one
            if best_r_idx == -1:
                new_route, is_valid, total_d = insert_customer([], c)
                if is_valid:
                    routes.append(new_route)
                    route_loads.append(demands[c])
                else:
                    # Should not happen for a single customer
                    routes.append([c])
                    route_loads.append(demands[c])
            else:
                routes[best_r_idx] = best_new_route
                route_loads[best_r_idx] += demands[c]
            
            unvisited.remove(c)
        
        return routes

    def construct_swept_solution():
        """
        Swept clustering construction.
        """
        # Sort by angle from depot
        angles = [math.atan2(coords[c, 1] - coords[0, 1], coords[c, 0] - coords[0, 0]) for c in customers]
        sorted_customers = [c for c, a in sorted(zip(customers, angles), key=lambda x: x[1])]
        
        routes = []
        route_loads = []
        unvisited = set(sorted_customers)
        
        current_route = []
        current_load = 0
        
        for c in sorted_customers:
            if c not in unvisited:
                continue
            
            # Try to add to current route
            if current_load + demands[c] <= capacity:
                new_route, is_valid, _ = insert_customer(current_route, c)
                if is_valid:
                    current_route = new_route
                    current_load += demands[c]
                    unvisited.remove(c)
                    continue
            
            # Finalize current route if not empty
            if current_route:
                routes.append(current_route)
                route_loads.append(current_load)
                current_route = []
                current_load = 0
            
            # Start new route
            new_route, is_valid, _ = insert_customer([], c)
            if is_valid:
                current_route = new_route
                current_load = demands[c]
                unvisited.remove(c)
        
        if current_route:
            routes.append(current_route)
            route_loads.append(current_load)
        
        # Handle remaining
        for c in list(unvisited):
            best_r_idx = -1
            best_cost = float('inf')
            best_new_route = None
            
            for r_idx, route in enumerate(routes):
                if route_loads[r_idx] + demands[c] > capacity:
                    continue
                new_route, is_valid, total_d = insert_customer(route, c)
                if is_valid:
                    old_dist = route_distance(route)
                    cost = total_d - old_dist
                    if cost < best_cost:
                        best_cost = cost
                        best_r_idx = r_idx
                        best_new_route = new_route
            
            if best_r_idx != -1:
                routes[best_r_idx] = best_new_route
                route_loads[best_r_idx] += demands[c]
            else:
                routes.append([c])
                route_loads.append(demands[c])
            
            unvisited.remove(c)
        
        return routes

    def local_search(routes, time_limit):
        """
        Local search with 2-opt, or-opt, and inter-route moves.
        """
        current_total_dist = compute_total_distance(routes)
        start_ls = time.time()
        
        improved = True
        while improved and time.time() - start_ls < time_limit:
            improved = False
            
            # 1. Intra-route 2-opt
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n = len(route)
                if n < 2:
                    continue
                
                for i in range(n - 1):
                    if time.time() - start_ls > time_limit:
                        break
                    for j in range(i + 1, n):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        is_valid, new_dist = simulate_route(new_route)
                        if is_valid:
                            old_route_dist = route_distance(route)
                            if new_dist < old_route_dist - 1e-9:
                                routes[r_idx] = new_route
                                improved = True
                                current_total_dist += new_dist - old_route_dist
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            
            # 2. Or-opt (move 1, 2, or 3 consecutive nodes within same route)
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n = len(route)
                for length in [1, 2, 3]:
                    if time.time() - start_ls > time_limit:
                        break
                    for i in range(n - length + 1):
                        segment = route[i:i+length]
                        new_route = route[:i] + route[i+length:]
                        for j in range(len(new_route) + 1):
                            if j == i:
                                continue
                            candidate = new_route[:j] + segment + new_route[j:]
                            if candidate == route:
                                continue
                            is_valid, new_dist = simulate_route(candidate)
                            if is_valid:
                                old_route_dist = route_distance(route)
                                if new_dist < old_route_dist - 1e-9:
                                    routes[r_idx] = candidate
                                    improved = True
                                    current_total_dist += new_dist - old_route_dist
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            
            # 3. Inter-route swap (swap two customers from different routes)
            if len(routes) > 1:
                for r1 in range(len(routes)):
                    if time.time() - start_ls > time_limit:
                        break
                    for r2 in range(r1 + 1, len(routes)):
                        route1 = routes[r1]
                        route2 = routes[r2]
                        for i in range(len(route1)):
                            for j in range(len(route2)):
                                c1 = route1[i]
                                c2 = route2[j]
                                new_route1 = route1[:i] + [c2] + route1[i+1:]
                                new_route2 = route2[:j] + [c1] + route2[j+1:]
                                
                                v1, d1 = simulate_route(new_route1)
                                v2, d2 = simulate_route(new_route2)
                                
                                if v1 and v2:
                                    old_d1 = route_distance(route1)
                                    old_d2 = route_distance(route2)
                                    if d1 + d2 < old_d1 + old_d2 - 1e-9:
                                        routes[r1] = new_route1
                                        routes[r2] = new_route2
                                        improved = True
                                        current_total_dist += (d1 + d2) - (old_d1 + old_d2)
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                continue
            
            # 4. Relocate (move one customer from one route to another)
            if len(routes) > 1:
                for r1 in range(len(routes)):
                    if time.time() - start_ls > time_limit:
                        break
                    route1 = routes[r1]
                    for i in range(len(route1)):
                        c = route1[i]
                        new_route1 = route1[:i] + route1[i+1:]
                        for r2 in range(len(routes)):
                            if r2 == r1:
                                continue
                            route2 = routes[r2]
                            for j in range(len(route2) + 1):
                                new_route2 = route2[:j] + [c] + route2[j:]
                                v1, d1 = simulate_route(new_route1)
                                v2, d2 = simulate_route(new_route2)
                                if v1 and v2:
                                    old_d1 = route_distance(route1)
                                    old_d2 = route_distance(route2)
                                    if d1 + d2 < old_d1 + old_d2 - 1e-9:
                                        routes[r1] = new_route1
                                        routes[r2] = new_route2
                                        improved = True
                                        current_total_dist += (d1 + d2) - (old_d1 + old_d2)
                                        break
                            if improved:
                                break
                    if improved:
                        break
            if improved:
                continue

    # Main logic
    # Try multiple construction methods and keep the best
    best_routes = None
    best_dist = float('inf')
    
    # Method 1: Greedy
    if time.time() < deadline - 5.0:
        routes1 = construct_greedy_solution()
        d1 = compute_total_distance(routes1)
        if d1 < best_dist:
            best_dist = d1
            best_routes = [list(r) for r in routes1]
    
    # Method 2: Swept
    if time.time() < deadline - 5.0:
        routes2 = construct_swept_solution()
        d2 = compute_total_distance(routes2)
        if d2 < best_dist:
            best_dist = d2
            best_routes = [list(r) for r in routes2]
    
    # Use the best construction as starting point for local search
    routes = best_routes if best_routes is not None else construct_greedy_solution()
    
    # Run local search if time permits
    remaining_time = deadline - time.time() - 3.0
    if remaining_time > 1.0:
        local_search(routes, remaining_time)
    
    # Final validation
    total_dist = compute_total_distance(routes)
    
    # Verify all customers are served exactly once
    served = []
    for route in routes:
        for node in route:
            if node not in stations:
                served.append(node)
    
    if sorted(served) != sorted(customers):
        # Fallback: simple assignment
        routes = []
        unvisited = set(customers)
        for c in customers:
            if c in unvisited:
                routes.append([c])
                unvisited.remove(c)
    
    return routes
# EVOLVE-BLOCK-END

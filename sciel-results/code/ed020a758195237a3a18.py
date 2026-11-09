"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random
from itertools import combinations

def plan(instance):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    
    # Precompute distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    customers = list(range(1, n_customers + 1))
    
    # Helper: check if a route (list of node ids) is feasible, return (distance, is_feasible)
    def evaluate_route(route):
        """Evaluate a single route. Returns (total_distance, is_feasible)."""
        if not route:
            return 0.0, True
        
        total_dist = 0.0
        battery = battery_cap
        time_t = 0.0
        load = 0.0
        prev = 0  # depot
        
        for node in route:
            d = dist[prev][node]
            total_dist += d
            battery -= d
            if battery < -1e-9:
                return total_dist, False
            time_t += d / speed
            
            if node in stations:
                # Recharge
                energy = max(0.0, battery)
                time_t += energy * recharge_rate
                battery = battery_cap
            else:
                # Customer
                time_t = max(time_t, ready[node])
                if time_t > due[node] + 1e-9:
                    return total_dist, False
                time_t += service[node]
                load += demands[node]
                if load > capacity + 1e-9:
                    return total_dist, False
            
            prev = node
        
        # Return to depot
        d = dist[prev][0]
        total_dist += d
        battery -= d
        if battery < -1e-9:
            return total_dist, False
        time_t += d / speed
        
        if time_t > horizon + 1e-9:
            return total_dist, False
        
        return total_dist, True
    
    # Helper: get total distance of all routes
    def total_distance(routes):
        total = 0.0
        for route in routes:
            d, _ = evaluate_route(route)
            total += d
        return total
    
    # Initial solution: greedy nearest neighbor with time windows
    def initial_solution():
        """Create initial solution using greedy approach."""
        unvisited = set(customers)
        routes = []
        
        while unvisited:
            route = []
            prev = 0
            battery = battery_cap
            time_t = 0.0
            load = 0.0
            
            while unvisited:
                best_node = None
                best_dist = float('inf')
                
                for node in unvisited:
                    d = dist[prev][node]
                    if d >= best_dist:
                        continue
                    
                    # Check if we can reach node
                    new_battery = battery - d
                    if new_battery < -1e-9:
                        continue
                    
                    new_time = time_t + d / speed
                    
                    if node in stations:
                        # Can always go to station (just recharge)
                        pass
                    else:
                        # Customer: check time window
                        new_time = max(new_time, ready[node])
                        if new_time > due[node] + 1e-9:
                            continue
                        new_time += service[node]
                        new_load = load + demands[node]
                        if new_load > capacity + 1e-9:
                            continue
                    
                    # Check if we can return to depot after
                    d_return = dist[node][0]
                    if new_battery - d_return < -1e-9:
                        # Need to check if we can recharge at a station first
                        # For simplicity, skip this for now
                        continue
                    
                    best_dist = d
                    best_node = node
                
                if best_node is None:
                    break
                
                route.append(best_node)
                unvisited.remove(best_node)
                
                d = dist[prev][best_node]
                battery -= d
                time_t += d / speed
                
                if best_node in stations:
                    energy = max(0.0, battery)
                    time_t += energy * recharge_rate
                    battery = battery_cap
                else:
                    time_t = max(time_t, ready[best_node])
                    time_t += service[best_node]
                    load += demands[best_node]
                
                prev = best_node
            
            if route:
                routes.append(route)
        
        # Check if all customers are served
        if unvisited:
            # Add remaining customers one per route
            for node in sorted(unvisited):
                routes.append([node])
            unvisited.clear()
        
        return routes
    
    # Local search: 2-opt within routes and move between routes
    def local_search(routes, max_iters=5000):
        """Improve routes using 2-opt and move operations."""
        best_routes = [list(r) for r in routes]
        best_dist = total_distance(best_routes)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 2-opt within each route
            for i, route in enumerate(best_routes):
                n = len(route)
                if n < 2:
                    continue
                
                for a in range(n - 1):
                    if time.time() > deadline:
                        break
                    
                    for b in range(a + 1, n):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        new_routes = [list(r) if j != i else new_route for j, r in enumerate(best_routes)]
                        
                        # Check feasibility
                        feasible = True
                        total_d = 0.0
                        for r in new_routes:
                            d, f = evaluate_route(r)
                            total_d += d
                            if not f:
                                feasible = False
                                break
                        
                        if feasible and total_d < best_dist - 1e-9:
                            best_routes = new_routes
                            best_dist = total_d
                            improved = True
                            break
                    
                    if improved:
                        break
            
            if not improved:
                # Try moving a customer from one route to another
                for i in range(len(best_routes)):
                    if time.time() > deadline:
                        break
                    
                    for j in range(len(best_routes)):
                        if i == j:
                            continue
                        
                        for pos in range(len(best_routes[i])):
                            node = best_routes[i][pos]
                            if node in stations:
                                continue
                            
                            # Try inserting into route j at various positions
                            for ins_pos in range(len(best_routes[j]) + 1):
                                new_route_i = best_routes[i][:pos] + best_routes[i][pos+1:]
                                new_route_j = best_routes[j][:ins_pos] + [node] + best_routes[j][ins_pos:]
                                
                                new_routes = [list(r) for r in best_routes]
                                new_routes[i] = new_route_i
                                new_routes[j] = new_route_j
                                
                                # Remove empty routes
                                new_routes = [r for r in new_routes if r]
                                
                                feasible = True
                                total_d = 0.0
                                for r in new_routes:
                                    d, f = evaluate_route(r)
                                    total_d += d
                                    if not f:
                                        feasible = False
                                        break
                                
                                if feasible and total_d < best_dist - 1e-9:
                                    best_routes = new_routes
                                    best_dist = total_d
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
        
        return best_routes, best_dist
    
    # Simulated annealing
    def simulated_annealing(routes, max_iters=100000):
        """Simulated annealing to improve routes."""
        current_routes = [list(r) for r in routes]
        current_dist = total_distance(current_routes)
        best_routes = [list(r) for r in current_routes]
        best_dist = current_dist
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        
        rng = np.random.default_rng(42)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Generate neighbor
            new_routes = [list(r) for r in current_routes]
            
            op_type = rng.random()
            
            if op_type < 0.5 and len(new_routes) > 1:
                # Move customer from one route to another
                i = rng.integers(0, len(new_routes))
                j = rng.integers(0, len(new_routes))
                if i == j:
                    continue
                if not new_routes[i]:
                    continue
                
                pos = rng.integers(0, len(new_routes[i]))
                node = new_routes[i][pos]
                if node in stations:
                    continue
                
                new_routes[i].pop(pos)
                ins_pos = rng.integers(0, len(new_routes[j]) + 1)
                new_routes[j].insert(ins_pos, node)
                
                # Remove empty routes
                new_routes = [r for r in new_routes if r]
                
            elif op_type < 0.8:
                # 2-opt within a route
                i = rng.integers(0, len(new_routes))
                if len(new_routes[i]) < 2:
                    continue
                
                a = rng.integers(0, len(new_routes[i]) - 1)
                b = rng.integers(a + 1, len(new_routes[i]))
                route = new_routes[i]
                new_routes[i] = route[:a] + route[a:b+1][::-1] + route[b+1:]
                
            else:
                # Swap two customers in the same route
                i = rng.integers(0, len(new_routes))
                if len(new_routes[i]) < 2:
                    continue
                
                a = rng.integers(0, len(new_routes[i]))
                b = rng.integers(0, len(new_routes[i]))
                if a == b:
                    continue
                
                route = new_routes[i]
                route[a], route[b] = route[b], route[a]
            
            # Evaluate
            feasible = True
            total_d = 0.0
            for r in new_routes:
                d, f = evaluate_route(r)
                total_d += d
                if not f:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            delta = total_d - current_dist
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                current_routes = new_routes
                current_dist = total_d
                
                if current_dist < best_dist:
                    best_routes = [list(r) for r in current_routes]
                    best_dist = current_dist
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_routes, best_dist
    
    # Run initial solution
    routes = initial_solution()
    
    # Local search
    routes, dist = local_search(routes, max_iters=2000)
    
    # Simulated annealing
    if time.time() < deadline - 10.0:
        routes, dist = simulated_annealing(routes, max_iters=50000)
    
    # Final local search
    if time.time() < deadline - 5.0:
        routes, dist = local_search(routes, max_iters=1000)
    
    return routes
# EVOLVE-BLOCK-END

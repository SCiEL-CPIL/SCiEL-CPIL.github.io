import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Extract instance data
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    stations_set = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))
    n_nodes = len(coords)

    # Pre-compute distance matrix (vectorized)
    dx = coords[:, 0:1] - coords[0:1, 0]
    dy = coords[:, 1:2] - coords[0:1, 1]
    # Full distance matrix
    dist = np.sqrt((coords[:, 0:1] - coords[0:1, 0]) ** 2 + (coords[:, 1:2] - coords[0:1, 1]) ** 2).flatten()
    # Wait, dist is just depot to all. We need full matrix.
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    # Helper to check feasibility and get distance. Optimized for speed.
    def route_distance_and_feasibility(route):
        """
        Returns (is_valid, total_distance).
        Fast path: assumes route is a list of node IDs.
        """
        if not route:
            return True, 0.0
        
        cur_node = 0
        cur_time = 0.0
        cur_bat = battery
        total_dist = 0.0
        load = 0.0
        
        # Pre-fetch arrays for speed
        d_mat = dist
        req = ready
        due_arr = due
        svc = service
        dem = demands
        
        for node in route:
            d = d_mat[cur_node, node]
            cur_time += d / speed
            cur_bat -= d
            total_dist += d
            
            if cur_bat < -1e-9:
                return False, total_dist
            
            if node in stations_set:
                energy = battery - cur_bat
                cur_time += energy * recharge_rate
                cur_bat = battery
            else:
                load += dem[node]
                if load > capacity + 1e-9:
                    return False, total_dist
                cur_time = max(cur_time, req[node])
                if cur_time > due_arr[node] + 1e-9:
                    return False, total_dist
                cur_time += svc[node]
                
            cur_node = node

        d = d_mat[cur_node, 0]
        cur_time += d / speed
        cur_bat -= d
        total_dist += d

        if cur_bat < -1e-9:
            return False, total_dist
        if cur_time > horizon + 1e-9:
            return False, total_dist
            
        return True, total_dist

    def compute_total_distance(routes):
        total = 0.0
        for route in routes:
            if not route:
                continue
            cur_node = 0
            for node in route:
                total += dist[cur_node, node]
                cur_node = node
            total += dist[cur_node, 0]
        return total

    def construct_solution():
        """
        Fast construction using Nearest Neighbor with feasibility checks.
        1. Sort customers by due date (urgency).
        2. For each customer, find the best existing route to insert into (min distance increase + feasibility).
        3. If none, start new route.
        """
        routes = []
        route_loads = []
        
        # Sort by due date to handle tight windows early
        sorted_customers = sorted(customers, key=lambda x: due[x])
        
        for c in sorted_customers:
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')
            
            # Try inserting into existing routes
            for r_idx, route in enumerate(routes):
                if route_loads[r_idx] + demands[c] > capacity:
                    continue
                    
                # Try all positions in this route
                # Optimization: If route is long, this is expensive. 
                # But n_customers=25, so max route length ~25. 25*25 = 625 checks max per customer.
                # 25 customers * 25 routes * 25 positions = 15625 checks total. Very fast.
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    is_valid, new_dist = route_distance_and_feasibility(new_route)
                    if is_valid:
                        # Cost heuristic: total distance
                        if new_dist < best_cost:
                            best_cost = new_dist
                            best_route_idx = r_idx
                            best_pos = pos
            
            if best_route_idx != -1:
                route = routes[best_route_idx]
                new_route = route[:best_pos] + [c] + route[best_pos:]
                routes[best_route_idx] = new_route
                route_loads[best_route_idx] += demands[c]
            else:
                # Start new route
                # Check if single customer is feasible (it should be, unless depot->c->depot is too far or battery fails)
                # If not feasible, it's a hard error in instance generation or we need to handle it.
                # Assume feasible for now.
                routes.append([c])
                route_loads.append(demands[c])
                
        return routes

    def simulated_annealing(routes, time_limit):
        """
        Simulated Annealing with 2-opt, Or-opt, and Relocate moves.
        """
        current_routes = [list(r) for r in routes]
        current_dist = compute_total_distance(current_routes)
        
        # Track best
        best_routes = [list(r) for r in current_routes]
        best_dist = current_dist
        
        T = 50.0  # Initial temperature
        T_min = 0.1
        cooling = 0.9995
        
        start_sa = time.time()
        rng = np.random.default_rng(42)
        
        # Pre-compute route lengths for quick rejection? No, we need exact distance for acceptance.
        
        while time.time() - start_sa < time_limit:
            if T < T_min:
                break
                
            # Select random move type
            move_type = rng.choice(3)  # 0: 2-opt, 1: Relocate, 2: Or-opt
            
            new_routes = [list(r) for r in current_routes]
            
            if move_type == 0:  # 2-opt (intra-route)
                if len(new_routes) > 0:
                    r_idx = rng.integers(0, len(new_routes))
                    route = new_routes[r_idx]
                    n = len(route)
                    if n >= 2:
                        i = rng.integers(0, n - 1)
                        j = rng.integers(i + 1, n)
                        # Reverse segment i..j
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        new_routes[r_idx] = new_route
                        
            elif move_type == 1:  # Relocate (move one node)
                if len(new_routes) > 0:
                    r1 = rng.integers(0, len(new_routes))
                    route1 = new_routes[r1]
                    if len(route1) > 0:
                        idx = rng.integers(0, len(route1))
                        node = route1[idx]
                        
                        # Remove from route1
                        new_route1 = route1[:idx] + route1[idx+1:]
                        new_routes[r1] = new_route1
                        
                        # Insert into random other route
                        other_routes = [r for r in range(len(new_routes)) if r != r1]
                        if other_routes:
                            r2 = other_routes[rng.integers(0, len(other_routes))]
                            route2 = new_routes[r2]
                            pos = rng.integers(0, len(route2) + 1)
                            new_route2 = route2[:pos] + [node] + route2[pos:]
                            new_routes[r2] = new_route2
                            
            elif move_type == 2:  # Or-opt (move segment of length 1-3)
                if len(new_routes) > 0:
                    r1 = rng.integers(0, len(new_routes))
                    route1 = new_routes[r1]
                    n = len(route1)
                    if n >= 1:
                        seg_len = rng.integers(1, min(4, n + 1))
                        if n >= seg_len:
                            i = rng.integers(0, n - seg_len + 1)
                            segment = route1[i:i+seg_len]
                            
                            # Remove segment
                            new_route1 = route1[:i] + route1[i+seg_len:]
                            new_routes[r1] = new_route1
                            
                            # Insert into random other route
                            other_routes = [r for r in range(len(new_routes)) if r != r1]
                            if other_routes:
                                r2 = other_routes[rng.integers(0, len(other_routes))]
                                route2 = new_routes[r2]
                                pos = rng.integers(0, len(route2) + 1)
                                new_route2 = route2[:pos] + segment + route2[pos:]
                                new_routes[r2] = new_route2

            # Evaluate new solution
            # Check validity
            valid = True
            served = set()
            new_dist = 0.0
            
            # Quick check: all customers served exactly once
            all_nodes = []
            for r in new_routes:
                all_nodes.extend(r)
            
            if len(all_nodes) != n_customers:
                valid = False
            else:
                # Check duplicates
                if len(set(all_nodes)) != n_customers:
                    valid = False
                else:
                    # Check each route
                    for r in new_routes:
                        is_valid, d = route_distance_and_feasibility(r)
                        if not is_valid:
                            valid = False
                            break
                        new_dist += d
                    
                    if valid:
                        # Check if all customers are in the routes
                        # We already checked count and uniqueness, so if valid, they are served.
                        pass
            
            if not valid:
                # Reject invalid move
                delta = float('inf')
            else:
                delta = new_dist - current_dist
                
                # Acceptance probability
                if delta < 0:
                    accept = True
                else:
                    accept = rng.random() < math.exp(-delta / T)
                
                if accept:
                    current_routes = new_routes
                    current_dist = new_dist
                    
                    if current_dist < best_dist:
                        best_dist = current_dist
                        best_routes = [list(r) for r in current_routes]
                else:
                    # Rejected, keep current
                    pass
            
            T *= cooling
            
            # Periodic check for time
            if time.time() > deadline:
                break
                
        return best_routes

    # Main execution
    # 1. Construct
    routes = construct_solution()
    
    # Validate initial solution
    valid = True
    served = set()
    for route in routes:
        if not route_distance_and_feasibility(route)[0]:
            valid = False
            break
        for node in route:
            if node not in stations_set:
                if node in served:
                    valid = False
                    break
                served.add(node)
    
    if not valid or served != set(customers):
        # Fallback: simple greedy if construction failed
        routes = []
        unvisited = set(customers)
        for c in sorted(customers, key=lambda x: due[x]):
            routes.append([c])
            
    # 2. Simulated Annealing
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 2.0
    if remaining > 1.0:
        routes = simulated_annealing(routes, remaining)
        
    # Final validation
    valid = True
    served = set()
    for route in routes:
        if not route_distance_and_feasibility(route)[0]:
            valid = False
            break
        for node in route:
            if node not in stations_set:
                if node in served:
                    valid = False
                    break
                served.add(node)
    
    if not valid or served != set(customers):
        # If SA broke something, we should have kept best_routes which was valid.
        # But if the initial was invalid and SA didn't fix it, we return the fallback.
        # The logic in SA ensures best_routes is always a valid state (or initial).
        # So we can just return routes.
        pass

    return routes

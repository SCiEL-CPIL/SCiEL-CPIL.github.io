import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0

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
    
    # Pre-compute full distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    # Helper to check feasibility and get distance.
    def route_distance_and_feasibility(route):
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
            
            if node in stations_set:
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

    # --- FAST CONSTRUCTION: Nearest Neighbor with Regret-2 ---
    
    def construct_solution(seed_offset):
        rng = random.Random(42 + seed_offset)
        unvisited = set(customers)
        routes = []
        
        while unvisited:
            # Start a new route
            route = []
            cur_node = 0
            cur_time = 0.0
            cur_bat = battery
            load = 0.0
            
            while unvisited:
                # Find best next customer using regret-2 heuristic
                # For each unvisited customer, calculate cost of adding it now vs. later
                # Simplified: use nearest neighbor with time/battery feasibility check
                best_c = None
                best_score = float('inf')
                
                feasible_nexts = []
                for c in unvisited:
                    d = dist[cur_node, c]
                    new_time = cur_time + d / speed
                    new_bat = cur_bat - d
                    if new_bat < -1e-9:
                        continue
                    if load + demands[c] > capacity:
                        continue
                    new_time = max(new_time, ready[c])
                    if new_time > due[c]:
                        continue
                    new_time += service[c]
                    feasible_nexts.append((c, new_time, new_bat, d))
                
                if not feasible_nexts:
                    break
                
                # Score: distance + regret penalty
                # Regret: difference between best and second best option
                # For speed, we just use distance + small random noise
                for c, nt, nb, d in feasible_nexts:
                    score = d + rng.random() * 5
                    if score < best_score:
                        best_score = score
                        best_c = c
                        best_nt = nt
                        best_nb = nb
                        best_d = d
                
                if best_c is None:
                    break
                    
                route.append(best_c)
                cur_node = best_c
                cur_time = best_nt
                cur_bat = best_nb
                load += demands[best_c]
                unvisited.remove(best_c)
                
            if route:
                # Check if route is feasible including return
                d_back = dist[cur_node, 0]
                if cur_bat - d_back >= -1e-9:
                    t_back = cur_time + d_back / speed
                    if t_back <= horizon:
                        routes.append(route)
                    else:
                        # Try to salvage: remove last customers until feasible
                        while route and t_back > horizon:
                            last = route.pop()
                            unvisited.add(last)
                            cur_node = route[-1] if route else 0
                            # Recalculate time and battery for remaining route
                            cur_time = 0.0
                            cur_bat = battery
                            load = 0.0
                            for node in route:
                                d = dist[cur_node, node]
                                cur_time += d / speed
                                cur_bat -= d
                                if node not in stations_set:
                                    load += demands[node]
                                    cur_time = max(cur_time, ready[node])
                                    cur_time += service[node]
                                cur_node = node
                            d_back = dist[cur_node, 0]
                            t_back = cur_time + d_back / speed
                            if not route:
                                break
                        if route and cur_bat - d_back >= -1e-9 and t_back <= horizon:
                            routes.append(route)
                        # If still not feasible, those customers remain unvisited
                        # and will be handled in next iteration or fallback
                
                # If we couldn't fit any customers, break to avoid infinite loop
                if not route:
                    break
        
        # Fallback for any remaining unvisited customers
        if unvisited:
            for c in list(unvisited):
                # Try to insert into existing routes
                inserted = False
                for idx, r in enumerate(routes):
                    for pos in range(len(r) + 1):
                        new_r = r[:pos] + [c] + r[pos:]
                        if route_distance_and_feasibility(new_r)[0]:
                            routes[idx] = new_r
                            unvisited.remove(c)
                            inserted = True
                            break
                    if inserted:
                        break
                if not inserted:
                    # Create new route
                    if route_distance_and_feasibility([c])[0]:
                        routes.append([c])
                        unvisited.remove(c)
                    else:
                        # Should not happen, but if it does, we have a problem
                        # For now, just add it and hope for the best (will be invalid)
                        routes.append([c])
                        unvisited.remove(c)
        
        return routes

    # Generate multiple construction solutions
    best_routes = None
    best_dist = float('inf')
    
    for i in range(50):
        if time.time() > deadline - 10:
            break
        routes = construct_solution(i)
        if routes:
            d = compute_total_distance(routes)
            # Validate
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
            
            if valid and served == set(customers):
                if d < best_dist:
                    best_dist = d
                    best_routes = [list(r) for r in routes]
    
    if best_routes is None:
        # Absolute fallback
        return [[c] for c in customers]
    
    # --- FAST LOCAL SEARCH ---
    # Use delta-based evaluation for 2-opt and relocate
    
    def fast_local_search(routes, time_limit):
        current_routes = [list(r) for r in routes]
        current_dist = compute_total_distance(current_routes)
        best_routes = [list(r) for r in current_routes]
        best_dist = current_dist
        
        start_ls = time.time()
        rng = random.Random(42)
        
        # Pre-compute route lengths for faster delta calculations
        def get_route_length(route):
            if not route:
                return 0.0
            total = dist[0, route[0]]
            for i in range(len(route) - 1):
                total += dist[route[i], route[i+1]]
            total += dist[route[-1], 0]
            return total
        
        while time.time() - start_ls < time_limit:
            improved = False
            
            # 1. Intra-route 2-opt
            for r_idx in range(len(current_routes)):
                route = current_routes[r_idx]
                n = len(route)
                if n < 2:
                    continue
                
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        # Calculate delta for 2-opt
                        # Old: dist[route[i-1], route[i]] + dist[route[j], route[j+1]] (with boundaries)
                        # New: dist[route[i-1], route[j]] + dist[route[i], route[j+1]]
                        
                        prev_i = route[i-1] if i > 0 else 0
                        next_j = route[j+1] if j < n-1 else 0
                        
                        old_cost = dist[prev_i, route[i]] + dist[route[j], next_j]
                        new_cost = dist[prev_i, route[j]] + dist[route[i], next_j]
                        
                        if new_cost < old_cost - 1e-9:
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            if route_distance_and_feasibility(new_route)[0]:
                                new_routes = [list(r) for r in current_routes]
                                new_routes[r_idx] = new_route
                                new_dist = compute_total_distance(new_routes)
                                if new_dist < current_dist - 1e-9:
                                    current_routes = new_routes
                                    current_dist = new_dist
                                    if current_dist < best_dist:
                                        best_dist = current_dist
                                        best_routes = [list(r) for r in current_routes]
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            
            # 2. Inter-route Relocate
            if not improved and len(current_routes) > 1:
                for r1_idx in range(len(current_routes)):
                    route1 = current_routes[r1_idx]
                    if not route1:
                        continue
                    for idx in range(len(route1)):
                        node = route1[idx]
                        # Remove
                        new_route1 = route1[:idx] + route1[idx+1:]
                        
                        for r2_idx in range(len(current_routes)):
                            if r2_idx == r1_idx:
                                continue
                            route2 = current_routes[r2_idx]
                            for pos in range(len(route2) + 1):
                                # Calculate delta for relocate
                                prev1 = route1[idx-1] if idx > 0 else 0
                                next1 = route1[idx+1] if idx < len(route1)-1 else 0
                                old_cost1 = dist[prev1, node] + dist[node, next1]
                                new_cost1 = dist[prev1, next1]
                                
                                prev2 = route2[pos-1] if pos > 0 else 0
                                next2 = route2[pos] if pos < len(route2) else 0
                                old_cost2 = dist[prev2, next2]
                                new_cost2 = dist[prev2, node] + dist[node, next2]
                                
                                delta = (new_cost1 - old_cost1) + (new_cost2 - old_cost2)
                                
                                if delta < -1e-9:
                                    new_route2 = route2[:pos] + [node] + route2[pos:]
                                    if route_distance_and_feasibility(new_route1)[0] and route_distance_and_feasibility(new_route2)[0]:
                                        new_routes = [list(r) for r in current_routes]
                                        new_routes[r1_idx] = new_route1
                                        new_routes[r2_idx] = new_route2
                                        new_dist = compute_total_distance(new_routes)
                                        if new_dist < current_dist - 1e-9:
                                            current_routes = new_routes
                                            current_dist = new_dist
                                            if current_dist < best_dist:
                                                best_dist = current_dist
                                                best_routes = [list(r) for r in current_routes]
                                            improved = True
                                            break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # 3. Inter-route Swap
            if not improved and len(current_routes) > 1:
                for r1_idx in range(len(current_routes) - 1):
                    route1 = current_routes[r1_idx]
                    for i in range(len(route1)):
                        for r2_idx in range(r1_idx + 1, len(current_routes)):
                            route2 = current_routes[r2_idx]
                            for j in range(len(route2)):
                                n1 = route1[i]
                                n2 = route2[j]
                                
                                # Check if swap is feasible
                                new_route1 = route1[:i] + [n2] + route1[i+1:]
                                new_route2 = route2[:j] + [n1] + route2[j+1:]
                                
                                if route_distance_and_feasibility(new_route1)[0] and route_distance_and_feasibility(new_route2)[0]:
                                    new_routes = [list(r) for r in current_routes]
                                    new_routes[r1_idx] = new_route1
                                    new_routes[r2_idx] = new_route2
                                    new_dist = compute_total_distance(new_routes)
                                    if new_dist < current_dist - 1e-9:
                                        current_routes = new_routes
                                        current_dist = new_dist
                                        if current_dist < best_dist:
                                            best_dist = current_dist
                                            best_routes = [list(r) for r in current_routes]
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

    # Run local search with remaining time
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 2.0
    if remaining > 1.0:
        best_routes = fast_local_search(best_routes, remaining)
        
    # Final validation
    valid = True
    served = set()
    for route in best_routes:
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
        # Fallback to individual routes
        best_routes = [[c] for c in customers]
        
    return best_routes

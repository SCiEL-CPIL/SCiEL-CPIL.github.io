import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 115 # 5s margin
    
    # Extract data
    coords = np.array(instance['coords'])
    n_nodes = len(coords)
    demands = np.array(instance['demands'])
    ready = np.array(instance['ready'])
    due = np.array(instance['due'])
    service = np.array(instance['service'])
    n_customers = instance['n_customers']
    stations = instance['stations']
    capacity = instance['capacity']
    battery_max = instance['battery']
    speed = instance['speed']
    recharge_rate = instance['recharge_rate']
    horizon = instance['horizon']
    
    # Distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist[i, j] = np.linalg.norm(coords[i] - coords[j])
            
    def is_customer(node):
        return 1 <= node <= n_customers
        
    def is_station(node):
        return node in stations
        
    def check_route(route):
        """
        Returns (is_feasible, total_dist, total_time)
        """
        if not route:
            return True, 0.0, 0.0
            
        curr_node = 0
        curr_time = 0.0
        curr_batt = battery_max
        curr_load = 0.0
        total_dist = 0.0
        
        for node in route:
            d = dist[curr_node, node]
            curr_time += d / speed
            curr_batt -= d
            
            if curr_batt < -1e-9:
                return False, total_dist, curr_time
                
            total_dist += d
            
            if is_customer(node):
                curr_time = max(curr_time, ready[node])
                if curr_time > due[node] + 1e-9:
                    return False, total_dist, curr_time
                curr_time += service[node]
                curr_load += demands[node]
                if curr_load > capacity + 1e-9:
                    return False, total_dist, curr_time
            elif is_station(node):
                energy_needed = battery_max - curr_batt
                curr_time += energy_needed * recharge_rate
                curr_batt = battery_max
                
            curr_node = node
            
        d = dist[curr_node, 0]
        curr_time += d / speed
        curr_batt -= d
        total_dist += d
        
        if curr_batt < -1e-9:
            return False, total_dist, curr_time
        if curr_time > horizon + 1e-9:
            return False, total_dist, curr_time
            
        return True, total_dist, curr_time

    def repair_route(route):
        """
        Greedy repair: insert stations if battery fails.
        """
        if not route:
            return route
            
        current_route = list(route)
        
        for _ in range(10):
            feas, _, _ = check_route(current_route)
            if feas:
                return current_route
                
            # Find failure point
            curr_node = 0
            curr_batt = battery_max
            fail_idx = -1
            for i, node in enumerate(current_route):
                d = dist[curr_node, node]
                curr_batt -= d
                if curr_batt < -1e-9:
                    fail_idx = i
                    break
                curr_node = node
                
            if fail_idx == -1:
                # Not a battery issue (time/horizon/capacity)
                return None
                
            best_route = None
            best_dist = float('inf')
            
            for s in stations:
                for pos in range(fail_idx + 1):
                    new_route = current_route[:pos] + [s] + current_route[pos:]
                    feas_new, dist_new, _ = check_route(new_route)
                    if feas_new and dist_new < best_dist:
                        best_dist = dist_new
                        best_route = new_route
                        
            if best_route is None:
                return None
            else:
                current_route = best_route
                
        return current_route

    # Initial Solution
    def initial_solution():
        customers = list(range(1, n_customers + 1))
        customer_angles = []
        for c in customers:
            angle = math.atan2(coords[c][1] - coords[0][1], coords[c][0] - coords[0][0])
            customer_angles.append((angle, c))
        customer_angles.sort()
        sorted_customers = [c for _, c in customer_angles]
        
        routes = []
        curr_route = []
        curr_load = 0.0
        
        for c in sorted_customers:
            if curr_load + demands[c] > capacity:
                if curr_route:
                    routes.append(curr_route)
                curr_route = [c]
                curr_load = demands[c]
            else:
                curr_route.append(c)
                curr_load += demands[c]
                
        if curr_route:
            routes.append(curr_route)
            
        # Repair
        repaired_routes = []
        for r in routes:
            rr = repair_route(r)
            if rr is not None:
                repaired_routes.append(rr)
            else:
                # If repair fails, split?
                # For now, assume it works or fallback
                pass
                
        return repaired_routes

    routes = initial_solution()
    
    # Fallback check
    all_customers = set(range(1, n_customers+1))
    covered = set()
    for r in routes:
        for n in r:
            if is_customer(n):
                covered.add(n)
                
    if covered != all_customers or not routes:
        return [[c] for c in range(1, n_customers+1)]

    current_routes = [list(r) for r in routes]
    route_dists = []
    for r in current_routes:
        feas, d, _ = check_route(r)
        if not feas:
            return [[c] for c in range(1, n_customers+1)]
        route_dists.append(d)
        
    best_routes = [list(r) for r in current_routes]
    best_dist = sum(route_dists)
    
    no_improve = 0
    
    while time.time() < deadline:
        improved = False
        
        # 2-Opt
        for i in range(len(current_routes)):
            route = current_routes[i]
            n = len(route)
            if n < 2: continue
            for p in range(n-1):
                for q in range(p+1, n):
                    new_route = route[:p] + route[p:q+1][::-1] + route[q+1:]
                    feas, d_new, _ = check_route(new_route)
                    if feas and d_new < route_dists[i] - 1e-6:
                        current_routes[i] = new_route
                        route_dists[i] = d_new
                        improved = True
                        no_improve = 0
                        total_new = sum(route_dists)
                        if total_new < best_dist:
                            best_dist = total_new
                            best_routes = [list(r) for r in current_routes]
                        break
                if improved:
                    break
            if improved:
                break
                
        # Relocate
        if not improved:
            for i in range(len(current_routes)):
                for j in range(len(current_routes)):
                    if i == j: continue
                    for u_idx, u in enumerate(current_routes[i]):
                        # Try moving u to j
                        route_i_without_u = current_routes[i][:u_idx] + current_routes[i][u_idx+1:]
                        
                        # Repair route i
                        route_i_repaired = repair_route(route_i_without_u)
                        if route_i_repaired is None:
                            continue
                        feas_i, d_i_new, _ = check_route(route_i_repaired)
                        if not feas_i:
                            continue
                            
                        # Try inserting u into j
                        for v_idx in range(len(current_routes[j]) + 1):
                            new_route_j = current_routes[j][:v_idx] + [u] + current_routes[j][v_idx:]
                            feas_j, d_j_new, _ = check_route(new_route_j)
                            if not feas_j:
                                continue
                                
                            old_dist = route_dists[i] + route_dists[j]
                            new_dist = d_i_new + d_j_new
                            
                            if new_dist < old_dist - 1e-6:
                                # Accept
                                current_routes[i] = route_i_repaired
                                current_routes[j] = new_route_j
                                route_dists[i] = d_i_new
                                route_dists[j] = d_j_new
                                improved = True
                                no_improve = 0
                                total_new = sum(route_dists)
                                if total_new < best_dist:
                                    best_dist = total_new
                                    best_routes = [list(r) for r in current_routes]
                                break # Break v_idx
                        if improved:
                            break # Break u_idx
                    if improved:
                        break # Break j
                if improved:
                    break # Break i
                    
        if not improved:
            no_improve += 1
            if no_improve > 20:
                # Perturb
                if current_routes:
                    i = random.randint(0, len(current_routes)-1)
                    if current_routes[i]:
                        u_idx = random.randint(0, len(current_routes[i])-1)
                        u = current_routes[i][u_idx]
                        current_routes[i].pop(u_idx)
                        j = random.randint(0, len(current_routes)-1)
                        v_idx = random.randint(0, len(current_routes[j])+1)
                        current_routes[j].insert(v_idx, u)
                        
                        # Repair
                        current_routes[i] = repair_route(current_routes[i])
                        current_routes[j] = repair_route(current_routes[j])
                        
                        # Update dists
                        for k in range(len(current_routes)):
                            if current_routes[k] is None:
                                current_routes[k] = []
                            feas, d, _ = check_route(current_routes[k])
                            route_dists[k] = d if feas else float('inf')
                        
                        no_improve = 0
                else:
                    break
        else:
            no_improve = 0
            
    # Final check
    for r in best_routes:
        feas, _, _ = check_route(r)
        if not feas:
            return [[c] for c in range(1, n_customers+1)]
            
    return best_routes

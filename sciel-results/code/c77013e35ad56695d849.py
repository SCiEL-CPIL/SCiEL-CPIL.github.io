import numpy as np
import time
import math
import random
from scipy.spatial.distance import cdist

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Extract parameters
    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    
    # Precompute distance matrix
    dist = cdist(coords, coords)
    
    # Customer IDs are 1 to n_customers
    customers = list(range(1, n_customers + 1))

    def simulate_route(customers_seq):
        """
        Simulates a route of customers. Returns (distance, time, valid, nodes_seq).
        Automatically inserts stations if needed to satisfy battery constraints.
        """
        if not customers_seq:
            return 0.0, 0.0, True, []
        
        total_dist = 0.0
        t = 0.0
        bat = battery_cap
        last = 0
        nodes_seq = []
        
        for c in customers_seq:
            d_direct = dist[last, c]
            
            if bat < d_direct:
                # Need to go to a station
                best_s = -1
                best_cost = float('inf')
                
                for s in stations:
                    d1 = dist[last, s]
                    d2 = dist[s, c]
                    # Check battery for last->s
                    if bat < d1:
                        continue
                    # Check battery for s->c (after full charge)
                    if battery_cap < d2:
                        continue
                    
                    # Cost is extra distance: d1 + d2 - d_direct
                    cost = d1 + d2 - d_direct
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s
                        
                if best_s == -1:
                    return 0.0, 0.0, False, [] # Cannot reach customer even with charging
                
                # Go to station
                d1 = dist[last, best_s]
                total_dist += d1
                bat -= d1
                t += d1 / speed
                
                # Charge
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
                last = best_s
                nodes_seq.append(best_s)
                
                # Go to customer
                d2 = dist[last, c]
                total_dist += d2
                bat -= d2
                t += d2 / speed
                last = c
                
            else:
                # Direct
                total_dist += d_direct
                bat -= d_direct
                t += d_direct / speed
                last = c
            
            nodes_seq.append(c)
            
            # Time window check
            t_start = max(t, ready[c])
            if t_start > due[c]:
                return 0.0, 0.0, False, []
            t = t_start + service[c]
        
        # Return to depot
        d_back = dist[last, 0]
        if bat < d_back:
            # Need to charge before returning?
            best_s = -1
            best_cost = float('inf')
            for s in stations:
                d1 = dist[last, s]
                d2 = dist[s, 0]
                if bat < d1:
                    continue
                if battery_cap < d2:
                    continue
                cost = d1 + d2 - d_back
                if cost < best_cost:
                    best_cost = cost
                    best_s = s
            
            if best_s == -1:
                return 0.0, 0.0, False, []
            
            d1 = dist[last, best_s]
            total_dist += d1
            bat -= d1
            t += d1 / speed
            charge_time = (battery_cap - bat) * recharge_rate
            t += charge_time
            bat = battery_cap
            last = best_s
            nodes_seq.append(best_s)
            
            d2 = dist[last, 0]
            total_dist += d2
            bat -= d2
            t += d2 / speed
        else:
            total_dist += d_back
            bat -= d_back
            t += d_back / speed
        
        if t > horizon:
            return 0.0, 0.0, False, []
        
        return total_dist, t, True, nodes_seq

    # --- Strategy: Clarke-Wright Savings with Feasibility Check ---
    
    # 1. Initial routes: Each customer in its own route
    routes = [[c] for c in customers]
    route_costs = []
    for r in routes:
        cost, _, valid, _ = simulate_route(r)
        if not valid:
            # Should not happen for single customers usually, but handle it
            # If a single customer is infeasible, the problem might be unsolvable or needs specific handling
            # For now, mark as high cost
            route_costs.append(float('inf'))
        else:
            route_costs.append(cost)
            
    # 2. Calculate savings for merging route i and route j
    # Savings = dist(depot, i) + dist(depot, j) - dist(i, j)
    # Note: This is a simplification. Real savings depend on the actual path with stations.
    # We will use the simplified savings to propose merges, then check feasibility.
    
    # Precompute distances from depot to all customers
    d_depot = dist[0, 1:n_customers+1]
    
    # Calculate potential savings between every pair of customers
    # savings[i, j] = d_depot[i] + d_depot[j] - dist[i+1, j+1]
    # i, j are indices 0..n_customers-1 corresponding to customers 1..n_customers
    
    # We need to track which route each customer belongs to
    customer_to_route = {c: i for i, r in enumerate(routes) for c in r}
    
    # List of potential merges (savings, i, j)
    # i, j are customer indices (1-based)
    potential_merges = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            # Only consider if they are in different routes
            if customer_to_route[i] != customer_to_route[j]:
                s = d_depot[i-1] + d_depot[j-1] - dist[i, j]
                potential_merges.append((s, i, j))
                
    # Sort by savings descending
    potential_merges.sort(reverse=True)
    
    # 3. Greedy Merge
    # We iterate through merges. If merging two routes is feasible (capacity, battery, time), we do it.
    
    # Helper to check if two routes can be merged
    # We try both orders: A+B and B+A
    def try_merge(route1, route2):
        """
        Tries to merge route1 and route2.
        Returns (new_route, is_valid, cost) or (None, False, 0)
        """
        # Try route1 + route2
        new_route = route1 + route2
        # Check capacity
        demand_sum = sum(demands[c] for c in new_route)
        if demand_sum > capacity:
            # Try other order? Capacity is symmetric, so if one fails, other fails too.
            return None, False, 0.0
            
        cost1, _, valid1, _ = simulate_route(new_route)
        if valid1:
            return new_route, True, cost1
            
        # Try route2 + route1
        new_route2 = route2 + route1
        cost2, _, valid2, _ = simulate_route(new_route2)
        if valid2:
            return new_route2, True, cost2
            
        return None, False, 0.0

    # We need an efficient way to update customer_to_route and route_costs
    # For n=25, even O(N^3) is fine.
    
    active_merges = [m for m in potential_merges if m[0] > 0]
    
    for s, i, j in active_merges:
        if time.time() > deadline - 10:
            break
            
        r1_idx = customer_to_route.get(i)
        r2_idx = customer_to_route.get(j)
        
        if r1_idx is None or r2_idx is None:
            continue
        if r1_idx == r2_idx:
            continue
            
        r1 = routes[r1_idx]
        r2 = routes[r2_idx]
        
        # Check if i is at the end of r1 and j is at the start of r2? 
        # Clarke-Wright assumes specific positions. 
        # However, since we are just concatenating, we can try all 4 combinations of ends?
        # Actually, standard CW merges end of one with start of another.
        # Let's try:
        # 1. r1 + r2 (end of r1 connects to start of r2)
        # 2. r2 + r1 (end of r2 connects to start of r1)
        # 3. r1 + r2[::-1] (end of r1 connects to end of r2)
        # 4. r2 + r1[::-1] (end of r2 connects to end of r1)
        
        best_new_route = None
        best_cost = float('inf')
        
        candidates = [
            r1 + r2,
            r2 + r1,
            r1 + r2[::-1],
            r2 + r1[::-1]
        ]
        
        for cand in candidates:
            demand_sum = sum(demands[c] for c in cand)
            if demand_sum > capacity:
                continue
            cost, _, valid, _ = simulate_route(cand)
            if valid and cost < best_cost:
                best_cost = cost
                best_new_route = cand
                
        if best_new_route is not None:
            # Perform merge
            # Remove r1 and r2, add best_new_route
            # To keep indices stable, we can mark r1 as the new one and delete r2
            # Or just rebuild the list? N is small.
            
            # Update routes list
            # Find indices
            idx1 = r1_idx
            idx2 = r2_idx
            
            # Remove the larger index first to keep the smaller index valid
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
                
            routes[idx1] = best_new_route
            routes.pop(idx2)
            
            # Update customer_to_route
            for c in best_new_route:
                customer_to_route[c] = idx1
                
            # Update route_costs
            route_costs[idx1] = best_cost
            route_costs.pop(idx2)
            
            # If idx1 > idx2 (which we swapped to avoid), we need to be careful.
            # Actually, we swapped idx1, idx2 so idx1 < idx2.
            # So we kept the lower index.
            
    # 4. Local Search (2-Opt / Or-Opt) on the resulting routes
    
    def local_search_route(route):
        """
        Applies 2-Opt and Or-Opt to a single route to minimize distance.
        """
        if len(route) < 2:
            return route, simulate_route(route)[0]
            
        current_route = route[:]
        current_cost, _, current_valid, _ = simulate_route(current_route)
        if not current_valid:
            return current_route, current_cost
            
        improved = True
        iterations = 0
        max_iterations = 50 # Limit iterations per route
        
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # 2-Opt
            n = len(current_route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_route = current_route[:i] + current_route[i:j+1][::-1] + current_route[j+1:]
                    # Check capacity (unchanged)
                    # Check feasibility
                    new_cost, _, new_valid, _ = simulate_route(new_route)
                    if new_valid and new_cost < current_cost - 1e-9:
                        current_route = new_route
                        current_cost = new_cost
                        improved = True
                        # Break loops to restart? Or just continue?
                        # Restarting is safer for convergence
                        break
                if improved:
                    break
                    
            if improved:
                continue
                
            # Or-Opt (move 1 or 2 nodes)
            # Move 1 node
            for i in range(n):
                node = current_route[i]
                for j in range(n):
                    if j == i or j == i + 1:
                        continue
                    # Remove node from i
                    new_route = current_route[:i] + current_route[i+1:]
                    # Insert at j
                    # Note: j index might shift if i < j
                    if j > i:
                        j -= 1
                    new_route = new_route[:j] + [node] + new_route[j:]
                    
                    new_cost, _, new_valid, _ = simulate_route(new_route)
                    if new_valid and new_cost < current_cost - 1e-9:
                        current_route = new_route
                        current_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
                    
            if improved:
                continue
                
            # Move 2 nodes
            if n >= 4:
                for i in range(n - 1):
                    nodes = current_route[i:i+2]
                    for j in range(n - 1):
                        if j >= i and j < i + 2:
                            continue
                        # Remove nodes from i
                        new_route = current_route[:i] + current_route[i+2:]
                        # Insert at j
                        if j > i:
                            j -= 2
                        new_route = new_route[:j] + nodes + new_route[j:]
                        
                        new_cost, _, new_valid, _ = simulate_route(new_route)
                        if new_valid and new_cost < current_cost - 1e-9:
                            current_route = new_route
                            current_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                        
        return current_route, current_cost

    # Apply local search to all routes
    final_routes = []
    total_dist = 0.0
    
    for r in routes:
        if not r:
            continue
        new_r, cost = local_search_route(r)
        if new_r:
            final_routes.append(new_r)
            total_dist += cost
        else:
            # Fallback if local search failed (shouldn't happen)
            final_routes.append(r)
            cost, _, valid, _ = simulate_route(r)
            if valid:
                total_dist += cost
            else:
                # This route is invalid. We must fix it.
                # Split it into single customers?
                for c in r:
                    final_routes.append([c])
                    c_cost, _, c_valid, _ = simulate_route([c])
                    if c_valid:
                        total_dist += c_cost

    # 5. Inter-route moves? (Relocate customer from route A to route B)
    # Given time constraints, we might skip this or do a very limited version.
    # Let's do a simple relocation pass if time permits.
    
    if time.time() < deadline - 15:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n_routes = len(final_routes)
            for i in range(n_routes):
                if len(final_routes[i]) <= 1:
                    continue
                for j in range(n_routes):
                    if i == j:
                        continue
                    # Try to move a customer from i to j
                    # Pick the customer in i that, if moved, improves total cost
                    # This is expensive. Let's just try moving the "most expensive" customer?
                    # Or just iterate all customers in i.
                    
                    for k in range(len(final_routes[i])):
                        c = final_routes[i][k]
                        
                        # Remove c from i
                        route_i_new = final_routes[i][:k] + final_routes[i][k+1:]
                        # Insert c into j (try all positions)
                        best_insert_cost = float('inf')
                        best_insert_pos = -1
                        best_route_j_new = None
                        
                        route_j = final_routes[j]
                        
                        # Check capacity for route_j + c
                        demand_j = sum(demands[x] for x in route_j)
                        if demand_j + demands[c] > capacity:
                            continue
                            
                        for pos in range(len(route_j) + 1):
                            cand = route_j[:pos] + [c] + route_j[pos:]
                            cost_j, _, valid_j, _ = simulate_route(cand)
                            if valid_j and cost_j < best_insert_cost:
                                best_insert_cost = cost_j
                                best_insert_pos = pos
                                best_route_j_new = cand
                                
                        if best_route_j_new is None:
                            continue
                            
                        # Check route_i_new
                        cost_i, _, valid_i, _ = simulate_route(route_i_new)
                        if not valid_i:
                            continue
                            
                        # Total change
                        old_cost_i, _, _, _ = simulate_route(final_routes[i])
                        old_cost_j, _, _, _ = simulate_route(route_j)
                        
                        new_total = cost_i + best_insert_cost
                        old_total = old_cost_i + old_cost_j
                        
                        if new_total < old_total - 1e-9:
                            # Perform move
                            final_routes[i] = route_i_new
                            final_routes[j] = best_route_j_new
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    # break outer loops? 
                    # The structure is nested. 
                    # We break inner, then check 'improved' in middle, break middle, check 'improved' in outer, break outer.
                    # This logic is tricky with multiple breaks. 
                    # Let's just break the while loop if improved, and restart the scan.
                    break

    # 6. Final Validation and Node Sequence Generation
    
    result_routes = []
    for r in final_routes:
        if not r:
            continue
        # Ensure validity one last time
        d, t, valid, nodes_seq = simulate_route(r)
        if valid:
            result_routes.append(nodes_seq)
        else:
            # Fallback: split into singles
            for c in r:
                d, t, valid_c, nodes_c = simulate_route([c])
                if valid_c:
                    result_routes.append(nodes_c)
                else:
                    # Absolute fallback: just the customer (might be invalid but better than crash)
                    result_routes.append([c])

    return result_routes

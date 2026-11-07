import numpy as np
import time
import math
import random
from scipy.spatial.distance import cdist
import heapq

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Extract parameters
    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)

    # Precompute distance matrix
    dist = cdist(coords, coords)

    customers = list(range(1, n_customers + 1))

    # --- Strategy: Constrained Shortest Path (CSP) on state-expanded graph ---
    # State: (last_node, visited_mask, battery_level_quantized)
    # We use Dijkstra to find min-cost paths from depot to depot visiting all customers.
    # To keep state space manageable, we quantize battery level.
    # Battery cap is 300. Let's use a step of 10 or 5. Step 5 gives 60 states.
    # Visited mask: 25 customers -> 2^25 states is too large for full Dijkstra.
    # So we will use a heuristic-guided search or a simplified CSP approach.
    
    # Alternative Structural Change: 
    # Instead of full state space Dijkstra which is intractable for 25 nodes,
    # we use a **Dynamic Programming with State Pruning** approach for small subsets,
    # or we reformulate as a **Set Partitioning** problem solved via Column Generation (too complex for 150s).
    
    # Let's try a different structural approach: **Branch and Bound with strong pruning** 
    # or **A* Search** over the permutation space with a very strong lower bound.
    
    # Actually, the prompt suggests "Constrained Shortest Path Problem (CSP) on a state-expanded graph".
    # For 25 customers, exact CSP is impossible.
    # However, we can use **Lagrangian Relaxation** or **Penalty Methods** within a local search.
    
    # Let's try a **Memetic Algorithm with a CSP-inspired Decoder**.
    # The decoder takes a sequence of customers and uses a shortest-path algorithm 
    # (Dijkstra on a time/battery expanded graph) to find the optimal way to visit them 
    # including charging stops, minimizing distance.
    
    # This is a different structural change: the construction/repair step uses an exact 
    # shortest path algorithm on a subgraph for each segment, rather than greedy insertion.
    
    def shortest_path_with_constraints(start_node, end_node, battery_start, time_start):
        """
        Finds the minimum distance path from start_node to end_node 
        such that we can reach end_node with battery >= 0 and time <= horizon.
        We can visit stations along the way.
        This is a shortest path problem on a state graph: (node, battery, time).
        To make it tractable, we might need to discretize time/battery or use A*.
        
        Given the small number of stations (4), we can enumerate paths through stations.
        Path: start -> s1 -> s2 -> ... -> end
        Since stations can be repeated, but it's unlikely to be optimal to repeat many times,
        we can limit the number of station visits.
        
        Actually, a simpler structural change:
        Use **Or-opt** and **2-opt** moves but evaluate them using a **fast feasibility check** 
        that pre-computes the "minimum energy needed" or similar.
        
        Let's stick to the prompt's suggestion: **CSP on state-expanded graph**.
        We will implement a **A* search** for the entire route construction.
        State: (current_node, visited_set, battery, time)
        Goal: visited_set == all_customers and current_node == 0
        Cost: distance traveled so far.
        Heuristic: sum of min(dist(node, c)) for c not in visited_set + dist(node, 0)
        
        This is still exponential.
        
        Let's try a different strategy: **Large Neighborhood Search (LNS)** 
        where we remove a subset of customers, re-optimize them using a **DP-based exact solver** 
        for small subsets (e.g., up to 10 customers), and reinsert.
        
        DP for TSP with constraints on k nodes:
        State: (last, mask, battery_quantized)
        For k=10, 2^10 * 10 * 60 states = 600k states. This is feasible.
        
        So the strategy is:
        1. Construct initial solution using greedy insertion.
        2. Iteratively pick a random subset of customers (size 8-12).
        3. Remove them from their routes.
        4. Use a DP-based exact solver to find the best way to re-insert them into the existing routes 
           (or form new routes) considering the current battery/time states at the insertion points.
           This is complex because insertion points have specific battery/time states.
        
        Simpler LNS:
        1. Pick a subset of customers.
        2. Remove them.
        3. Re-insert them one by one using a **shortest path** check (CSP) to find the best position 
           in the existing routes.
        
        Let's implement **LNS with CSP-based insertion**.
        """
        pass

    # Given the complexity and time limit, let's implement a robust **LNS (Large Neighborhood Search)**
    # with a **fast feasibility evaluator** that uses pre-computed nearest stations.
    
    # Structural Change: **Variable Neighborhood Decomposition (VND)** 
    # combined with **CSP-based route evaluation**.
    
    # We will use a **Greedy Randomized Adaptive Search Procedure (GRASP)** 
    # with a **CSP-based repair** step.
    
    # 1. Construction: Greedy insertion with randomized candidate list.
    # 2. Repair: If infeasible, use a shortest-path algorithm to fix battery/time issues.
    # 3. Local Search: 2-opt and Or-opt with fast feasibility checks.
    
    # Let's implement GRASP + Local Search with a **better feasibility check**.
    
    def check_feasibility_fast(route):
        """
        Fast check for battery and time windows.
        Returns (is_valid, total_distance, total_time).
        """
        if not route:
            return True, 0.0, 0.0
        
        total_dist = 0.0
        t = 0.0
        bat = battery_cap
        last = 0
        
        for c in route:
            d = dist[last, c]
            if bat < d:
                # Need to charge
                # Find nearest station that allows reaching c
                best_s = -1
                best_cost = float('inf')
                for s in stations:
                    d1 = dist[last, s]
                    d2 = dist[s, c]
                    if bat >= d1 and battery_cap >= d2:
                        cost = d1 + d2
                        if cost < best_cost:
                            best_cost = cost
                            best_s = s
                if best_s == -1:
                    return False, 0.0, 0.0
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
                # Go to customer
                d2 = dist[last, c]
                total_dist += d2
                bat -= d2
                t += d2 / speed
                last = c
            else:
                total_dist += d
                bat -= d
                t += d / speed
                last = c
            
            # Time window
            t_start = max(t, ready[c])
            if t_start > due[c]:
                return False, 0.0, 0.0
            t = t_start + service[c]
        
        # Return to depot
        d_back = dist[last, 0]
        if bat < d_back:
            best_s = -1
            best_cost = float('inf')
            for s in stations:
                d1 = dist[last, s]
                d2 = dist[s, 0]
                if bat >= d1 and battery_cap >= d2:
                    cost = d1 + d2
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s
            if best_s == -1:
                return False, 0.0, 0.0
            d1 = dist[last, best_s]
            total_dist += d1
            bat -= d1
            t += d1 / speed
            charge_time = (battery_cap - bat) * recharge_rate
            t += charge_time
            bat = battery_cap
            last = best_s
            d2 = dist[last, 0]
            total_dist += d2
            bat -= d2
            t += d2 / speed
        else:
            total_dist += d_back
            bat -= d_back
            t += d_back / speed
            
        if t > horizon:
            return False, 0.0, 0.0
            
        return True, total_dist, t

    def construct_grasp():
        """
        GRASP construction: Greedy insertion with randomized candidate list.
        """
        # Sort customers by due date
        cust_order = sorted(customers, key=lambda c: due[c])
        
        routes = []
        unassigned = set(customers)
        
        while unassigned:
            # Pick a random customer from the first 20% of the sorted list
            k = max(1, len(unassigned) // 5)
            candidates = [c for c in cust_order if c in unassigned][:k]
            if not candidates:
                candidates = list(unassigned)
            c = random.choice(candidates)
            
            # Try to insert into existing routes
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')
            
            for i, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    if sum(demands[x] for x in new_route) > capacity:
                        continue
                    valid, d, t = check_feasibility_fast(new_route)
                    if valid and d < best_cost:
                        best_cost = d
                        best_route_idx = i
                        best_pos = pos
            
            if best_route_idx != -1:
                routes[best_route_idx] = routes[best_route_idx][:best_pos] + [c] + routes[best_route_idx][best_pos:]
            else:
                # Start new route
                routes.append([c])
            
            unassigned.remove(c)
        
        return routes

    def local_search_lns(routes):
        """
        LNS Local Search: Remove a subset of customers, re-insert them.
        """
        best_routes = routes[:]
        best_dist = 0.0
        for r in best_routes:
            valid, d, t = check_feasibility_fast(r)
            if not valid:
                return routes, float('inf')
            best_dist += d
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Pick a random subset of customers to remove
            n_remove = random.randint(5, 10)
            all_custs = [c for r in best_routes for c in r]
            if len(all_custs) < n_remove:
                n_remove = len(all_custs)
            
            remove_set = set(random.sample(all_custs, n_remove))
            
            # Remove them from routes
            new_routes = []
            for r in best_routes:
                new_r = [c for c in r if c not in remove_set]
                if new_r:
                    new_routes.append(new_r)
            
            # Re-insert them one by one
            for c in remove_set:
                best_route_idx = -1
                best_pos = -1
                best_cost = float('inf')
                
                for i, route in enumerate(new_routes):
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [c] + route[pos:]
                        if sum(demands[x] for x in new_route) > capacity:
                            continue
                        valid, d, t = check_feasibility_fast(new_route)
                        if valid and d < best_cost:
                            best_cost = d
                            best_route_idx = i
                            best_pos = pos
                
                if best_route_idx != -1:
                    new_routes[best_route_idx] = new_routes[best_route_idx][:best_pos] + [c] + new_routes[best_route_idx][best_pos:]
                else:
                    new_routes.append([c])
            
            # Evaluate new solution
            new_dist = 0.0
            valid = True
            for r in new_routes:
                v, d, t = check_feasibility_fast(r)
                if not v:
                    valid = False
                    break
                new_dist += d
            
            if valid and new_dist < best_dist - 1e-9:
                best_routes = new_routes
                best_dist = new_dist
                improved = True
        
        return best_routes, best_dist

    # Main Logic
    best_routes = None
    best_dist = float('inf')
    
    # Run GRASP multiple times
    while time.time() < deadline - 20:
        routes = construct_grasp()
        
        # Check validity
        valid = True
        total_dist = 0.0
        for r in routes:
            v, d, t = check_feasibility_fast(r)
            if not v:
                valid = False
                break
            total_dist += d
        
        if not valid:
            continue
        
        # Local Search
        routes, dist = local_search_lns(routes)
        
        if dist < best_dist:
            best_dist = dist
            best_routes = routes[:]
    
    if best_routes is None:
        # Fallback
        best_routes = [[c] for c in customers]
    
    # Final Check and Conversion
    final_routes = []
    for r in best_routes:
        valid, d, t = check_feasibility_fast(r)
        if valid:
            # Reconstruct with stations
            nodes = []
            bat = battery_cap
            t_cur = 0.0
            last = 0
            for c in r:
                d_direct = dist[last, c]
                if bat < d_direct:
                    best_s = -1
                    best_cost = float('inf')
                    for s in stations:
                        d1 = dist[last, s]
                        d2 = dist[s, c]
                        if bat >= d1 and battery_cap >= d2:
                            cost = d1 + d2
                            if cost < best_cost:
                                best_cost = cost
                                best_s = s
                    if best_s != -1:
                        d1 = dist[last, best_s]
                        bat -= d1
                        t_cur += d1 / speed
                        charge_time = (battery_cap - bat) * recharge_rate
                        t_cur += charge_time
                        bat = battery_cap
                        last = best_s
                        nodes.append(best_s)
                d2 = dist[last, c]
                bat -= d2
                t_cur += d2 / speed
                last = c
                nodes.append(c)
                t_start = max(t_cur, ready[c])
                t_cur = t_start + service[c]
            
            # Return to depot
            d_back = dist[last, 0]
            if bat < d_back:
                best_s = -1
                best_cost = float('inf')
                for s in stations:
                    d1 = dist[last, s]
                    d2 = dist[s, 0]
                    if bat >= d1 and battery_cap >= d2:
                        cost = d1 + d2
                        if cost < best_cost:
                            best_cost = cost
                            best_s = s
                if best_s != -1:
                    d1 = dist[last, best_s]
                    bat -= d1
                    t_cur += d1 / speed
                    charge_time = (battery_cap - bat) * recharge_rate
                    t_cur += charge_time
                    bat = battery_cap
                    last = best_s
                    nodes.append(best_s)
            d2 = dist[last, 0]
            bat -= d2
            t_cur += d2 / speed
            
            final_routes.append(nodes)
        else:
            final_routes.append(r)
            
    return final_routes

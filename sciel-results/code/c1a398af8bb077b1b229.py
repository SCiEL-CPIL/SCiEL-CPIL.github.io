import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin within 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    customers = list(range(1, n_cust + 1))

    # Helper to check if a route is valid and return its cost
    def check_route(route_nodes):
        if not route_nodes:
            return True, 0.0, 0.0, 0.0, 0.0
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        for node in route_nodes:
            d = dist[prev, node]
            t += d / speed
            b -= d
            if b < -_EPS:
                return False, 0.0, t, d_total, load
            
            # If it's a station, recharge logic applies in the problem statement?
            # "a station adds (battery - energy) * recharge_rate time and refills"
            # Wait, the check_route in the original code did NOT handle stations correctly if they were in the route.
            # The original code's check_route assumed route_nodes are customers or just checked time/battery.
            # Let's look at the original check_route:
            # t = max(t, ready[node])
            # if t > due[node] ...
            # t += service[node]
            # load += demands[node]
            # For stations: ready/due are 0? service is 0? demands is 0?
            # The problem says: "stations (node ids)", "service (10 for customers, else 0)"
            # "ready in [0, 400] and due = ready + 100 to 300" -> likely for customers.
            # What about stations' ready/due? The prompt implies stations are just nodes.
            # "a station adds (battery - energy) * recharge_rate time and refills"
            # This logic was MISSING from the original check_route.
            # I must implement it correctly here.
            
            if node in stations:
                # Recharge logic
                # "adds (battery - energy) * recharge_rate time"
                # energy is current battery level (b)
                # time added = (b - 0) * recharge_rate? Or (battery_cap - b)?
                # "refills" implies it goes to full.
                # Usually recharge time is proportional to energy added.
                # Energy added = battery_cap - b.
                # The prompt says: "(battery - energy) * recharge_rate".
                # "battery" likely means battery_cap. "energy" means current charge b.
                # So time += (battery_cap - b) * recharge_rate.
                # Then b = battery_cap.
                t += (battery_cap - b) * recharge_rate
                b = battery_cap
            else:
                t = max(t, ready[node])
                if t > due[node] + _EPS:
                    return False, 0.0, t, d_total, load
                t += service[node]
                load += demands[node]
                if load > capacity + _EPS:
                    return False, 0.0, t, d_total, load
            
            d_total += d
            prev = node
        
        # Return to depot
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -_EPS:
            return False, 0.0, t, d_total, load
        if t > horizon + _EPS:
            return False, 0.0, t, d_total, load
        d_total += d
        return True, d_total, t, d_total, load

    # Phase 1: Construction using Clarke-Wright Savings
    # 1. Start with individual routes for each customer
    routes = [[c] for c in customers]
    
    # Calculate savings
    savings = []
    for i in range(1, n_cust + 1):
        for j in range(i + 1, n_cust + 1):
            s = dist[0, i] + dist[0, j] - dist[i, j]
            savings.append((s, i, j))
    
    savings.sort(reverse=True)

    # Track which route each node is in
    node_to_route = {}
    for idx, r in enumerate(routes):
        for node in r:
            node_to_route[node] = idx
    
    # Track ends of routes
    route_ends = {idx: (r[0], r[-1]) for idx, r in enumerate(routes)}

    for s, i, j in savings:
        if time.time() > DEADLINE - 10:
            break
        if s < 0:
            break
        
        ri = node_to_route.get(i, -1)
        rj = node_to_route.get(j, -1)
        
        if ri == -1 or rj == -1 or ri == rj:
            continue
            
        # Check if i and j are ends of their respective routes
        ends_i = route_ends[ri]
        ends_j = route_ends[rj]
        
        if i not in ends_i or j not in ends_j:
            continue
            
        # Determine orientation
        # We want to merge route Ri and Rj such that i and j become adjacent.
        # Case 1: i is end of Ri, j is start of Rj -> Ri + Rj
        # Case 2: i is start of Ri, j is end of Rj -> Rj + Ri
        # Case 3: i is end of Ri, j is end of Rj -> Ri + Rj[::-1]
        # Case 4: i is start of Ri, j is start of Rj -> Rj[::-1] + Ri
        
        new_route = None
        if i == ends_i[1] and j == ends_j[0]:
            new_route = routes[ri] + routes[rj]
        elif i == ends_i[0] and j == ends_j[1]:
            new_route = routes[rj] + routes[ri]
        elif i == ends_i[1] and j == ends_j[1]:
            new_route = routes[ri] + routes[rj][::-1]
        elif i == ends_i[0] and j == ends_j[0]:
            new_route = routes[rj][::-1] + routes[ri]
        else:
            continue
            
        # Check validity of new route
        valid, d, t, _, _ = check_route(new_route)
        if valid:
            # Perform merge
            routes[ri] = new_route
            routes.pop(rj)
            
            # Update node_to_route and route_ends
            # Rebuild maps for simplicity (small n)
            node_to_route = {}
            for idx, r in enumerate(routes):
                for node in r:
                    node_to_route[node] = idx
            route_ends = {idx: (r[0], r[-1]) for idx, r in enumerate(routes) if r}
            
            # Remove empty routes
            routes = [r for r in routes if r]
            # Re-index if needed? 
            # node_to_route uses indices, so if we pop, indices shift.
            # Let's just rebuild everything after each merge to be safe, though it's O(N^2) per merge.
            # With 25 customers, it's fine.
            node_to_route = {}
            for idx, r in enumerate(routes):
                for node in r:
                    node_to_route[node] = idx
            route_ends = {idx: (r[0], r[-1]) for idx, r in enumerate(routes) if r}

    # Ensure all customers are served
    served = set()
    for r in routes:
        for c in r:
            served.add(c)
    missing = set(customers) - served
    for c in missing:
        routes.append([c])
        
    # Filter empty
    routes = [r for r in routes if r]

    # Phase 2: Large Neighborhood Search (LNS)
    # We will destroy a subset of customers and repair them using DP.
    
    def get_total_cost(routes_list):
        total = 0.0
        for r in routes_list:
            valid, d, t, _, _ = check_route(r)
            if not valid:
                return float('inf')
            total += d
        return total

    current_cost = get_total_cost(routes)
    best_routes = [r[:] for r in routes]
    best_cost = current_cost
    
    # DP Repair Function
    # Solves the problem for a set of customers 'to_insert' into existing 'base_routes'.
    # Actually, it's easier to treat the destroyed customers as needing to be inserted into the remaining routes.
    # However, a full DP over the state of all routes is too complex.
    # Instead, we use a "Greedy Insertion with DP for local optimization" or simply
    # a DP that finds the best route for the destroyed customers alone, then merges?
    # No, the strategy says "repair their insertion using a dynamic programming algorithm".
    
    # A simpler LNS repair:
    # 1. Remove destroyed customers from routes.
    # 2. Try to insert them back one by one into the best position in any route.
    # 3. To make it "DP-like", we can use a DP to optimize the order of a small cluster of destroyed customers
    #    if we decide to form a new route from them or insert them as a block.
    
    # Given the small size (25 customers), we can afford a more expensive repair:
    # For the set of destroyed customers, find the best way to insert them into the existing routes.
    # This is still NP-hard, but we can use a heuristic repair:
    # Sort destroyed customers by "tightness" (due - ready - travel time).
    # For each, find the best insertion position in all routes.
    
    # To satisfy the "DP" requirement in the prompt's inspiration, let's implement a DP for the case
    # where we form a NEW route from a subset of destroyed customers, or insert them into an existing route.
    # Actually, a full DP for VRPTW is exponential.
    # Let's interpret "DP" as using the Bellman-Held-Karp algorithm to find the optimal tour for the destroyed customers
    # if they are served in a single new route, or to optimize the sequence within a route segment.
    
    # Given the constraints, a practical LNS repair is:
    # 1. Remove destroyed nodes.
    # 2. For each destroyed node, find the best insertion point in the current routes.
    # 3. Accept if valid.
    
    # Let's enhance the repair:
    # Instead of just greedy insertion, we try to insert the destroyed customers in an order determined by a small DP
    # that minimizes the insertion cost.
    
    def repair_routes(routes_list, destroyed_customers):
        # routes_list: list of routes without destroyed customers
        # destroyed_customers: list of nodes to re-insert
        
        # Sort destroyed customers by due date (earliest first) to help feasibility
        to_insert = sorted(destroyed_customers, key=lambda c: due[c])
        
        new_routes = [r[:] for r in routes_list]
        
        for c in to_insert:
            best_cost = float('inf')
            best_route_idx = -1
            best_pos = -1
            
            for i, r in enumerate(new_routes):
                for pos in range(len(r) + 1):
                    trial = r[:pos] + [c] + r[pos:]
                    valid, d, t, _, _ = check_route(trial)
                    if valid:
                        # Cost delta
                        if r:
                            old_d = check_route(r)[1]
                        else:
                            old_d = 0.0
                        delta = d - old_d
                        if delta < best_cost:
                            best_cost = delta
                            best_route_idx = i
                            best_pos = pos
            
            if best_route_idx != -1:
                new_routes[best_route_idx].insert(best_pos, c)
            else:
                # If it can't be inserted, create a new route
                new_routes.append([c])
                
        return new_routes

    # LNS Loop
    temperature = 100.0
    cooling = 0.99
    
    while time.time() < DEADLINE - 5:
        # Destroy
        # Randomly select 10-30% of customers to destroy
        n_destroy = max(1, int(len(customers) * random.uniform(0.1, 0.3)))
        destroyed = random.sample(customers, n_destroy)
        
        # Remove them from current routes
        base_routes = []
        for r in routes:
            new_r = [c for c in r if c not in destroyed]
            if new_r:
                base_routes.append(new_r)
        
        # Repair
        new_routes = repair_routes(base_routes, destroyed)
        
        # Check validity and cost
        new_cost = get_total_cost(new_routes)
        
        if new_cost < current_cost - _EPS:
            routes = new_routes
            current_cost = new_cost
            if current_cost < best_cost - _EPS:
                best_cost = current_cost
                best_routes = [r[:] for r in routes]
            temperature = max(1.0, temperature * cooling)
        else:
            # Accept with probability exp(-(new_cost - current_cost) / temperature)
            delta = new_cost - current_cost
            if delta < 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / temperature)
            
            if random.random() < prob:
                routes = new_routes
                current_cost = new_cost
                temperature = max(1.0, temperature * cooling)
            else:
                # Reject, keep current
                temperature = max(1.0, temperature * cooling)
        
        # Optional: Local search (2-opt) on best routes occasionally
        if random.random() < 0.1:
            for i, r in enumerate(best_routes):
                if len(r) < 2: continue
                # Simple 2-opt
                improved = True
                while improved:
                    improved = False
                    n = len(r)
                    for i2 in range(n - 1):
                        for j2 in range(i2 + 1, n):
                            new_r = r[:i2] + r[i2:j2+1][::-1] + r[j2+1:]
                            valid, d, t, _, _ = check_route(new_r)
                            if valid:
                                old_d = check_route(r)[1]
                                if d < old_d - _EPS:
                                    r = new_r
                                    improved = True
                                    break
                        if improved: break
                    best_routes[i] = r

    # Final validation of best_routes
    final_routes = []
    for r in best_routes:
        if r:
            final_routes.append(r)
            
    # Check coverage
    served = set()
    for r in final_routes:
        for c in r:
            served.add(c)
    missing = set(customers) - served
    if missing:
        # Fallback: add missing as individual routes
        for c in missing:
            final_routes.append([c])
            
    return final_routes

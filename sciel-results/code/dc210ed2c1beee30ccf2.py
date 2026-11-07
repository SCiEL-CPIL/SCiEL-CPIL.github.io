"""Rewritten program for EV routing with time windows."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import copy

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
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
    cust_set = set(customers)
    station_set = set(stations)

    # Precompute demand array for fast access
    cust_demand = np.zeros(n_nodes)
    for c in customers:
        cust_demand[c] = demands[c]

    def simulate_route(route):
        """
        Simulates a route. Returns (feasible, total_dist, end_time, end_battery).
        """
        if not route:
            return True, 0.0, 0.0, battery_cap

        cur_time = 0.0
        cur_battery = battery_cap
        total_dist = 0.0
        cur_pos = 0

        for node in route:
            d = dist[cur_pos][node]
            cur_time += d / speed
            cur_battery -= d
            total_dist += d

            if cur_battery < -1e-9:
                return False, total_dist, cur_time, cur_battery

            if node in station_set:
                charge_time = (battery_cap - cur_battery) * recharge_rate
                cur_time += charge_time
                cur_battery = battery_cap
            else:
                # Customer
                if cur_time < ready[node]:
                    cur_time = ready[node]
                if cur_time > due[node] + 1e-9:
                    return False, total_dist, cur_time, cur_battery
                cur_time += service[node]
            
            cur_pos = node

        # Return to depot
        d = dist[cur_pos][0]
        cur_time += d / speed
        cur_battery -= d
        total_dist += d

        if cur_battery < -1e-9:
            return False, total_dist, cur_time, cur_battery

        if cur_time > horizon + 1e-9:
            return False, total_dist, cur_time, cur_battery

        return True, total_dist, cur_time, cur_battery

    def route_demand(route):
        return sum(demands[c] for c in route if c in cust_set)

    def fix_battery(route):
        """
        Attempts to make a route feasible by inserting charging stations.
        Returns (feasible_route, distance) or (None, None).
        """
        f, d, _, _ = simulate_route(route)
        if f:
            return route, d
        
        best_route = None
        best_d = float('inf')
        
        # Try inserting one station at all positions
        for st in stations:
            for pos in range(len(route) + 1):
                cand = route[:pos] + [st] + route[pos:]
                f, d, _, _ = simulate_route(cand)
                if f and d < best_d:
                    best_d = d
                    best_route = cand
                    
        if best_route is not None:
            return best_route, best_d
            
        # If one station isn't enough, try two (expensive, but rare)
        # Only do this if the route is small or we have time
        if len(route) < 15:
            for st1 in stations:
                for st2 in stations:
                    for pos1 in range(len(route) + 1):
                        # Avoid inserting st2 at pos1 if st1 is already there? 
                        # Simple approach: generate candidate with st1, then insert st2
                        cand1 = route[:pos1] + [st1] + route[pos1:]
                        for pos2 in range(len(cand1) + 1):
                            cand2 = cand1[:pos2] + [st2] + cand1[pos2:]
                            # Optional: skip if st2 is adjacent to st1 and redundant? 
                            # simulate_route handles logic, so just check
                            f, d, _, _ = simulate_route(cand2)
                            if f and d < best_d:
                                best_d = d
                                best_route = cand2
                                
        return best_route, best_d

    def construct_savings():
        """
        Clarke-Wright savings construction with battery/time constraints.
        """
        # Initialize with single customers
        active_routes = []
        for c in customers:
            # Ensure singleton is feasible (it usually is, but fix if not)
            r, d = fix_battery([c])
            if r is None:
                r = [c] # Fallback, likely infeasible but we'll handle later
            active_routes.append(r)
        
        # Calculate savings between all pairs of routes
        # To be efficient, we re-calculate savings as routes merge.
        # Standard CW: sort savings s(i,j) = d(0,i)+d(0,j)-d(i,j)
        # Here, routes are lists. The "end" of a route is the last node.
        
        # For simplicity and robustness in this context, we do a greedy merge:
        # Find the pair of routes that gives the largest distance saving when merged.
        
        while len(active_routes) > 1:
            if time.time() > deadline - 10:
                break
                
            best_save = 0
            best_i = -1
            best_j = -1
            best_merged = None
            
            n = len(active_routes)
            for i in range(n):
                ri = active_routes[i]
                for j in range(i + 1, n):
                    rj = active_routes[j]
                    
                    # Check capacity
                    if route_demand(ri) + route_demand(rj) > capacity:
                        continue
                    
                    # Try merging in 4 orientations
                    # ri + rj, ri + rj[::-1], ri[::-1] + rj, ri[::-1] + rj[::-1]
                    orientations = [
                        ri + rj,
                        ri + rj[::-1],
                        ri[::-1] + rj,
                        ri[::-1] + rj[::-1]
                    ]
                    
                    for cand in orientations:
                        # Quick distance check without full simulation if possible?
                        # Full simulation is safer for time windows.
                        f, d, _, _ = simulate_route(cand)
                        if f:
                            # Calculate saving
                            d_ri, _, _, _ = simulate_route(ri)
                            d_rj, _, _, _ = simulate_route(rj)
                            save = d_ri + d_rj - d
                            if save > best_save:
                                best_save = save
                                best_i = i
                                best_j = j
                                best_merged = cand
                                break # No need to check other orientations for this pair if this one works?
                                # Actually, we should check all orientations to find the best one for this pair.
                                # But for speed, taking the first feasible one with good saving is often okay.
                                # Let's refine: check all 4 and pick best d.
                    
                    # Refinement: Check all 4 orientations for this pair and pick best
                    best_pair_d = float('inf')
                    best_pair_cand = None
                    for cand in orientations:
                        f, d, _, _ = simulate_route(cand)
                        if f and d < best_pair_d:
                            best_pair_d = d
                            best_pair_cand = cand
                            
                    if best_pair_cand is not None:
                        d_ri, _, _, _ = simulate_route(ri)
                        d_rj, _, _, _ = simulate_route(rj)
                        save = d_ri + d_rj - best_pair_d
                        if save > best_save:
                            best_save = save
                            best_i = i
                            best_j = j
                            best_merged = best_pair_cand
            
            if best_i == -1:
                # No more feasible merges
                break
                
            # Merge
            new_route = best_merged
            # If direct merge failed, try fix_battery
            f, _, _, _ = simulate_route(new_route)
            if not f:
                fixed, _ = fix_battery(new_route)
                if fixed is not None:
                    new_route = fixed
                else:
                    # Can't merge, skip this pair? 
                    # To avoid infinite loop, we must remove one of them or mark them.
                    # Simple strategy: if we can't merge, we stop trying to merge this specific pair?
                    # Actually, if best_save > 0 but merge fails even with fix, we might be stuck.
                    # Let's just break if we can't merge the best candidate.
                    break
            
            active_routes[best_i] = new_route
            active_routes.pop(best_j)
            
        return active_routes

    def two_opt_intra(route):
        """
        Returns improved route if 2-opt finds improvement, else None.
        """
        if len(route) < 3:
            return None
        
        best_route = route
        best_dist = simulate_route(route)[1]
        
        n = len(route)
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse segment i+1 to j
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    f, d, _, _ = simulate_route(new_route)
                    if f and d < best_dist - 1e-6:
                        best_dist = d
                        best_route = new_route
                        route = new_route # Update current route for next iteration
                        improved = True
                        break
                if improved:
                    break
        return best_route

    def or_opt_move(routes, i, j, k):
        """
        Move segment [j:k] from route i to another position or route.
        Returns (improved, new_routes)
        """
        route_i = routes[i]
        if k > len(route_i) or j < 0 or k - j > 3:
            return False, routes
        
        segment = route_i[j:k]
        custs_in_seg = [c for c in segment if c in cust_set]
        if not custs_in_seg:
            return False, routes

        # Remove segment from route i
        new_route_i = route_i[:j] + route_i[k:]
        f_i, d_i, _, _ = simulate_route(new_route_i)
        if not f_i:
            return False, routes
        
        old_total_dist = simulate_route(route_i)[1]
        
        # Try inserting into same route i
        for pos in range(len(new_route_i) + 1):
            # Skip positions that result in same order
            if pos >= j and pos <= k:
                continue
            new_route_i_test = new_route_i[:pos] + segment + new_route_i[pos:]
            if route_demand(new_route_i_test) > capacity:
                continue
            f, d, _, _ = simulate_route(new_route_i_test)
            if f and d < old_total_dist - 1e-6:
                new_routes = routes[:]
                new_routes[i] = new_route_i_test
                return True, new_routes

        # Try inserting into other routes
        for k2 in range(len(routes)):
            if k2 == i:
                continue
            route_k = routes[k2]
            old_dist_k = simulate_route(route_k)[1]
            
            for pos in range(len(route_k) + 1):
                new_route_k = route_k[:pos] + segment + route_k[pos:]
                if route_demand(new_route_k) > capacity:
                    continue
                f, d, _, _ = simulate_route(new_route_k)
                if f:
                    new_total = d_i + d
                    old_total = old_total_dist + old_dist_k
                    if new_total < old_total - 1e-6:
                        new_routes = routes[:]
                        if new_route_i:
                            new_routes[i] = new_route_i
                        else:
                            new_routes[i] = []
                        new_routes[k2] = new_route_k
                        return True, new_routes
        
        return False, routes

    def try_remove_station(route):
        """
        Try to remove any station from the route if feasible.
        Returns improved route or None.
        """
        best_route = route
        best_dist = simulate_route(route)[1]
        
        for i in range(len(route)):
            if route[i] in station_set:
                new_route = route[:i] + route[i+1:]
                f, d, _, _ = simulate_route(new_route)
                if f and d < best_dist - 1e-6:
                    best_dist = d
                    best_route = new_route
        return best_route

    def local_search(routes):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. 2-Opt Intra
            for i in range(len(routes)):
                if time.time() > deadline - 5:
                    break
                new_route = two_opt_intra(routes[i])
                if new_route and new_route != routes[i]:
                    routes[i] = new_route
                    improved = True

            # 2. Or-Opt Inter/Intra
            for i in range(len(routes)):
                if time.time() > deadline - 5:
                    break
                route = routes[i]
                for j in range(len(route)):
                    for length in [1, 2, 3]:
                        k = j + length
                        if k > len(route):
                            continue
                        imp, new_routes = or_opt_move(routes, i, j, k)
                        if imp:
                            routes = new_routes
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            # 3. Station Removal
            for i in range(len(routes)):
                if time.time() > deadline - 5:
                    break
                new_route = try_remove_station(routes[i])
                if new_route and new_route != routes[i]:
                    routes[i] = new_route
                    improved = True

            # 4. Clean up empty routes
            new_routes = [r for r in routes if r]
            if len(new_routes) != len(routes):
                routes = new_routes
                improved = True

        return routes

    # LNS (Large Neighborhood Search) inspired by the second program
    def lns_destroy_repair(routes):
        """
        Randomly destroy part of the solution and repair it.
        """
        if len(routes) == 0:
            return routes
        
        # Pick a random route to destroy
        idx = random.randint(0, len(routes) - 1)
        route = routes[idx]
        
        # Remove customers from this route
        customers_to_reinsert = [c for c in route if c in cust_set]
        
        if not customers_to_reinsert:
            return routes
        
        # Remove them from the route
        new_route = [c for c in route if c not in cust_set]
        
        # If the route is now empty, remove it
        if not new_route:
            routes = [r for i, r in enumerate(routes) if i != idx]
        else:
            routes[idx] = new_route
        
        # Re-insert customers one by one
        random.shuffle(customers_to_reinsert)
        
        for c in customers_to_reinsert:
            best_cost = float('inf')
            best_route_idx = -1
            best_pos = -1
            best_new_route = None
            
            for i, r in enumerate(routes):
                if route_demand(r) + demands[c] > capacity:
                    continue
                
                for pos in range(len(r) + 1):
                    cand = r[:pos] + [c] + r[pos:]
                    f, d, _, _ = simulate_route(cand)
                    if f and d < best_cost:
                        best_cost = d
                        best_route_idx = i
                        best_pos = pos
                        best_new_route = cand
            
            if best_route_idx == -1:
                # Couldn't insert, create new route
                r, d = fix_battery([c])
                if r is None:
                    r = [c]
                routes.append(r)
            else:
                routes[best_route_idx] = best_new_route
                
        return routes

    # Initial Construction
    routes = construct_savings()
    
    # Local Search
    routes = local_search(routes)
    
    # LNS Phase
    if time.time() < deadline - 20:
        best_routes = copy.deepcopy(routes)
        best_score = sum(simulate_route(r)[1] for r in best_routes)
        
        for _ in range(100): # Limit LNS iterations
            if time.time() > deadline - 10:
                break
            
            # Destroy and Repair
            candidate = lns_destroy_repair(copy.deepcopy(routes))
            
            # Local Search on candidate
            candidate = local_search(candidate)
            
            # Evaluate
            cand_score = sum(simulate_route(r)[1] for r in candidate)
            
            if cand_score < best_score:
                best_score = cand_score
                best_routes = copy.deepcopy(candidate)
                routes = candidate
            else:
                # Accept with some probability? Or just keep best.
                # For simplicity, keep best.
                pass

    routes = best_routes

    # Final Repair: Ensure all customers are served and feasible
    all_served = set()
    for r in routes:
        all_served.update(c for c in r if c in cust_set)
    
    missing = set(customers) - all_served
    for m in missing:
        inserted = False
        for i in range(len(routes)):
            for pos in range(len(routes[i]) + 1):
                new_route = routes[i][:pos] + [m] + routes[i][pos:]
                if route_demand(new_route) > capacity:
                    continue
                f, _, _, _ = simulate_route(new_route)
                if f:
                    routes[i] = new_route
                    inserted = True
                    break
            if inserted:
                break
        if not inserted:
            # Last resort: add as separate route
            r, d = fix_battery([m])
            if r is None:
                r = [m]
            routes.append(r)

    # Final Feasibility Check & Repair
    for i in range(len(routes)):
        f, _, _, _ = simulate_route(routes[i])
        if not f:
            # Try to fix by inserting stations
            fixed, _ = fix_battery(routes[i])
            if fixed is not None:
                routes[i] = fixed
            else:
                # Split into singletons if still infeasible
                for c in routes[i]:
                    if c in station_set: continue
                    r, d = fix_battery([c])
                    if r is None:
                        r = [c]
                    routes.append(r)
                routes[i] = []

    routes = [r for r in routes if r]
    return routes

# EVOLVE-BLOCK-END

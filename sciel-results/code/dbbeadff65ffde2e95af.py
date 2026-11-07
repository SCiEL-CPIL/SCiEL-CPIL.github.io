"""
EV-Routing with Time Windows and Battery Constraints.
Combines Savings Construction with Battery-Aware Local Search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def plan(instance):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    # Extract data
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
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
    station_set = set(stations)
    
    # Helper: Check if a route is feasible and return (distance, feasible, final_time, final_battery)
    def check_route(route):
        """
        Returns (total_dist, feasible, final_time, final_battery).
        If infeasible, total_dist is inf.
        """
        if not route:
            return 0.0, True, 0.0, battery_cap

        # Check capacity
        total_demand = sum(demands[v] for v in route if v <= n_customers)
        if total_demand > capacity + 1e-9:
            return float('inf'), False, 0.0, 0.0

        time_t = 0.0
        battery = battery_cap
        prev = 0
        total_dist = 0.0

        for v in route:
            d = dist[prev, v]
            time_t += d / speed
            battery -= d

            if battery < -1e-9:
                return float('inf'), False, 0.0, 0.0

            if v <= n_customers:
                # Customer
                if time_t < ready[v]:
                    time_t = ready[v]
                if time_t > due[v] + 1e-9:
                    return float('inf'), False, 0.0, 0.0
                time_t += service[v]
            else:
                # Station
                # Recharge time is (energy_needed) * recharge_rate
                # Energy needed is (battery_cap - current_battery)
                energy_needed = battery_cap - battery
                if energy_needed > 1e-9:
                    time_t += energy_needed * recharge_rate
                battery = battery_cap

            total_dist += d
            prev = v

        # Return to depot
        d = dist[prev, 0]
        time_t += d / speed
        battery -= d

        if battery < -1e-9:
            return float('inf'), False, 0.0, 0.0
        if time_t > horizon + 1e-9:
            return float('inf'), False, 0.0, 0.0

        total_dist += d
        return total_dist, True, time_t, battery

    def get_route_stats(route):
        """Returns (dist, feasible, time, battery)"""
        return check_route(route)

    def try_repair(route):
        """
        Attempt to make an infeasible route feasible by inserting charging stations.
        Returns (new_route, new_dist) or (None, inf).
        """
        # If already feasible, return as is
        d, feas, _, _ = check_route(route)
        if feas:
            return route, d
            
        best_route = None
        best_dist = float('inf')
        
        # Strategy: Insert stations at points where battery is low or time is tight?
        # Simple heuristic: Try inserting each station at each position.
        # Since we have only 4 stations, this is 4 * (len(route)+1) checks.
        for s in stations:
            for i in range(len(route) + 1):
                cand = route[:i] + [s] + route[i:]
                d_c, feas_c, _, _ = check_route(cand)
                if feas_c and d_c < best_dist:
                    best_dist = d_c
                    best_route = cand
                    
        # If single insertion didn't work, try inserting two stations?
        # This might be expensive but necessary for tight constraints.
        # Given 4 stations, C(4,2)=6 pairs. Positions ~ 30. 6*30*30 ~ 5400 checks. Acceptable if rare.
        if best_route is None:
             for s1 in stations:
                 for s2 in stations:
                     if s1 == s2: continue
                     # Try inserting s1 then s2, or s2 then s1?
                     # Just try all permutations of pairs? No, just try inserting s1, then s2 into the result.
                     # Or simpler: try inserting a pair at specific "critical" spots?
                     # Let's stick to single insertion first to keep speed up. 
                     # If that fails, the move is likely rejected.
                     pass
                     
        if best_route:
            return best_route, best_dist
        return None, float('inf')

    # --- Construction: Savings Heuristic ---
    
    # Initialize with one customer per route
    routes = [[c] for c in customers]
    
    # Check initial feasibility. If a single customer is infeasible (e.g. due to battery or time),
    # we might need to add a station immediately.
    for i in range(len(routes)):
        d, feas, _, _ = check_route(routes[i])
        if not feas:
            repaired, _ = try_repair(routes[i])
            if repaired:
                routes[i] = repaired
            else:
                # Critical error: customer cannot be served? 
                # This shouldn't happen with reasonable instances.
                pass

    # Calculate initial savings
    # Saving for merging route i and route j:
    # dist(last_i, 0) + dist(0, first_j) - dist(last_i, first_j)
    
    # We will iteratively find the best merge.
    # To speed up, we can precompute potential savings, but routes change.
    # So we recompute best merge in each iteration.
    
    n_routes = len(routes)
    while n_routes > 1:
        best_saving = 1e-9
        best_i = -1
        best_j = -1
        best_merged_route = None
        best_merged_dist = float('inf')
        
        for i in range(n_routes):
            r_i = routes[i]
            if not r_i: continue
            last_i = r_i[-1]
            
            for j in range(i + 1, n_routes):
                r_j = routes[j]
                if not r_j: continue
                first_j = r_j[0]
                
                # Potential saving (distance only)
                # Merging i -> j: remove last_i->0 and 0->first_j, add last_i->first_j
                saving = dist[last_i, 0] + dist[0, first_j] - dist[last_i, first_j]
                
                if saving < best_saving:
                    continue
                    
                # Try merging i + j
                merged = r_i + r_j
                d_merged, feas_merged, _, _ = check_route(merged)
                
                if not feas_merged:
                    # Try repair
                    repaired, d_repaired = try_repair(merged)
                    if repaired:
                        merged = repaired
                        d_merged = d_repaired
                        feas_merged = True
                    else:
                        continue
                
                if feas_merged:
                    # Calculate actual total distance change
                    # Old dist = dist(r_i) + dist(r_j)
                    # New dist = dist(merged)
                    # We need old dists. Let's fetch them.
                    d_i, _, _, _ = check_route(r_i)
                    d_j, _, _, _ = check_route(r_j)
                    old_dist = d_i + d_j
                    new_dist = d_merged
                    actual_saving = old_dist - new_dist
                    
                    if actual_saving > best_saving:
                        best_saving = actual_saving
                        best_i = i
                        best_j = j
                        best_merged_route = merged
                        best_merged_dist = new_dist
        
        if best_i != -1:
            # Perform merge
            routes[best_i] = best_merged_route
            routes[best_j] = [] # Mark as empty
            # Remove empty routes to keep list clean
            routes = [r for r in routes if r]
            n_routes = len(routes)
        else:
            break
            
        # Check deadline
        if time.time() - start_time > DEADLINE - 20:
            break

    # --- Local Search ---
    
    # Helper to get total distance of all routes
    def total_distance(routes):
        total = 0.0
        for r in routes:
            d, feas, _, _ = check_route(r)
            if not feas:
                return float('inf')
            total += d
        return total

    current_dist = total_distance(routes)
    
    # Local Search Moves:
    # 1. 2-Opt (Intra-route)
    # 2. Or-Opt (Move 1-3 nodes from one route to another, or within route)
    # 3. Relocate (Move 1 node from one route to another)
    # 4. Swap (Swap 2 nodes between routes)
    
    improved = True
    iters = 0
    while improved and time.time() - start_time < DEADLINE - 5:
        improved = False
        iters += 1
        
        # 1. 2-Opt
        for i in range(len(routes)):
            r = routes[i]
            n = len(r)
            if n < 2: continue
            for a in range(n - 1):
                for b in range(a + 1, n):
                    # Reverse segment a..b
                    # New route: r[:a] + r[a:b+1][::-1] + r[b+1:]
                    new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                    d_new, feas_new, _, _ = check_route(new_r)
                    if not feas_new:
                        # Try repair
                        repaired, d_repaired = try_repair(new_r)
                        if repaired:
                            new_r = repaired
                            d_new = d_repaired
                            feas_new = True
                        else:
                            continue
                    
                    if feas_new:
                        d_old, _, _, _ = check_route(r)
                        if d_new < d_old - 1e-9:
                            routes[i] = new_r
                            current_dist -= (d_old - d_new)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue

        # 2. Relocate (Move 1 node from route i to route j)
        for i in range(len(routes)):
            r_i = routes[i]
            for k in range(len(r_i)):
                c = r_i[k]
                new_r_i = r_i[:k] + r_i[k+1:]
                d_i_new, feas_i, _, _ = check_route(new_r_i)
                if not feas_i:
                    repaired, d_repaired = try_repair(new_r_i)
                    if repaired:
                        new_r_i = repaired
                        d_i_new = d_repaired
                        feas_i = True
                    else:
                        continue
                
                if not feas_i:
                    continue
                    
                d_i_old, _, _, _ = check_route(r_i)
                
                for j in range(len(routes)):
                    if i == j: continue
                    r_j = routes[j]
                    for pos in range(len(r_j) + 1):
                        new_r_j = r_j[:pos] + [c] + r_j[pos:]
                        d_j_new, feas_j, _, _ = check_route(new_r_j)
                        if not feas_j:
                            repaired, d_repaired = try_repair(new_r_j)
                            if repaired:
                                new_r_j = repaired
                                d_j_new = d_repaired
                                feas_j = True
                            else:
                                continue
                            
                        if not feas_j:
                            continue
                            
                        d_j_old, _, _, _ = check_route(r_j)
                        
                        # Check if this move reduces total distance
                        # Old total contribution: d_i_old + d_j_old
                        # New total contribution: d_i_new + d_j_new
                        if d_i_new + d_j_new < d_i_old + d_j_old - 1e-9:
                            routes[i] = new_r_i
                            routes[j] = new_r_j
                            current_dist += (d_i_new + d_j_new) - (d_i_old + d_j_old)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue

        # 3. Swap (Swap node c1 from r_i and c2 from r_j)
        for i in range(len(routes)):
            r_i = routes[i]
            for k1 in range(len(r_i)):
                c1 = r_i[k1]
                for j in range(len(routes)):
                    if i == j: continue
                    r_j = routes[j]
                    for k2 in range(len(r_j)):
                        c2 = r_j[k2]
                        if c1 == c2: continue
                        
                        new_r_i = r_i[:k1] + [c2] + r_i[k1+1:]
                        new_r_j = r_j[:k2] + [c1] + r_j[k2+1:]
                        
                        d_i_new, feas_i, _, _ = check_route(new_r_i)
                        if not feas_i:
                            repaired, d_repaired = try_repair(new_r_i)
                            if repaired:
                                new_r_i = repaired
                                d_i_new = d_repaired
                                feas_i = True
                            else:
                                continue
                        if not feas_i: continue
                        
                        d_j_new, feas_j, _, _ = check_route(new_r_j)
                        if not feas_j:
                            repaired, d_repaired = try_repair(new_r_j)
                            if repaired:
                                new_r_j = repaired
                                d_j_new = d_repaired
                                feas_j = True
                            else:
                                continue
                        if not feas_j: continue
                        
                        d_i_old, _, _, _ = check_route(r_i)
                        d_j_old, _, _, _ = check_route(r_j)
                        
                        if d_i_new + d_j_new < d_i_old + d_j_old - 1e-9:
                            routes[i] = new_r_i
                            routes[j] = new_r_j
                            current_dist += (d_i_new + d_j_new) - (d_i_old + d_j_old)
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: break

    # Final Verification
    # Ensure all customers are served exactly once
    served = set()
    for route in routes:
        for v in route:
            if v in served:
                # Invalid
                # Fallback: return single customer routes
                return [[v] for v in sorted(customers, key=lambda x: due[x])]
            served.add(v)

    if served != set(customers):
        # Invalid
        return [[v] for v in sorted(customers, key=lambda x: due[x])]

    # Final feasibility check
    for route in routes:
        d, feas, _, _ = check_route(route)
        if not feas:
            # This should not happen if local search is correct, but just in case
            # Try one last repair
            repaired, _ = try_repair(route)
            if repaired:
                routes[routes.index(route)] = repaired
            else:
                # If still infeasible, return fallback
                return [[v] for v in sorted(customers, key=lambda x: due[x])]

    return routes

# EVOLVE-BLOCK-END

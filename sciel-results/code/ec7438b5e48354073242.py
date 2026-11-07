import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 148.0  # Safety margin for 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Node IDs: 0 depot, 1..n_cust customers, then stations
    n_nodes = n_cust + len(stations) + 1
    cust_ids = list(range(1, n_cust + 1))
    station_ids = set(stations)
    
    # Precompute distance matrix
    cx = coords[:, 0]
    cy = coords[:, 1]
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
            dist[i, j] = d
            dist[j, i] = d

    def is_customer(node):
        return 1 <= node <= n_cust

    def is_station(node):
        return node in station_ids

    def check_route(route):
        """
        Returns (total_dist, valid, final_time, final_battery)
        """
        if not route:
            return 0.0, True, 0.0, battery_max
        
        total_dist = 0.0
        t = 0.0
        b = battery_max
        load = 0.0
        prev = 0
        
        # Check capacity
        for node in route:
            if is_customer(node):
                load += demands[node]
        if load > capacity + _EPS:
            return float('inf'), False, 0.0, 0.0

        for node in route:
            d = dist[prev, node]
            total_dist += d
            t += d / speed
            b -= d
            
            if b < -_EPS:
                return float('inf'), False, 0.0, 0.0
            
            if is_station(node):
                # Recharge
                t += (battery_max - b) * recharge_rate
                b = battery_max
            elif is_customer(node):
                t = max(t, ready[node])
                if t > due[node] + _EPS:
                    return float('inf'), False, 0.0, 0.0
                t += service[node]
            
            prev = node
            
        # Return to depot
        d = dist[prev, 0]
        total_dist += d
        t += d / speed
        b -= d
        
        if b < -_EPS:
            return float('inf'), False, 0.0, 0.0
            
        if t > horizon + _EPS:
            return float('inf'), False, 0.0, 0.0
            
        return total_dist, True, t, b

    def try_insert_station(route, max_stations=2):
        """
        If route is infeasible due to battery, try inserting a station.
        Returns (new_route, new_cost) or (None, inf)
        """
        total_dist, valid, t, b = check_route(route)
        if valid:
            return route, total_dist
        
        best_route = None
        best_cost = float('inf')
        
        # Level 1: Single station insertion
        for s in stations:
            for i in range(len(route) + 1):
                cand = route[:i] + [s] + route[i:]
                c, v, _, _ = check_route(cand)
                if v and c < best_cost:
                    best_cost = c
                    best_route = cand
                    
        if best_route:
            return best_route, best_cost

        if max_stations >= 2:
            # Level 2: Two station insertions
            for s1 in stations:
                for i1 in range(len(route) + 1):
                    cand1 = route[:i1] + [s1] + route[i1:]
                    for s2 in stations:
                        for i2 in range(len(cand1) + 1):
                            cand2 = cand1[:i2] + [s2] + cand1[i2:]
                            c, v, _, _ = check_route(cand2)
                            if v and c < best_cost:
                                best_cost = c
                                best_route = cand2

        if best_route:
            return best_route, best_cost
        return None, float('inf')

    def get_route_cost(route):
        return check_route(route)[0]

    # --- Construction: Greedy Insertion ---
    # 1. Sort customers by due time (tightest first)
    remaining = sorted(cust_ids, key=lambda x: due[x])
    routes = []
    
    while remaining and time.time() < DEADLINE:
        # Start a new route with the first remaining customer
        seed = remaining[0]
        current_route = [seed]
        remaining.remove(seed)
        
        # Calculate initial cost
        cur_cost, cur_valid, _, _ = check_route(current_route)
        if not cur_valid:
             # Single customer should almost always be valid, but just in case
             rep, rcost = try_insert_station(current_route)
             if rep:
                 current_route = rep
                 cur_cost = rcost
             else:
                 # If even single customer is infeasible, something is wrong or instance is infeasible.
                 # But problem states it's solvable.
                 pass

        # Greedily insert remaining customers
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            best_cand = None
            best_pos = -1
            best_cost = float('inf')
            
            # Try to insert each remaining customer at each position
            for c in remaining:
                # Quick capacity check
                load = sum(demands[n] for n in current_route if is_customer(n))
                if load + demands[c] > capacity + _EPS:
                    continue
                
                for pos in range(len(current_route) + 1):
                    cand = current_route[:pos] + [c] + current_route[pos:]
                    
                    # Check feasibility
                    c_cost, v, _, _ = check_route(cand)
                    
                    if not v:
                        # Try to repair with station
                        rep, rep_cost = try_insert_station(cand)
                        if rep:
                            c_cost = rep_cost
                            v = True
                            cand = rep
                        else:
                            continue

                    if v and c_cost < best_cost:
                        best_cost = c_cost
                        best_cand = c
                        best_pos = pos
                        # We store the route with station if it was repaired
                        if c_cost != check_route(current_route[:pos] + [c] + current_route[pos:])[0]:
                             # This check is expensive, let's just store the candidate route
                             # Actually, we need to reconstruct the candidate route if it was repaired.
                             # To save time, we only accept if it's strictly better.
                             # Let's just use the repaired route if it was repaired.
                             pass 
                             # Note: 'cand' variable holds the repaired route if v became True via repair.
                             # But we need to be careful: if v was True initially, cand is the simple insertion.
                             # If v was False then repaired, cand is the repaired insertion.
                             # So 'cand' is always the valid route to use.
                             # However, we are iterating 'c' and 'pos'. We need to remember the actual route.
                             # Let's store the candidate route itself.
                             
            # Refinement: The loop above doesn't store the actual route object if it was repaired.
            # Let's re-do the best candidate selection properly.
            
            # Reset
            best_cand_idx = -1
            best_pos = -1
            best_cost = float('inf')
            best_route_obj = None
            
            for idx, c in enumerate(remaining):
                load = sum(demands[n] for n in current_route if is_customer(n))
                if load + demands[c] > capacity + _EPS:
                    continue
                    
                for pos in range(len(current_route) + 1):
                    simple_cand = current_route[:pos] + [c] + current_route[pos:]
                    c_cost, v, _, _ = check_route(simple_cand)
                    
                    if v:
                        if c_cost < best_cost:
                            best_cost = c_cost
                            best_cand_idx = idx
                            best_pos = pos
                            best_route_obj = simple_cand
                    else:
                        # Try repair
                        rep, rep_cost = try_insert_station(simple_cand)
                        if rep:
                            if rep_cost < best_cost:
                                best_cost = rep_cost
                                best_cand_idx = idx
                                best_pos = pos
                                best_route_obj = rep
                                # Note: best_route_obj includes the station. 
                                # We must remove 'c' from 'remaining' based on best_cand_idx.
            
            if best_cand_idx != -1:
                current_route = best_route_obj
                remaining.pop(best_cand_idx)
                improved = True
        
        routes.append(current_route)

    # --- Local Search ---
    # We have a set of valid routes. Now optimize.
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    while improved and time.time() < DEADLINE:
        improved = False
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            break

        # 1. Intra-route 2-Opt
        for ri in range(len(routes)):
            if time.time() > DEADLINE: break
            route = routes[ri]
            n = len(route)
            if n < 2: continue
            
            cur_cost, cur_valid, _, _ = check_route(route)
            if not cur_valid:
                rep, rcost = try_insert_station(route)
                if rep:
                    routes[ri] = rep
                    cur_cost = rcost
                    cur_valid = True
                else:
                    continue
            
            best_route = list(route)
            best_cost = cur_cost
            route_improved = True
            max_iter = 20 # Allow more iterations for 2-opt
            while route_improved and max_iter > 0 and time.time() < DEADLINE:
                route_improved = False
                max_iter -= 1
                for i in range(n - 1):
                    if time.time() > DEADLINE: break
                    for j in range(i + 1, n):
                        # Standard 2-opt reversal
                        cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                        c, v, _, _ = check_route(cand)
                        
                        if not v:
                             # Try repair? 2-opt with repair is expensive. 
                             # Usually 2-opt is applied to valid routes. 
                             # If a 2-opt move breaks battery, we skip it unless repair is cheap.
                             # For now, skip if invalid.
                             continue

                        if v and c < best_cost - 1e-7:
                            best_cost = c
                            best_route = cand
                            route_improved = True
                            # Break inner loops to restart? 
                            # Standard 2-opt often does best improvement per pass.
                            # Let's continue searching for better in this pass.
            
            if best_cost < cur_cost - 1e-7:
                routes[ri] = best_route
                improved = True
                no_improve_count = 0

        # 2. Inter-route Relocate
        if time.time() > DEADLINE: break
        relocate_improved = False
        for ri in range(len(routes)):
            if time.time() > DEADLINE or relocate_improved: break
            for ci in range(len(routes[ri])):
                if time.time() > DEADLINE or relocate_improved: break
                cust = routes[ri][ci]
                
                # Remove from ri
                temp_from = routes[ri][:]
                del temp_from[ci]
                c_from, v_from, _, _ = check_route(temp_from)
                if not v_from:
                    # If removing a customer makes route invalid? 
                    # Removing a customer reduces load and distance, but might break time windows?
                    # No, removing a customer cannot break time windows or battery (distance decreases).
                    # It might change the sequence, but we are just removing.
                    # Actually, if we remove a customer, the remaining customers are visited in same relative order.
                    # Time at subsequent customers might decrease (arrive earlier), which is fine.
                    # Battery at subsequent customers might increase (less distance), which is fine.
                    # So v_from should always be True if original was valid.
                    pass
                
                for rj in range(len(routes)):
                    if rj == ri: continue
                    if time.time() > DEADLINE or relocate_improved: break
                    for cj in range(len(routes[rj]) + 1):
                        if time.time() > DEADLINE or relocate_improved: break
                        cand_to = routes[rj][:cj] + [cust] + routes[rj][cj:]
                        
                        # Capacity check
                        load_to = sum(demands[n] for n in cand_to if is_customer(n))
                        if load_to > capacity + _EPS:
                            continue

                        c_to, v_to, _, _ = check_route(cand_to)
                        
                        if not v_to:
                            # Try repair
                            rep, rep_cost = try_insert_station(cand_to)
                            if rep:
                                c_to = rep_cost
                                v_to = True
                                cand_to = rep
                            else:
                                continue

                        if v_to:
                            old_cost = c_from + get_route_cost(routes[rj])
                            new_cost = c_from + c_to
                            if new_cost < old_cost - 1e-7:
                                routes[ri] = temp_from
                                routes[rj] = cand_to
                                improved = True
                                relocate_improved = True
                                no_improve_count = 0
                                break
                    if relocate_improved: break
                if relocate_improved: break
            if relocate_improved: break
        
        # Remove empty routes
        routes = [r for r in routes if r]
        
        # 3. Inter-route Swap
        if time.time() > DEADLINE: break
        swapped = False
        for ri in range(len(routes)):
            if swapped or time.time() > DEADLINE: break
            for rj in range(ri + 1, len(routes)):
                if swapped or time.time() > DEADLINE: break
                for ci in range(len(routes[ri])):
                    if swapped or time.time() > DEADLINE: break
                    cust1 = routes[ri][ci]
                    for cj in range(len(routes[rj])):
                        if swapped or time.time() > DEADLINE: break
                        cust2 = routes[rj][cj]
                        
                        # Capacity check
                        load_i = sum(demands[n] for n in routes[ri] if is_customer(n))
                        load_j = sum(demands[n] for n in routes[rj] if is_customer(n))
                        new_load_i = load_i - demands[cust1] + demands[cust2]
                        new_load_j = load_j - demands[cust2] + demands[cust1]
                        
                        if new_load_i > capacity + _EPS or new_load_j > capacity + _EPS:
                            continue
                            
                        new_route_i = routes[ri][:]
                        new_route_i[ci] = cust2
                        new_route_j = routes[rj][:]
                        new_route_j[cj] = cust1
                        
                        c_i, v_i, _, _ = check_route(new_route_i)
                        c_j, v_j, _, _ = check_route(new_route_j)
                        
                        if not v_i or not v_j:
                             # Try repair? Swap with repair is complex. Skip for now.
                             continue

                        if v_i and v_j:
                            old_c_i = get_route_cost(routes[ri])
                            old_c_j = get_route_cost(routes[rj])
                            if old_c_i + old_c_j > c_i + c_j + 1e-7:
                                routes[ri] = new_route_i
                                routes[rj] = new_route_j
                                improved = True
                                swapped = True
                                no_improve_count = 0
                                break
                if swapped: break
            if swapped: break

    # Final validation and fallback
    total_dist = 0.0
    served = set()
    valid = True
    for r in routes:
        c, v, _, _ = check_route(r)
        if not v:
            valid = False
            break
        total_dist += c
        for node in r:
            if node in served:
                valid = False
                break
            served.add(node)
        if not valid:
            break
            
    if not valid or served != set(range(1, n_cust + 1)):
        # Fallback: one customer per route
        customers = list(range(1, n_cust + 1))
        customers.sort(key=lambda v: due[v])
        routes = [[v] for v in customers]
        # Ensure validity of fallback
        for i in range(len(routes)):
            if not check_route(routes[i])[1]:
                rep, _ = try_insert_station(routes[i])
                if rep:
                    routes[i] = rep

    return routes

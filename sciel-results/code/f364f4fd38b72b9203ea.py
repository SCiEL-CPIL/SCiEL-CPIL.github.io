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
    
    # Precompute distance matrix using vectorized NumPy
    cx = coords[:, 0]
    cy = coords[:, 1]
    # dist[i,j] = Euclidean distance
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

        # Level 2: Two station insertions (if max_stations >= 2)
        if max_stations >= 2:
            for s1 in stations:
                for i in range(len(route) + 1):
                    # Insert s1 first
                    temp = route[:i] + [s1] + route[i:]
                    c1, v1, _, _ = check_route(temp)
                    if v1:
                        if c1 < best_cost:
                            best_cost = c1
                            best_route = temp
                        continue # Already valid, no need for second station
                    # If still invalid, try adding s2
                    for s2 in stations:
                        for j in range(len(temp) + 1):
                            cand = temp[:j] + [s2] + temp[j:]
                            c, v, _, _ = check_route(cand)
                            if v and c < best_cost:
                                best_cost = c
                                best_route = cand
            if best_route:
                return best_route, best_cost

        return None, float('inf')

    # --- Construction: Savings Algorithm ---
    
    # Initialize with one customer per route
    routes = [[c] for c in cust_ids]
    # Pre-check single customer routes, insert station if needed
    for i in range(len(routes)):
        if not check_route(routes[i])[1]:
            rep, _ = try_insert_station(routes[i])
            if rep:
                routes[i] = rep
            else:
                # Fallback: should not happen for single customer with battery 300
                pass

    # Calculate savings for merging routes i and j
    # Savings = dist[i_last][0] + dist[0][j_first] - dist[i_last][j_first]
    # But we need to check feasibility.
    
    # We will greedily merge the pair with the highest "potential" saving that is feasible.
    
    active = [True] * len(routes)
    changed = True
    while changed:
        changed = False
        best_saving = 0
        best_i = -1
        best_j = -1
        best_merged = None
        best_cost = float('inf')
        
        for i in range(len(routes)):
            if not active[i]: continue
            for j in range(i + 1, len(routes)):
                if not active[j]: continue
                
                # Try merging i and j in both orders: i->j and j->i
                # Order 1: i followed by j
                merged1 = routes[i] + routes[j]
                c1, v1, _, _ = check_route(merged1)
                if v1:
                    # Calculate saving: cost(i) + cost(j) - cost(merged)
                    ci, _, _, _ = check_route(routes[i])
                    cj, _, _, _ = check_route(routes[j])
                    saving = ci + cj - c1
                    if saving > best_saving:
                        best_saving = saving
                        best_i = i
                        best_j = j
                        best_merged = merged1
                        best_cost = c1
                else:
                    # Try inserting station to make it feasible?
                    # This is expensive. Only do if saving is high?
                    # For now, skip infeasible merges in construction.
                    pass
                
                # Order 2: j followed by i
                merged2 = routes[j] + routes[i]
                c2, v2, _, _ = check_route(merged2)
                if v2:
                    ci, _, _, _ = check_route(routes[i])
                    cj, _, _, _ = check_route(routes[j])
                    saving = ci + cj - c2
                    if saving > best_saving:
                        best_saving = saving
                        best_i = i
                        best_j = j
                        best_merged = merged2
                        best_cost = c2
        
        if best_i != -1:
            # Merge
            routes[best_i] = best_merged
            active[best_j] = False
            changed = True
            
    # Remove inactive routes
    routes = [r for i, r in enumerate(routes) if active[i]]
    
    # --- Local Search ---
    
    # Helper to get route cost
    def get_route_cost(route):
        return check_route(route)[0]
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        
        # 1. Intra-route 2-Opt
        for ri in range(len(routes)):
            if time.time() > DEADLINE: break
            route = routes[ri]
            n = len(route)
            if n < 2: continue
            
            cur_cost, cur_valid, _, _ = check_route(route)
            if not cur_valid:
                # Try to repair first
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
            max_iter = 10
            while route_improved and max_iter > 0:
                route_improved = False
                max_iter -= 1
                for i in range(n - 1):
                    if time.time() > DEADLINE: break
                    for j in range(i + 1, n):
                        cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost - 1e-7:
                            best_cost = c
                            best_route = cand
                            route_improved = True
                # If we improved, we might need to re-check n? No, length is same.
                # But we should restart the i,j loops? The standard 2-opt does first improvement or best improvement.
                # This loop does best improvement within one pass, then repeats.
            
            if best_cost < cur_cost - 1e-7:
                routes[ri] = best_route
                improved = True
        
        # 2. Inter-route Relocate
        if time.time() > DEADLINE: break
        for ri in range(len(routes)):
            if time.time() > DEADLINE: break
            for ci in range(len(routes[ri])):
                if time.time() > DEADLINE: break
                cust = routes[ri][ci]
                # Remove from ri
                temp_from = routes[ri][:]
                del temp_from[ci]
                c_from, v_from, _, _ = check_route(temp_from)
                if not v_from:
                    continue
                
                for rj in range(len(routes)):
                    if rj == ri: continue
                    if time.time() > DEADLINE: break
                    for cj in range(len(routes[rj]) + 1):
                        if time.time() > DEADLINE: break
                        cand_to = routes[rj][:cj] + [cust] + routes[rj][cj:]
                        c_to, v_to, _, _ = check_route(cand_to)
                        if not v_to:
                            continue
                        
                        # Check capacity
                        load_to = sum(demands[n] for n in cand_to if is_customer(n))
                        if load_to > capacity + _EPS:
                            continue
                        
                        old_cost = c_from + get_route_cost(routes[rj])
                        new_cost = c_from + c_to
                        if new_cost < old_cost - 1e-7:
                            routes[ri] = temp_from
                            routes[rj] = cand_to
                            improved = True
                            # Restart relocation search?
                            # Simplest: break out and restart outer while loop
                            ri = -1
                            break
                    if ri == -1: break
                if ri == -1: break
            if ri == -1: break
        
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
                        
                        if v_i and v_j:
                            old_c_i = get_route_cost(routes[ri])
                            old_c_j = get_route_cost(routes[rj])
                            if old_c_i + old_c_j > c_i + c_j + 1e-7:
                                routes[ri] = new_route_i
                                routes[rj] = new_route_j
                                improved = True
                                swapped = True
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

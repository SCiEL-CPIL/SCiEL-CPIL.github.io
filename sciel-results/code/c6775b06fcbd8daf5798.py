"""EV routing with time windows and battery; construction + local search with charging optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def plan(instance):
    t0 = time.time()
    deadline = t0 + 142.0  # Safety margin before 150s kill

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    N = len(coords)
    # Precompute distance matrix
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    cust = list(range(1, n_customers + 1))
    station_set = set(stations)
    
    # Identify customer vs station for quick checks
    is_customer = np.zeros(N, dtype=bool)
    is_customer[1:n_customers+1] = True

    def check_sequence_time_cap(seq):
        """
        Checks if a sequence of customers (and depot 0 at start/end) satisfies
        capacity and time windows, ignoring battery.
        Returns (is_ok, total_time, total_dist).
        seq: list of customer IDs.
        """
        if not seq:
            return True, 0.0, 0.0
        
        # Capacity check
        total_demand = sum(demands[v] for v in seq)
        if total_demand > capacity:
            return False, 0.0, 0.0

        t = 0.0
        d = 0.0
        prev = 0
        for v in seq:
            dd = dist[prev, v]
            t += dd / speed
            d += dd
            # Wait if necessary
            if t < ready[v]:
                t = ready[v]
            # Check due time
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0
            # Service time
            t += service[v]
            prev = v
        
        # Return to depot
        dd = dist[prev, 0]
        t += dd / speed
        d += dd
        
        if t > horizon + 1e-9:
            return False, 0.0, 0.0
            
        return True, t, d

    def optimize_charging(seq, time_limit=1e9):
        """
        Given a sequence of customers (valid for time/capacity), determines if
        it can be made battery-feasible by inserting charging stations.
        Returns (is_feasible, full_route_with_stations, total_dist).
        
        Strategy: 
        We simulate the trip. We track battery. 
        If we cannot reach the next node, we must charge at the current node 
        (if it's a station) or travel to the nearest station that allows us to 
        reach the next node.
        
        However, a more robust "greedy" repair for local search:
        1. Try the direct path. If battery fails, find the earliest point we 
           must charge.
        2. The optimal policy is to charge as little as possible or just enough 
           to reach the next required stop? No, the problem says "refills" 
           (implies full charge) and time is (battery - energy) * rate. 
           Wait, "refills" usually means battery becomes full. 
           Let's re-read: "a station adds (battery - energy) * recharge_rate time and refills".
           This implies battery becomes `battery_cap`.
           
        Algorithm:
        1. Start at depot with full battery.
        2. Iterate through customers.
        3. If distance to next customer > current battery, we MUST stop at a station 
           before reaching that customer.
        4. Which station? The one that minimizes total distance (or time) while 
           ensuring we can reach the next customer.
           Actually, since charging refills to full, the choice of station matters 
           for distance. We should pick the station that is on the way or closest 
           to the path, such that dist(station, next_customer) <= battery_cap.
        5. If no such station exists, the sequence is infeasible.
        """
        if not seq:
            return True, [], 0.0

        route = []
        curr_node = 0
        curr_bat = battery_cap
        total_dist = 0.0
        t = 0.0 # We don't strictly need time here for battery logic, but good for consistency if we wanted to check time again. 
                # The check_sequence_time_cap already verified time windows are met WITHOUT charging delays. 
                # Adding charging delays might violate time windows! 
                # So we MUST re-check time windows after inserting stations.
        
        # To be safe and fast, let's first check if the sequence is time-feasible 
        # *without* any charging delays. If it's not, it's invalid.
        # If it is, we insert charging. The added charging time might push it over 
        # the horizon or due dates. 
        # So `optimize_charging` should return the route, and the caller should 
        # verify the *final* route with stations against time windows.
        
        # Let's modify the approach: 
        # 1. Check Time/Capacity of customer sequence.
        # 2. If ok, try to find a valid charging pattern.
        # 3. If charging pattern exists, check if the *resulting* route (with stations) 
        #    still satisfies time windows.
        
        # Implementation of greedy charging insertion:
        # We traverse the sequence. 
        # If we can't reach the next customer, we look for a station S such that:
        #   dist(curr, S) <= curr_bat
        #   dist(S, next_cust) <= battery_cap (since it refills)
        # We pick the S that minimizes dist(curr, S) + dist(S, next_cust) 
        # or simply the nearest valid S? 
        # Minimizing distance is good for the score.
        
        valid = True
        final_route = []
        
        # We need to track time to ensure we don't violate time windows due to charging.
        # Let's simulate time as well.
        t_curr = 0.0
        
        for i, v in enumerate(seq):
            # Check if we can reach v from curr_node
            dd = dist[curr_node, v]
            if curr_bat < dd - 1e-9:
                # Need to charge
                # Find best station
                best_station = -1
                best_cost = float('inf')
                
                for s in stations:
                    d1 = dist[curr_node, s]
                    if d1 > curr_bat + 1e-9:
                        continue
                    # After charging at s, battery is full.
                    # Can we reach v from s?
                    d2 = dist[s, v]
                    if d2 > battery_cap + 1e-9:
                        continue
                    
                    # Cost metric: distance added.
                    # Original path: curr -> v (infeasible)
                    # New path: curr -> s -> v
                    # Added distance: d1 + d2 - dd (approx, if s is off path)
                    # Actually, we just want to minimize total distance.
                    # Let's just pick the one that minimizes d1 + d2.
                    cost = d1 + d2
                    if cost < best_cost:
                        best_cost = cost
                        best_station = s
                
                if best_station == -1:
                    return False, None, 0.0
                
                # Travel to station
                d1 = dist[curr_node, best_station]
                t_curr += d1 / speed
                curr_bat -= d1
                total_dist += d1
                
                # Check if we arrive too late for station? Stations don't have TW, but we must arrive.
                # No TW for stations, but we must arrive before horizon.
                
                # Charge
                # Time to charge: (battery_cap - curr_bat) * recharge_rate? 
                # No, "adds (battery - energy) * recharge_rate time and refills".
                # Energy used = d1. So remaining battery before charge = curr_bat (which is battery_cap - d1).
                # Wait, curr_bat is the battery *after* traveling d1.
                # So energy needed to refill = battery_cap - curr_bat.
                charge_time = (battery_cap - curr_bat) * recharge_rate
                t_curr += charge_time
                curr_bat = battery_cap
                
                final_route.append(best_station)
                curr_node = best_station
                
                # Now travel to customer v
                dd = dist[curr_node, v]
                if curr_bat < dd - 1e-9:
                    # Should not happen if we checked d2 <= battery_cap
                    return False, None, 0.0
                
                t_curr += dd / speed
                curr_bat -= dd
                total_dist += dd
                
            else:
                # Direct travel is feasible
                t_curr += dd / speed
                curr_bat -= dd
                total_dist += dd

            # Arrive at customer v
            # Time window check
            if t_curr < ready[v]:
                t_curr = ready[v]
            if t_curr > due[v] + 1e-9:
                return False, None, 0.0
            
            t_curr += service[v]
            final_route.append(v)
            curr_node = v
        
        # Return to depot
        dd = dist[curr_node, 0]
        if curr_bat < dd - 1e-9:
            # Need to charge before returning
            best_station = -1
            best_cost = float('inf')
            for s in stations:
                d1 = dist[curr_node, s]
                if d1 > curr_bat + 1e-9:
                    continue
                d2 = dist[s, 0]
                if d2 > battery_cap + 1e-9:
                    continue
                cost = d1 + d2
                if cost < best_cost:
                    best_cost = cost
                    best_station = s
            
            if best_station == -1:
                return False, None, 0.0
            
            # Go to station
            d1 = dist[curr_node, best_station]
            t_curr += d1 / speed
            curr_bat -= d1
            total_dist += d1
            
            # Charge
            charge_time = (battery_cap - curr_bat) * recharge_rate
            t_curr += charge_time
            curr_bat = battery_cap
            
            final_route.append(best_station)
            curr_node = best_station
            
            # Go to depot
            dd = dist[curr_node, 0]
            t_curr += dd / speed
            curr_bat -= dd
            total_dist += dd
        else:
            # Direct return
            t_curr += dd / speed
            curr_bat -= dd
            total_dist += dd
        
        if t_curr > horizon + 1e-9:
            return False, None, 0.0
            
        return True, final_route, total_dist

    def check_full_route(route):
        """
        Verifies a complete route (customers + stations) is valid.
        Returns (ok, total_dist).
        """
        if not route:
            return True, 0.0
        
        # Capacity check (only customers count)
        total_demand = 0
        for v in route:
            if is_customer[v]:
                total_demand += demands[v]
        if total_demand > capacity:
            return False, 0.0

        t = 0.0
        bat = battery_cap
        d = 0.0
        prev = 0
        
        for v in route:
            dd = dist[prev, v]
            if bat < dd - 1e-9:
                return False, 0.0
            bat -= dd
            t += dd / speed
            d += dd
            
            if is_customer[v]:
                if t < ready[v]:
                    t = ready[v]
                if t > due[v] + 1e-9:
                    return False, 0.0
                t += service[v]
            else:
                # Station
                # Charge time
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
            
            prev = v
        
        # Return to depot
        dd = dist[prev, 0]
        if bat < dd - 1e-9:
            return False, 0.0
        bat -= dd
        t += dd / speed
        d += dd
        
        if t > horizon + 1e-9:
            return False, 0.0
            
        return True, d

    # ---------- Construction ----------
    # Use a nearest neighbor / cheapest insertion hybrid
    routes = []
    remaining = list(cust)
    
    # Sort customers by ready time to help with time windows
    remaining.sort(key=lambda x: ready[x])

    while remaining:
        route = []
        # Try to build a route
        while True:
            if time.time() > deadline:
                break
            
            best_v = None
            best_pos = None
            best_dist = float('inf')
            best_route = None
            
            # Try inserting each remaining customer into current route
            for v in remaining:
                for pos in range(len(route) + 1):
                    # Create new customer sequence
                    new_seq = route[:pos] + [v] + route[pos:]
                    
                    # Check time/capacity feasibility of customer sequence
                    ok_tc, _, _ = check_sequence_time_cap(new_seq)
                    if not ok_tc:
                        continue
                    
                    # Try to add charging
                    ok_ch, full_route, d = optimize_charging(new_seq)
                    if not ok_ch:
                        continue
                    
                    # Final check (should pass, but just in case)
                    ok_final, d_final = check_full_route(full_route)
                    if not ok_final:
                        continue
                    
                    if d_final < best_dist:
                        best_dist = d_final
                        best_v = v
                        best_pos = pos
                        best_route = full_route
            
            if best_v is None:
                break
            
            route = best_route
            remaining.remove(best_v)
        
        if route:
            routes.append(route)

    # ---------- Local Search ----------
    def total_dist(routes):
        s = 0.0
        for r in routes:
            ok, d = check_full_route(r)
            if ok:
                s += d
        return s

    best_total = total_dist(routes)
    best_routes = [list(r) for r in routes]

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Relocate: Move one customer from route i to route k
        # We operate on the *customer* sequences, then re-optimize charging.
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri = routes[i]
            # Extract customers from ri
            ri_custs = [v for v in ri if is_customer[v]]
            
            if len(ri_custs) <= 1:
                continue
                
            for j in range(len(ri_custs)):
                v = ri_custs[j]
                # Remove v from ri_custs
                src_custs_without = ri_custs[:j] + ri_custs[j+1:]
                
                # Check if source route is still feasible without v (and re-optimized)
                ok_s, src_route_new, d_s = optimize_charging(src_custs_without)
                if not ok_s:
                    continue
                ok_s_final, d_s_final = check_full_route(src_route_new)
                if not ok_s_final:
                    continue
                
                best_gain = 1e-9
                best_target = None
                best_new_tgt_route = None
                best_new_tgt_custs = None
                
                for k in range(len(routes)):
                    if k == i:
                        continue
                    rk = routes[k]
                    rk_custs = [v for v in rk if is_customer[v]]
                    
                    for pos in range(len(rk_custs) + 1):
                        new_tgt_custs = rk_custs[:pos] + [v] + rk_custs[pos:]
                        
                        # Check target feasibility
                        ok_t, tgt_route_new, d_t = optimize_charging(new_tgt_custs)
                        if not ok_t:
                            continue
                        ok_t_final, d_t_final = check_full_route(tgt_route_new)
                        if not ok_t_final:
                            continue
                        
                        # Calculate gain
                        # Old dist: d_s_final (approx, needs re-calc if we change) + d_t_old
                        # We need current distances. Let's cache them or re-calc.
                        # For simplicity, re-calc current dists if not cached.
                        # To save time, we can approximate gain using d_t_final + d_s_final vs current.
                        # But current dists might change if we accept.
                        # Let's just compute the new total dist for the pair.
                        
                        # We need the distance of the *current* routes i and k.
                        # We don't have them cached. Let's compute them once per iteration of i?
                        # Or just compare d_t_final + d_s_final against a baseline.
                        # Baseline: dist(ri) + dist(rk).
                        
                        # To avoid recalculating dist(ri) and dist(rk) for every candidate,
                        # we can just check if the new combined dist is smaller than the old combined dist.
                        # We need old combined dist.
                        
                        # Let's compute old dists for i and k once before the inner loop?
                        # No, i is fixed, k varies.
                        
                        # Let's just do a full check.
                        # If this is too slow, we can break.
                        
                        # Optimization: Only check if d_t_final + d_s_final < current_best_pair_dist
                        # We don't know current_best_pair_dist without calculating it.
                        
                        # Let's calculate current dists for i and k.
                        # This is expensive. 
                        # Alternative: Store distances in the routes list? 
                        # Let's maintain a list of (route, dist) pairs.
                        
                        # For now, let's just do a simple check:
                        # If d_t_final + d_s_final < (dist(ri) + dist(rk)) - epsilon
                        # We need dist(ri) and dist(rk).
                        
                        # Let's fetch them.
                        _, d_ri = check_full_route(ri)
                        _, d_rk = check_full_route(rk)
                        
                        gain = (d_t_final + d_s_final) - (d_rk + d_ri)
                        if -gain > best_gain:
                            best_gain = -gain
                            best_target = k
                            best_new_tgt_route = tgt_route_new
                            best_new_tgt_custs = new_tgt_custs
                            best_src_route_new = src_route_new
                            best_d_s = d_s_final
                            best_d_t = d_t_final
                            best_d_ri = d_ri
                            best_d_rk = d_rk

                if best_target is not None:
                    routes[i] = best_src_route_new
                    routes[best_target] = best_new_tgt_route
                    if not routes[i]:
                        routes.pop(i)
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue

        # 2. Swap: Exchange customers between two routes
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri = routes[i]
            ri_custs = [v for v in ri if is_customer[v]]
            
            for k in range(i + 1, len(routes)):
                rk = routes[k]
                rk_custs = [v for v in rk if is_customer[v]]
                
                found_swap = False
                for a in range(len(ri_custs)):
                    if found_swap:
                        break
                    for b in range(len(rk_custs)):
                        vi = ri_custs[a]
                        vk = rk_custs[b]
                        
                        new_ri_custs = list(ri_custs)
                        new_rk_custs = list(rk_custs)
                        new_ri_custs[a] = vk
                        new_rk_custs[b] = vi
                        
                        ok_i, new_ri, d_i = optimize_charging(new_ri_custs)
                        ok_k, new_rk, d_k = optimize_charging(new_rk_custs)
                        
                        if not (ok_i and ok_k):
                            continue
                        
                        ok_i_f, d_i_f = check_full_route(new_ri)
                        ok_k_f, d_k_f = check_full_route(new_rk)
                        
                        if not (ok_i_f and ok_k_f):
                            continue
                        
                        _, d_ri = check_full_route(ri)
                        _, d_rk = check_full_route(rk)
                        
                        if d_i_f + d_k_f < d_ri + d_rk - 1e-9:
                            routes[i] = new_ri
                            routes[k] = new_rk
                            improved = True
                            found_swap = True
                            break
                if found_swap:
                    break
            if improved:
                break
        
        if improved:
            continue

        # 3. 2-opt intra route (on customer sequence)
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri = routes[i]
            ri_custs = [v for v in ri if is_customer[v]]
            n = len(ri_custs)
            if n < 2:
                continue
            
            found = False
            for a in range(n - 1):
                if found:
                    break
                for b in range(a + 1, n):
                    new_ri_custs = ri_custs[:a] + ri_custs[a:b+1][::-1] + ri_custs[b+1:]
                    ok, new_ri, d = optimize_charging(new_ri_custs)
                    if not ok:
                        continue
                    ok_f, d_f = check_full_route(new_ri)
                    if not ok_f:
                        continue
                    
                    _, d_ri = check_full_route(ri)
                    
                    if d_f < d_ri - 1e-9:
                        routes[i] = new_ri
                        improved = True
                        found = True
                        break
            if found:
                break
        if improved:
            continue

    cur = total_dist(routes)
    if cur < best_total:
        best_total = cur
        best_routes = [list(r) for r in routes]

    return best_routes
# EVOLVE-BLOCK-END

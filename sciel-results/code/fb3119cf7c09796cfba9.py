import numpy as np
import time

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0
    
    coords = np.array(instance["coords"])
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    D = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            D[i][j] = np.linalg.norm(coords[i] - coords[j])
            
    def check_route(route):
        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        curr_load = 0
        total_dist = 0.0
        
        for next_node in route:
            dist = D[curr_node][next_node]
            curr_time += dist / speed
            curr_battery -= dist
            total_dist += dist
            
            if curr_battery < -1e-9:
                return total_dist, False, curr_time, curr_battery

            if next_node <= n_customers:
                curr_time = max(curr_time, ready[next_node])
                if curr_time > due[next_node] + 1e-9:
                    return total_dist, False, curr_time, curr_battery
                curr_time += service[next_node]
                curr_load += demands[next_node]
                if curr_load > capacity:
                    return total_dist, False, curr_time, curr_battery
            elif next_node in stations:
                energy_added = battery_max - curr_battery
                curr_time += energy_added * recharge_rate
                curr_battery = battery_max
            curr_node = next_node
            
        dist = D[curr_node][0]
        curr_time += dist / speed
        curr_battery -= dist
        total_dist += dist
        
        if curr_battery < -1e-9 or curr_time > horizon + 1e-9:
            return total_dist, False, curr_time, curr_battery
            
        return total_dist, True, curr_time, curr_battery

    def try_fix_battery(route):
        """
        If route is infeasible due to battery, try inserting a station.
        Returns (new_route, feasible, dist) or None.
        """
        # Find first point where battery fails
        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        curr_load = 0
        
        for i, next_node in enumerate(route):
            dist = D[curr_node][next_node]
            curr_time += dist / speed
            curr_battery -= dist
            
            if curr_battery < -1e-9:
                # Try inserting station at i
                for s in stations:
                    new_route = route[:i] + [s] + route[i:]
                    d, f, _, _ = check_route(new_route)
                    if f:
                        return new_route, True, d
                return None
            
            if next_node <= n_customers:
                curr_time = max(curr_time, ready[next_node])
                if curr_time > due[next_node] + 1e-9:
                    return None # Time window violation, station won't help (unless station is before, but we are at i)
                curr_time += service[next_node]
                curr_load += demands[next_node]
                if curr_load > capacity:
                    return None
            elif next_node in stations:
                energy_added = battery_max - curr_battery
                curr_time += energy_added * recharge_rate
                curr_battery = battery_max
            curr_node = next_node
            
        # If we get here, battery didn't fail in the loop, so check return leg
        dist = D[curr_node][0]
        curr_time += dist / speed
        curr_battery -= dist
        
        if curr_battery < -1e-9:
             # Try inserting station at end (before return)
             for s in stations:
                 new_route = route + [s]
                 d, f, _, _ = check_route(new_route)
                 if f:
                     return new_route, True, d
             return None
             
        return None # Should not reach here if check_route said infeasible only due to battery

    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda x: due[x])
    
    routes = []
    unvisited = set(customers)
    
    while unvisited:
        c = min(unvisited, key=lambda x: due[x])
        unvisited.remove(c)
        
        best_candidate = None
        best_idx = -1
        best_cost = float('inf')
        
        for r_idx, route in enumerate(routes):
            for p in range(len(route) + 1):
                cand_route = route[:p] + [c] + route[p:]
                dist, feasible, _, _ = check_route(cand_route)
                if feasible:
                    if dist < best_cost:
                        best_cost = dist
                        best_candidate = cand_route
                        best_idx = r_idx
                else:
                    res = try_fix_battery(cand_route)
                    if res:
                        new_route, _, new_dist = res
                        if new_dist < best_cost:
                            best_cost = new_dist
                            best_candidate = new_route
                            best_idx = r_idx
        
        if best_candidate is not None:
            routes[best_idx] = best_candidate
        else:
            routes.append([c])
            
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Remove unnecessary stations
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            for i in range(len(route)):
                if route[i] in stations:
                    new_route = route[:i] + route[i+1:]
                    dist, feasible, _, _ = check_route(new_route)
                    if feasible:
                        routes[r_idx] = new_route
                        improved = True
                        break
            if improved:
                continue # Re-evaluate this route? Or just move on. Moving on is safer.
                
        # 2-Opt Intra
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            if len(route) < 2: continue
            for i in range(len(route)):
                for j in range(i+1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    dist, feasible, _, _ = check_route(new_route)
                    if feasible:
                        old_dist, _, _, _ = check_route(route)
                        if dist < old_dist - 1e-6:
                            routes[r_idx] = new_route
                            improved = True
                            
        # Relocate Inter
        for r_idx1 in range(len(routes)):
            route1 = routes[r_idx1]
            for i in range(len(route1)):
                c = route1[i]
                for r_idx2 in range(len(routes)):
                    if r_idx1 == r_idx2: continue
                    route2 = routes[r_idx2]
                    for j in range(len(route2) + 1):
                        new_route1 = route1[:i] + route1[i+1:]
                        new_route2 = route2[:j] + [c] + route2[j:]
                        
                        d1, f1, _, _ = check_route(new_route1)
                        d2, f2, _, _ = check_route(new_route2)
                        
                        if f1 and f2:
                            old_d1, _, _, _ = check_route(route1)
                            old_d2, _, _, _ = check_route(route2)
                            if d1 + d2 < old_d1 + old_d2 - 1e-6:
                                routes[r_idx1] = new_route1
                                routes[r_idx2] = new_route2
                                improved = True
                                break
                    if improved: break
                if improved: break
            if improved: break
            
    return routes

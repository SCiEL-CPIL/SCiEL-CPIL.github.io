"""EV-routing solver: improved heuristic with charging station support and aggressive local search."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import heapq
import math


def _euclidean(coords):
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def plan(instance):
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    dist_mat = _euclidean(coords)
    
    # Customer set
    customers = list(range(1, n_customers + 1))
    m = len(customers)
    
    def route_distance(route):
        """Return total distance of route (depot -> ... -> depot)."""
        if not route:
            return 0.0
        d = dist_mat[0, route[0]]
        for i in range(len(route) - 1):
            d += dist_mat[route[i], route[i + 1]]
        d += dist_mat[route[-1], 0]
        return d
    
    def check_feasibility_with_charging(route, max_charge_stops=2):
        """Check if a route is feasible, optionally inserting charging stations.
        Returns (feasible, total_dist, total_time, route_with_stations)."""
        if not route:
            return True, 0.0, 0.0, []
        
        total_demand = sum(demands[c] for c in route)
        if total_demand > capacity + 1e-9:
            return False, 0.0, 0.0, []
        
        # Try without charging first
        t = 0.0
        bat = float(battery_max)
        total_d = 0.0
        prev = 0
        for c in route:
            d = dist_mat[prev, c]
            t += d / speed
            bat -= d
            if bat < -1e-9:
                break
            if t < ready[c]:
                t = ready[c]
            if t > due[c] + 1e-9:
                return False, 0.0, 0.0, []
            t += service[c]
            total_d += d
            prev = c
        
        if bat >= -1e-9:
            d = dist_mat[prev, 0]
            t += d / speed
            bat -= d
            if bat >= -1e-9 and t <= horizon + 1e-9:
                return True, total_d + d, t, route
        
        # Try inserting one charging station
        if max_charge_stops >= 1:
            best_insert = None
            for s in stations:
                # Try inserting s after each customer
                for i in range(len(route)):
                    # Check if we can reach s from route[i]
                    d_to_s = dist_mat[route[i], s]
                    if bat_after_prev < d_to_s:
                        continue
                    
                    # Simulate with charging at s
                    t2 = 0.0
                    bat2 = float(battery_max)
                    total_d2 = 0.0
                    prev2 = 0
                    feasible2 = True
                    route2 = []
                    
                    for idx, c in enumerate(route):
                        d = dist_mat[prev2, c]
                        t2 += d / speed
                        bat2 -= d
                        if bat2 < -1e-9:
                            feasible2 = False
                            break
                        if t2 < ready[c]:
                            t2 = ready[c]
                        if t2 > due[c] + 1e-9:
                            feasible2 = False
                            break
                        t2 += service[c]
                        total_d2 += d
                        route2.append(c)
                        prev2 = c
                        
                        # Insert charging station after this customer
                        if idx == i:
                            d_cs = dist_mat[prev2, s]
                            t2 += d_cs / speed
                            bat2 -= d_cs
                            if bat2 < -1e-9:
                                feasible2 = False
                                break
                            # Charging time
                            charge_time = (battery_max - bat2) * recharge_rate
                            t2 += charge_time
                            bat2 = battery_max
                            total_d2 += d_cs
                            route2.append(s)
                            prev2 = s
                    
                    if feasible2:
                        d_back = dist_mat[prev2, 0]
                        t2 += d_back / speed
                        bat2 -= d_back
                        if bat2 >= -1e-9 and t2 <= horizon + 1e-9:
                            total_d2 += d_back
                            route2.append(0)  # depot
                            if best_insert is None or total_d2 < best_insert[1]:
                                best_insert = (route2[:-1], total_d2, t2)
                
                # Also try inserting before first customer
                d_to_s = dist_mat[0, s]
                if d_to_s <= battery_max:
                    t2 = 0.0
                    bat2 = battery_max - d_to_s
                    charge_time = (battery_max - bat2) * recharge_rate
                    t2 += d_to_s / speed + charge_time
                    bat2 = battery_max
                    total_d2 = d_to_s
                    prev2 = s
                    feasible2 = True
                    route2 = [s]
                    
                    for c in route:
                        d = dist_mat[prev2, c]
                        t2 += d / speed
                        bat2 -= d
                        if bat2 < -1e-9:
                            feasible2 = False
                            break
                        if t2 < ready[c]:
                            t2 = ready[c]
                        if t2 > due[c] + 1e-9:
                            feasible2 = False
                            break
                        t2 += service[c]
                        total_d2 += d
                        route2.append(c)
                        prev2 = c
                    
                    if feasible2:
                        d_back = dist_mat[prev2, 0]
                        t2 += d_back / speed
                        bat2 -= d_back
                        if bat2 >= -1e-9 and t2 <= horizon + 1e-9:
                            total_d2 += d_back
                            if best_insert is None or total_d2 < best_insert[1]:
                                best_insert = (route2, total_d2, t2)
            
            if best_insert is not None:
                return True, best_insert[1], best_insert[2], best_insert[0]
        
        return False, 0.0, 0.0, []
    
    def check_feasibility(route):
        """Check if a route is feasible without charging."""
        if not route:
            return True, 0.0, 0.0
        
        t = 0.0
        bat = float(battery_max)
        total_d = 0.0
        total_demand = 0.0
        
        for c in route:
            total_demand += demands[c]
        if total_demand > capacity + 1e-9:
            return False, 0.0, 0.0
        
        prev = 0
        for c in route:
            d = dist_mat[prev, c]
            t += d / speed
            bat -= d
            if bat < -1e-9:
                return False, 0.0, 0.0
            if t < ready[c]:
                t = ready[c]
            if t > due[c] + 1e-9:
                return False, 0.0, 0.0
            t += service[c]
            total_d += d
            prev = c
        
        d = dist_mat[prev, 0]
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return False, 0.0, 0.0
        if t > horizon + 1e-9:
            return False, 0.0, 0.0
        
        return True, total_d + d, t
    
    # Sort customers by due date
    customers_sorted = sorted(customers, key=lambda c: due[c])
    
    routes = []
    
    for c in customers_sorted:
        best_route_idx = -1
        best_pos = -1
        best_dist = float('inf')
        
        for r_idx, route in enumerate(routes):
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                feasible, total_d, _ = check_feasibility(new_route)
                if feasible and total_d < best_dist:
                    best_dist = total_d
                    best_route_idx = r_idx
                    best_pos = pos
        
        if best_route_idx >= 0:
            routes[best_route_idx] = routes[best_route_idx][:best_pos] + [c] + routes[best_route_idx][best_pos:]
        else:
            feasible, _, _ = check_feasibility([c])
            if feasible:
                routes.append([c])
            else:
                # Try with charging
                feasible, total_d, _, route_with_charging = check_feasibility_with_charging([c])
                if feasible:
                    routes.append(route_with_charging)
                else:
                    routes.append([c])
    
    # Aggressive local search: 2-opt + Or-opt
    def two_opt(route):
        improved = True
        while improved:
            improved = False
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if new_route == route:
                        continue
                    old_d = route_distance(route)
                    new_d = route_distance(new_route)
                    if new_d < old_d - 1e-9:
                        feasible, _, _ = check_feasibility(new_route)
                        if feasible:
                            route = new_route
                            improved = True
        return route
    
    def or_opt(route):
        """Or-opt: move 1-3 consecutive customers to another position."""
        improved = True
        while improved:
            improved = False
            n = len(route)
            for length in [1, 2, 3]:
                for i in range(n - length + 1):
                    segment = route[i:i+length]
                    for j in range(n - length + 1):
                        if j == i:
                            continue
                        # Remove segment and insert at j
                        remaining = route[:i] + route[i+length:]
                        if j > i:
                            j -= length
                        new_route = remaining[:j] + segment + remaining[j:]
                        if new_route == route:
                            continue
                        old_d = route_distance(route)
                        new_d = route_distance(new_route)
                        if new_d < old_d - 1e-9:
                            feasible, _, _ = check_feasibility(new_route)
                            if feasible:
                                route = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return route
    
    # Apply local search multiple times
    for iteration in range(3):
        for r_idx in range(len(routes)):
            routes[r_idx] = two_opt(routes[r_idx])
            routes[r_idx] = or_opt(routes[r_idx])
    
    # Try to merge routes if possible
    merged = True
    while merged:
        merged = False
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                # Try all combinations of merging route i and route j
                best_merge = None
                best_merge_dist = float('inf')
                
                # Try inserting each customer from route j into route i
                for c_idx, c in enumerate(routes[j]):
                    for pos in range(len(routes[i]) + 1):
                        new_route_i = routes[i][:pos] + [c] + routes[i][pos:]
                        remaining_j = routes[j][:c_idx] + routes[j][c_idx+1:]
                        
                        # Check if new_route_i is feasible
                        feasible_i, dist_i, _ = check_feasibility(new_route_i)
                        if not feasible_i:
                            continue
                        
                        # Check if remaining_j is feasible (or empty)
                        if remaining_j:
                            feasible_j, dist_j, _ = check_feasibility(remaining_j)
                            if not feasible_j:
                                continue
                            total_dist = dist_i + dist_j
                        else:
                            total_dist = dist_i
                        
                        if total_dist < best_merge_dist:
                            best_merge_dist = total_dist
                            best_merge = (new_route_i, remaining_j)
                
                if best_merge is not None:
                    new_i, new_j = best_merge
                    if new_j:
                        routes[i] = new_i
                        routes[j] = new_j
                    else:
                        routes[i] = new_i
                        routes.pop(j)
                    merged = True
                    break
            if merged:
                break
    
    # Final local search after merging
    for r_idx in range(len(routes)):
        routes[r_idx] = two_opt(routes[r_idx])
        routes[r_idx] = or_opt(routes[r_idx])
    
    return routes
# EVOLVE-BLOCK-END

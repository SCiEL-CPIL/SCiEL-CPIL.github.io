# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def _compute_dist(coords):
    n = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def _expand_route(customer_route, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations):
    if not customer_route:
        return [], 0.0
    
    route = []
    current_node = 0
    bat = battery_cap
    total_time = 0.0
    total_dist = 0.0
    
    for c in customer_route:
        while True:
            d = dist[current_node, c]
            if bat >= d:
                break
            best_s = None
            best_d_s = float('inf')
            for s in stations:
                d_to_s = dist[current_node, s]
                if bat >= d_to_s and dist[s, c] <= battery_cap:
                    if d_to_s < best_d_s:
                        best_d_s = d_to_s
                        best_s = s
            if best_s is None:
                return None, 0.0
            d_to_s = dist[current_node, best_s]
            total_time += d_to_s / speed
            bat -= d_to_s
            total_dist += d_to_s
            total_time += (battery_cap - bat) * recharge_rate
            bat = battery_cap
            route.append(best_s)
            current_node = best_s
        
        d = dist[current_node, c]
        total_time += d / speed
        bat -= d
        total_dist += d
        start_service = max(total_time, ready[c])
        if start_service > due[c]:
            return None, 0.0
        total_time = start_service + service[c]
        route.append(c)
        current_node = c
    
    # Return to depot
    while True:
        d = dist[current_node, 0]
        if bat >= d:
            break
        best_s = None
        best_d_s = float('inf')
        for s in stations:
            d_to_s = dist[current_node, s]
            if bat >= d_to_s and dist[s, 0] <= battery_cap:
                if d_to_s < best_d_s:
                    best_d_s = d_to_s
                    best_s = s
        if best_s is None:
            return None, 0.0
        d_to_s = dist[current_node, best_s]
        total_time += d_to_s / speed
        bat -= d_to_s
        total_dist += d_to_s
        total_time += (battery_cap - bat) * recharge_rate
        bat = battery_cap
        route.append(best_s)
        current_node = best_s
    
    d = dist[current_node, 0]
    total_time += d / speed
    bat -= d
    total_dist += d
    
    if total_time > horizon:
        return None, 0.0
    
    return route, total_dist

def _solve_instance(instance, deadline):
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    dist = _compute_dist(coords)
    station_set = set(stations)
    customers = list(range(1, n_customers + 1))
    
    def try_expand(cust_seq):
        return _expand_route(cust_seq, dist, ready, due, service, battery, speed,
                             recharge_rate, horizon, stations)
    
    def extract_customers(route):
        return [n for n in route if n not in station_set and n != 0]
    
    # Precompute: for each node, the nearest station reachable with full battery
    # and the cost to go via that station
    def can_reach_direct(node, target, bat):
        return bat >= dist[node, target]
    
    def get_station_detour(node, target, bat):
        """Return (extra_dist, station) or (inf, None) if no station works."""
        best_s = None
        best_extra = float('inf')
        for s in stations:
            d1 = dist[node, s]
            d2 = dist[s, target]
            if bat >= d1 and d2 <= battery:
                extra = d1 + d2 - dist[node, target]
                if extra < best_extra:
                    best_extra = extra
                    best_s = s
        return best_extra, best_s
    
    # Phase 1: Savings-based construction
    # Sort customers by due time for initial ordering
    cust_sorted = sorted(customers, key=lambda c: due[c])
    
    # Create initial single-customer routes
    initial_routes = []
    for c in cust_sorted:
        exp, d = try_expand([c])
        if exp is not None:
            initial_routes.append([c])
    
    # Compute savings: s(i,j) = dist(0,i) + dist(0,j) - dist(i,j)
    # We merge routes when savings are positive and feasibility holds
    n_routes = len(initial_routes)
    route_nodes = [[c] for c in cust_sorted]
    
    # Precompute savings between all pairs
    savings = np.zeros((n_routes, n_routes))
    for i in range(n_routes):
        for j in range(i+1, n_routes):
            a = route_nodes[i][0]
            b = route_nodes[j][0]
            savings[i, j] = dist[0, a] + dist[0, b] - dist[a, b]
    
    # Merge routes greedily by savings
    changed = True
    while changed and time.time() < deadline - 30:
        changed = False
        best_sav = 0
        best_i, best_j = -1, -1
        for i in range(len(route_nodes)):
            for j in range(i+1, len(route_nodes)):
                if savings[i, j] > best_sav:
                    best_sav = savings[i, j]
                    best_i, best_j = i, j
        
        if best_i == -1:
            break
        
        # Try to merge route best_j into route best_i (append or prepend)
        merged = None
        # Try appending j to end of i
        combined = route_nodes[best_i] + route_nodes[best_j]
        exp, _ = try_expand(combined)
        if exp is not None and sum(demands[c] for c in combined) <= capacity:
            merged = combined
        # Try prepending j to start of i
        if merged is None:
            combined = route_nodes[best_j] + route_nodes[best_i]
            exp, _ = try_expand(combined)
            if exp is not None and sum(demands[c] for c in combined) <= capacity:
                merged = combined
        
        if merged is not None:
            route_nodes[best_i] = merged
            route_nodes.pop(best_j)
            # Update savings
            n_routes = len(route_nodes)
            new_savings = np.zeros((n_routes, n_routes))
            for i in range(n_routes):
                for j in range(i+1, n_routes):
                    a = route_nodes[i][-1]  # last customer
                    b = route_nodes[j][0]   # first customer
                    # Approximate savings using endpoints
                    new_savings[i, j] = dist[0, route_nodes[i][0]] + dist[0, route_nodes[j][0]] - dist[route_nodes[i][-1], route_nodes[j][0]]
            savings = new_savings
            changed = True
    
    # Build final routes from construction
    routes = []
    for cust_seq in route_nodes:
        if cust_seq:
            exp, _ = try_expand(cust_seq)
            if exp is not None:
                routes.append(exp)
            else:
                # Split into singles
                for c in cust_seq:
                    e, _ = try_expand([c])
                    if e is not None:
                        routes.append(e)
    
    # Phase 2: Local search - 2-opt and or-opt with fast delta evaluation
    def route_distance(cust_seq):
        """Fast distance estimate without full expansion (just customer-to-customer + depot legs)."""
        if not cust_seq:
            return 0.0
        d = dist[0, cust_seq[0]]
        for k in range(len(cust_seq) - 1):
            d += dist[cust_seq[k], cust_seq[k+1]]
        d += dist[cust_seq[-1], 0]
        return d
    
    def local_search(cust_seq, max_iters=20):
        if len(cust_seq) < 2:
            return cust_seq
        
        best = list(cust_seq)
        best_exp, best_full_dist = try_expand(best)
        if best_exp is None:
            return cust_seq
        best_est = route_distance(best)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break
            improved = False
            
            # 2-opt
            n = len(best)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_seq = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    new_est = route_distance(new_seq)
                    if new_est < best_est - 0.5:
                        exp, fd = try_expand(new_seq)
                        if exp is not None and fd < best_full_dist - 1e-9:
                            best = new_seq
                            best_est = new_est
                            best_exp = exp
                            best_full_dist = fd
                            improved = True
            
            # Or-opt (move 1 or 2 customers)
            for seg_len in [1, 2]:
                for i in range(len(best) - seg_len):
                    seg = best[i:i+seg_len]
                    rest = best[:i] + best[i+seg_len:]
                    for j in range(len(rest) + 1):
                        new_seq = rest[:j] + seg + rest[j:]
                        if new_seq == best:
                            continue
                        new_est = route_distance(new_seq)
                        if new_est < best_est - 0.5:
                            exp, fd = try_expand(new_seq)
                            if exp is not None and fd < best_full_dist - 1e-9:
                                best = new_seq
                                best_est = new_est
                                best_exp = exp
                                best_full_dist = fd
                                improved = True
                                break
                if improved:
                    break
            if not improved:
                break
        
        return best
    
    if time.time() < deadline - 15:
        optimized = []
        for route in routes:
            cust_seq = extract_customers(route)
            opt_seq = local_search(cust_seq)
            exp, _ = try_expand(opt_seq)
            if exp is not None:
                optimized.append(exp)
            else:
                optimized.append(route)
        routes = optimized
    
    # Phase 3: Inter-route moves
    if time.time() < deadline - 10 and len(routes) > 1:
        improved = True
        max_swap_iters = 3
        for _ in range(max_swap_iters):
            if not improved:
                break
            improved = False
            for i in range(len(routes)):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, len(routes)):
                    if time.time() > deadline - 10:
                        break
                    cust_i = extract_customers(routes[i])
                    cust_j = extract_customers(routes[j])
                    if not cust_i or not cust_j:
                        continue
                    
                    best_move = None
                    best_gain = 0
                    
                    # Try moving one customer from i to j
                    for c in cust_i:
                        new_i = [x for x in cust_i if x != c]
                        new_j = cust_j + [c]
                        if sum(demands[x] for x in new_i) > capacity or sum(demands[x] for x in new_j) > capacity:
                            continue
                        exp_i, di = try_expand(new_i)
                        exp_j, dj = try_expand(new_j)
                        if exp_i is None or exp_j is None:
                            continue
                        _, cur_di = try_expand(cust_i)
                        _, cur_dj = try_expand(cust_j)
                        gain = (cur_di + cur_dj) - (di + dj)
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('move', i, j, c, exp_i, exp_j)
                    
                    # Try swapping
                    for c_i in cust_i:
                        for c_j in cust_j:
                            new_i = [x for x in cust_i if x != c_i] + [c_j]
                            new_j = [x for x in cust_j if x != c_j] + [c_i]
                            if sum(demands[x] for x in new_i) > capacity or sum(demands[x] for x in new_j) > capacity:
                                continue
                            exp_i, di = try_expand(new_i)
                            exp_j, dj = try_expand(new_j)
                            if exp_i is None or exp_j is None:
                                continue
                            _, cur_di = try_expand(cust_i)
                            _, cur_dj = try_expand(cust_j)
                            gain = (cur_di + cur_dj) - (di + dj)
                            if gain > best_gain:
                                best_gain = gain
                                best_move = ('swap', i, j, c_i, c_j, exp_i, exp_j)
                    
                    if best_move is not None:
                        if best_move[0] == 'move':
                            routes[best_move[2]] = best_move[4]
                            routes[best_move[3]] = best_move[5]
                        else:
                            routes[best_move[2]] = best_move[5]
                            routes[best_move[3]] = best_move[6]
                        improved = True
    
    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140
    routes = _solve_instance(instance, deadline)
    return routes
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _evaluate_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """
    Evaluate a single route.
    Returns (distance, penalty, is_feasible).
    """
    if not route:
        return 0.0, 0.0, True
    
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    penalty = 0.0
    is_feasible = True
    
    if total_demand > capacity:
        is_feasible = False
        penalty += (total_demand - capacity) * 1000.0
        
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0  # depot
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        time_t += d / speed
        total_dist += d
        
        if battery < -1e-9:
            is_feasible = False
            penalty += abs(battery) * 1000.0
            
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v]:
                is_feasible = False
                penalty += (time_t - due[v]) * 100.0
            time_t += service[v]
        else:
            # Charging station
            if battery < battery_max:
                energy_needed = battery_max - battery
                time_t += energy_needed * recharge_rate
                battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    time_t += d / speed
    total_dist += d
    
    if battery < -1e-9:
        is_feasible = False
        penalty += abs(battery) * 1000.0
    if time_t > horizon:
        is_feasible = False
        penalty += (time_t - horizon) * 100.0
        
    return total_dist, penalty, is_feasible


def _evaluate_solution(routes, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """
    Evaluate the entire solution (list of routes).
    Returns (total_distance, total_penalty, is_feasible).
    """
    total_dist = 0.0
    total_penalty = 0.0
    is_feasible = True
    
    for route in routes:
        d, p, f = _evaluate_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        total_dist += d
        total_penalty += p
        if not f:
            is_feasible = False
            
    return total_dist, total_penalty, is_feasible


def _construct_initial_routes(instance, dist):
    """
    Construct initial routes using a Clarke-Wright savings heuristic,
    ensuring feasibility via _evaluate_route.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    # Start with each customer as its own route
    routes = [[i] for i in range(1, n_customers + 1)]
    
    # Calculate savings
    savings = []
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            ci = i + 1
            cj = j + 1
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            savings.append((s, ci, cj))
    
    # Sort by savings descending
    savings.sort(reverse=True)
    
    # Track which route each customer belongs to
    route_of = {}
    for idx, r in enumerate(routes):
        for c in r:
            route_of[c] = idx
    
    # Try to merge routes
    for s, ci, cj in savings:
        ri = route_of.get(ci, -1)
        rj = route_of.get(cj, -1)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        
        # Try both orderings
        best_new_route = None
        best_dist = float('inf')
        
        for order in [[routes[ri], routes[rj]], [routes[rj], routes[ri]]]:
            merged = order[0] + order[1]
            # Check feasibility
            d, p, f = _evaluate_route(merged, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
            if f and d < best_dist:
                best_dist = d
                best_new_route = merged
        
        if best_new_route is not None:
            routes[ri] = best_new_route
            routes[rj] = []
            for c in best_new_route:
                route_of[c] = ri
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    return routes


def _two_opt_improve(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply 2-opt improvements to a route."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 5
    
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        for i in range(n - 1):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return route
                prev_i = route[i-1] if i > 0 else 0
                next_j = route[j+1] if j < n - 1 else 0
                
                d_old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[i]][next_j]
                
                if d_new < d_old - 1e-9:
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d, p, f = _evaluate_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if f:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    
    return route


def _or_opt_improve(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply Or-Opt improvements (move 1, 2, or 3 consecutive nodes)."""
    n = len(route)
    if n < 2:
        return route
    
    improved = True
    max_passes = 3
    while improved and max_passes > 0:
        improved = False
        max_passes -= 1
        
        for L in [1, 2, 3]:
            if n < L + 1: continue
            for i in range(n - L + 1):
                if time.time() > deadline:
                    return route
                # Nodes to move
                moving = route[i:i+L]
                # Remaining route
                rest = route[:i] + route[i+L:]
                
                # Try inserting at every position in rest
                for j in range(len(rest) + 1):
                    if j == i or j == i+L:
                        continue
                    new_route = rest[:j] + moving + rest[j:]
                    # Check if it's the same as original
                    if new_route == route:
                        continue
                    
                    d_new, p_new, f_new = _evaluate_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                    if f_new:
                        # Calculate distance change to accept only if better
                        d_old, p_old, f_old = _evaluate_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                        if d_new < d_old - 1e-9:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return route


def _solve_instance(instance, time_limit=40):
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _dist_matrix(coords)
    
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]
    
    # Initial Solution using Savings
    routes = _construct_initial_routes(instance, dist)
    
    # Ensure all customers are covered.
    covered = set()
    for r in routes:
        covered.update(r)
    missing = [c for c in range(1, n_customers + 1) if c not in covered]
    for c in missing:
        routes.append([c])
        
    # Apply local search improvements initially
    for i in range(len(routes)):
        routes[i] = _two_opt_improve(routes[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline)
        routes[i] = _or_opt_improve(routes[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon, deadline)
        
    # Parallel Tempering Setup
    n_replicas = 3
    temperatures = [100.0, 50.0, 10.0] # Hot, Warm, Cold
    replicas = [list(routes) for _ in range(n_replicas)]
    replica_scores = []
    replica_penalties = []
    replica_feasible = []
    
    for i in range(n_replicas):
        d, p, f = _evaluate_solution(replicas[i], dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        score = d + p
        replica_scores.append(score)
        replica_penalties.append(p)
        replica_feasible.append(f)
        
    best_routes = None
    best_score = float('inf')
    best_feasible_score = float('inf')
    
    # Find initial best feasible
    for i in range(n_replicas):
        if replica_feasible[i]:
            if replica_scores[i] < best_feasible_score:
                best_feasible_score = replica_scores[i]
                best_routes = [r[:] for r in replicas[i]]
                
    if best_routes is None:
        # If no feasible solution found, pick the one with lowest penalty
        min_pen_idx = min(range(n_replicas), key=lambda i: replica_penalties[i])
        best_routes = [r[:] for r in replicas[min_pen_idx]]
        best_feasible_score = replica_scores[min_pen_idx]

    # SA Loop
    step = 0
    while time.time() < deadline - 2:
        # Pick a random replica to move
        r_idx = random.randint(0, n_replicas - 1)
        old_routes = replicas[r_idx]
        old_score = replica_scores[r_idx]
        
        # Generate a move
        move_type = random.random()
        
        new_routes = [r[:] for r in old_routes]
        
        # Helper to find route index of a customer
        def get_route_index(routes, c):
            for idx, r in enumerate(routes):
                if c in r:
                    return idx
            return -1
            
        # Helper to get position of customer in route
        def get_pos(route, c):
            return route.index(c)
            
        if move_type < 0.4:
            # Relocate: Move a customer from one route to another
            all_customers = []
            for r in new_routes:
                all_customers.extend(r)
            
            if not all_customers:
                continue
                
            c = random.choice(all_customers)
            src_r_idx = get_route_index(new_routes, c)
            if src_r_idx == -1:
                continue
                
            src_route = new_routes[src_r_idx]
            pos_c = get_pos(src_route, c)
            
            # Remove from source
            src_route.pop(pos_c)
            
            # Pick destination route
            dst_r_idx = random.randint(0, len(new_routes) - 1)
            if dst_r_idx == src_r_idx:
                # Move within same route
                if len(src_route) < 1:
                    src_route.insert(pos_c, c)
                    continue
                dst_route = src_route
                ins_pos = random.randint(0, len(dst_route))
                dst_route.insert(ins_pos, c)
            else:
                dst_route = new_routes[dst_r_idx]
                ins_pos = random.randint(0, len(dst_route))
                dst_route.insert(ins_pos, c)
                
            # Remove empty routes
            new_routes = [r for r in new_routes if r]
            
        elif move_type < 0.7:
            # Swap: Exchange two customers from different routes
            if len(new_routes) < 2:
                continue
            r1, r2 = random.sample(range(len(new_routes)), 2)
            route1 = new_routes[r1]
            route2 = new_routes[r2]
            
            if not route1 or not route2:
                continue
                
            c1 = random.choice(route1)
            c2 = random.choice(route2)
            
            pos1 = route1.index(c1)
            pos2 = route2.index(c2)
            
            route1[pos1] = c2
            route2[pos2] = c1
            
        else:
            # 2-Opt within a route
            if not new_routes:
                continue
            r_idx_move = random.randint(0, len(new_routes) - 1)
            route = new_routes[r_idx_move]
            if len(route) < 2:
                continue
            
            i = random.randint(0, len(route) - 2)
            j = random.randint(i + 1, len(route) - 1)
            
            route[i:j+1] = route[i:j+1][::-1]
            
        # Evaluate new solution
        new_dist, new_pen, new_feas = _evaluate_solution(new_routes, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        new_score = new_dist + new_pen
        
        # Acceptance Criteria
        delta = new_score - old_score
        T = temperatures[r_idx]
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            replicas[r_idx] = new_routes
            replica_scores[r_idx] = new_score
            replica_penalties[r_idx] = new_pen
            replica_feasible[r_idx] = new_feas
            
            # Update best feasible
            if new_feas:
                if new_score < best_feasible_score:
                    best_feasible_score = new_score
                    best_routes = [r[:] for r in new_routes]
            
            # Parallel Tempering Swap
            if r_idx > 0:
                swap_idx = r_idx - 1
                if random.random() < 0.1:
                    delta_swap = replica_scores[r_idx] - replica_scores[swap_idx]
                    if delta_swap < 0:
                        pass
                    else:
                        prob = math.exp(-delta_swap / temperatures[r_idx])
                        if random.random() < prob:
                            replicas[r_idx], replicas[swap_idx] = replicas[swap_idx], replicas[r_idx]
                            replica_scores[r_idx], replica_scores[swap_idx] = replica_scores[swap_idx], replica_scores[r_idx]
                            replica_penalties[r_idx], replica_penalties[swap_idx] = replica_penalties[swap_idx], replica_penalties[r_idx]
                            replica_feasible[r_idx], replica_feasible[swap_idx] = replica_feasible[swap_idx], replica_feasible[r_idx]
            
        step += 1
        
    return best_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    
    # 3 instances, 150s total, leave 10s buffer -> 40s per instance
    per_instance_time = 40
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END

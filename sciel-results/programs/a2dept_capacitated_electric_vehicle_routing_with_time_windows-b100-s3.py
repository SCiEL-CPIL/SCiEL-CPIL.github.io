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
    Penalty includes time window violations, battery violations, capacity violations.
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
            # If battery is negative, subsequent calculations might be invalid, 
            # but we continue to accumulate penalty. 
            # To avoid weirdness, we can assume we 'charge' enough to continue? 
            # No, better to just accumulate penalty and break or continue with 0 battery?
            # Let's just accumulate penalty.
            
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
    Construct initial routes using a simple greedy approach.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    # Sort customers by angle from depot
    depot_x, depot_y = coords[0]
    customers = list(range(1, n_customers + 1))
    angles = []
    for c in customers:
        dx = coords[c][0] - depot_x
        dy = coords[c][1] - depot_y
        angle = math.atan2(dy, dx)
        angles.append((angle, c))
    angles.sort()
    
    routes = []
    current_route = []
    current_demand = 0
    current_dist = 0.0
    current_battery = battery_max
    current_time = 0.0
    current_pos = 0
    
    for angle, c in angles:
        # Check if we can add c to current route
        # Simplified check: just capacity and rough battery
        if current_demand + demands[c] > capacity:
            # Close current route
            if current_route:
                routes.append(current_route)
            current_route = [c]
            current_demand = demands[c]
            # Reset state for new route? 
            # This simple heuristic might produce infeasible routes regarding battery/time.
            # We will rely on SA to fix them or penalize them.
            # For better init, let's just group by capacity and angle.
            current_pos = 0
            current_battery = battery_max
            current_time = 0.0
            continue
            
        current_route.append(c)
        current_demand += demands[c]
        
    if current_route:
        routes.append(current_route)
        
    return routes


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
    
    # Initial Solution
    routes = _construct_initial_routes(instance, dist)
    
    # Ensure all customers are covered. 
    # The simple init above might miss some if logic is flawed. 
    # Let's verify and add missing ones as single routes.
    covered = set()
    for r in routes:
        covered.update(r)
    missing = [c for c in range(1, n_customers + 1) if c not in covered]
    for c in missing:
        routes.append([c])
        
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
        best_feasible_score = replica_scores[min_pen_idx] # Use score as proxy, though infeasible

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
            # Pick a random customer
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
                # Move within same route (like 2-opt relocation)
                if len(src_route) < 1:
                    # Put it back
                    src_route.insert(pos_c, c)
                    continue
                dst_route = src_route
                # Pick insertion point
                ins_pos = random.randint(0, len(dst_route))
                dst_route.insert(ins_pos, c)
            else:
                # Move to different route
                dst_route = new_routes[dst_r_idx]
                ins_pos = random.randint(0, len(dst_route))
                dst_route.insert(ins_pos, c)
                
            # Remove empty routes
            new_routes = [r for r in new_routes if r]
            
        elif move_type < 0.7:
            # Swap: Exchange two customers from different routes
            # Pick two random routes
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
            
            # Reverse segment i to j
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
                # Try to swap with a cooler replica
                swap_idx = r_idx - 1
                if random.random() < 0.1: # 10% chance to attempt swap
                    delta_swap = replica_scores[r_idx] - replica_scores[swap_idx]
                    # Metropolis criterion for swap: accept if hotter has worse score
                    # Here r_idx is hotter (higher T) if indices are ordered by T desc?
                    # Let's assume indices 0,1,2 correspond to T 100, 50, 10.
                    # So r_idx is hotter than swap_idx.
                    # We accept swap if Score(r_idx) < Score(swap_idx) OR random < exp(-(Score(r_idx)-Score(swap_idx))/T(r_idx))
                    # Actually, standard PT: accept swap with prob 1 if S_hot < S_cold, else exp(-(S_hot - S_cold)/T_hot)
                    # Here S_hot = replica_scores[r_idx], S_cold = replica_scores[swap_idx]
                    # T_hot = temperatures[r_idx]
                    if delta_swap < 0:
                        # Hotter replica has better score, swap is always good? 
                        # No, we want to keep the good solution in the cold replica?
                        # Standard PT: 
                        # If S_hot < S_cold, accept with prob 1.
                        # Else accept with prob exp(-(S_hot - S_cold) / T_hot)
                        pass
                    else:
                        prob = math.exp(-delta_swap / temperatures[r_idx])
                        if random.random() < prob:
                            # Swap
                            replicas[r_idx], replicas[swap_idx] = replicas[swap_idx], replicas[r_idx]
                            replica_scores[r_idx], replica_scores[swap_idx] = replica_scores[swap_idx], replica_scores[r_idx]
                            replica_penalties[r_idx], replica_penalties[swap_idx] = replica_penalties[swap_idx], replica_penalties[r_idx]
                            replica_feasible[r_idx], replica_feasible[swap_idx] = replica_feasible[swap_idx], replica_feasible[r_idx]
            
        step += 1
        
        # Cooling (optional, but PT usually keeps T constant or cools slowly)
        # Let's keep T constant for simplicity in this short run
        
    return best_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    
    # 3 instances, 150s total, leave 10s buffer -> 40s per instance
    per_instance_time = 40
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END

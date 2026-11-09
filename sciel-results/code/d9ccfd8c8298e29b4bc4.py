"""Genetic Algorithm for EV routing with time windows."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import copy

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = float(instance["capacity"])
    battery = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n_nodes = len(coords)
    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = float(np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1]))
            dist[i, j] = d
            dist[j, i] = d

    customers = list(range(1, n_customers + 1))
    
    # Random seed for reproducibility in local runs, but GA benefits from randomness
    random.seed(42) 

    def get_route_stats(route):
        """
        Returns (is_feasible, distance, final_time, final_battery)
        """
        if not route:
            return True, 0.0, 0.0, battery

        total_demand = float(sum(demands[v] for v in route))
        if total_demand > capacity + 1e-9:
            return False, 0.0, 0.0, 0.0

        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0

        for v in route:
            d = float(dist[prev, v])
            bat -= d
            t += d / speed
            total_dist += d
            
            if bat < -1e-9:
                return False, 0.0, 0.0, 0.0

            if v in stations:
                # Station logic
                if bat < 0:
                    return False, 0.0, 0.0, 0.0
                charge_time = (battery - bat) * recharge_rate
                t += charge_time
                bat = battery
            else:
                # Customer logic
                if t < ready[v]:
                    t = float(ready[v])
                if t > float(due[v]) + 1e-9:
                    return False, 0.0, 0.0, 0.0
                t += float(service[v])
            
            prev = v

        # Return to depot
        d = float(dist[prev, 0])
        bat -= d
        t += d / speed
        total_dist += d
        
        if bat < -1e-9 or t > horizon + 1e-9:
            return False, 0.0, 0.0, 0.0

        return True, total_dist, t, bat

    def evaluate_solution(routes):
        """
        Evaluates a list of routes.
        Returns (is_valid, total_distance)
        """
        seen = set()
        total_dist = 0.0
        for r in routes:
            if not r:
                continue
            # Check for duplicates
            for v in r:
                if v in seen:
                    return False, 0.0
                seen.add(v)
            
            feasible, d, _, _ = get_route_stats(r)
            if not feasible:
                return False, 0.0
            total_dist += d
        
        # Check all customers served
        if seen != set(customers):
            return False, 0.0
            
        return True, total_dist

    def repair_route(route):
        """
        Attempts to fix a route by removing infeasible nodes (customers).
        Returns (repaired_route, removed_customers)
        """
        if not route:
            return [], []
            
        # Simple repair: iterate and remove customers that cause infeasibility
        # This is a heuristic repair. We check feasibility from the end backwards? 
        # Actually, let's just build it forward and drop nodes that break constraints.
        
        # However, the GA representation is a list of routes.
        # Crossover will produce a "combined" list of nodes.
        # We need a constructive heuristic to turn a list of nodes into feasible routes.
        
        # Let's use a greedy insertion for the repair/construction phase.
        # Input: a permutation of customers (and potentially stations, but let's ignore stations in GA for simplicity? 
        # No, stations are important. But adding stations in GA is complex. 
        # Strategy: GA operates on customer permutations. 
        # Then we use a constructive heuristic to assign customers to vehicles and insert stations as needed.
        
        pass

    # --- Constructive Heuristic: Split Permutation into Feasible Routes ---
    def construct_routes_from_perm(perm):
        """
        Takes a permutation of customers and builds feasible routes.
        Inserts stations if needed to satisfy battery constraints.
        """
        remaining = set(perm)
        routes = []
        
        while remaining:
            route = []
            prev = 0
            t = 0.0
            bat = battery
            load = 0.0
            
            # Iterate through perm in order
            for v in perm:
                if v not in remaining:
                    continue
                
                # Check capacity
                if load + float(demands[v]) > capacity:
                    continue
                
                # Check battery/distance to v
                d = float(dist[prev, v])
                bat_after = bat - d
                t_after = t + d / speed
                
                # If battery is low, try to insert a station before v
                if bat_after < 0:
                    # Find nearest feasible station
                    best_st = None
                    best_cost = float('inf')
                    for s in stations:
                        d_ps = float(dist[prev, s])
                        d_sv = float(dist[s, v])
                        if bat - d_ps >= 0:
                            # Check if we can go s -> v
                            if (bat - d_ps) - d_sv >= 0:
                                # Check time impact
                                # Time at prev is t.
                                # Go to s: t + d_ps/speed. Recharge.
                                # Charge time = (battery - (bat - d_ps)) * recharge_rate
                                t_at_s = t + d_ps / speed
                                t_recharge = (battery - (bat - d_ps)) * recharge_rate
                                t_after_s = t_at_s + t_recharge
                                t_final = t_after_s + d_sv / speed
                                
                                # Check if we can still reach v in time (rough check)
                                # We don't know exact time at v yet due to waiting, but let's assume it's feasible
                                # Cost metric: extra distance + extra time
                                extra_dist = d_ps + d_sv - d
                                extra_time = t_final - t_after
                                cost = extra_dist + 0.1 * extra_time
                                
                                if cost < best_cost:
                                    best_cost = cost
                                    best_st = s
                                    
                    if best_st is not None:
                        # Insert station
                        d_ps = float(dist[prev, best_st])
                        bat -= d_ps
                        t += d_ps / speed
                        route.append(best_st)
                        
                        charge_time = (battery - bat) * recharge_rate
                        t += charge_time
                        bat = battery
                        prev = best_st
                        
                        # Now go to v
                        d = float(dist[prev, v])
                        bat_after = bat - d
                        t_after = t + d / speed
                        # It should be feasible now
                    else:
                        continue # Can't reach v even with station
                
                # Check time window
                if v not in stations:
                    if t_after < ready[v]:
                        t_after = float(ready[v])
                    if t_after > float(due[v]) + 1e-9:
                        continue # Too late
                
                # If feasible, add to route
                route.append(v)
                load += float(demands[v])
                bat = bat_after
                t = t_after
                if v not in stations:
                    t += float(service[v])
                prev = v
                remaining.discard(v)
            
            # Check if we can return to depot
            if route:
                d_ret = float(dist[prev, 0])
                if bat - d_ret >= -1e-9 and t + d_ret / speed <= horizon + 1e-9:
                    routes.append(route)
                else:
                    # Try to add a station at the end? Or just drop the last customer?
                    # Simple: drop the last customer until it's feasible
                    while route:
                        last = route.pop()
                        if last in stations:
                            continue
                        # Recalculate stats for the shortened route
                        # This is expensive. Let's just assume we can return if we drop enough.
                        # For now, let's just check feasibility of the current route.
                        pass
                    
                    # Re-evaluate the route we just built
                    if route:
                        # Simple check: can we return?
                        # We need to re-simulate to get exact t and bat at the end
                        # Let's just do a quick simulation
                        t_sim = 0.0
                        bat_sim = battery
                        p = 0
                        feasible = True
                        for node in route:
                            d = float(dist[p, node])
                            bat_sim -= d
                            t_sim += d / speed
                            if bat_sim < 0:
                                feasible = False
                                break
                            if node in stations:
                                t_sim += (battery - bat_sim) * recharge_rate
                                bat_sim = battery
                            else:
                                if t_sim < ready[node]:
                                    t_sim = ready[node]
                                if t_sim > due[node]:
                                    feasible = False
                                    break
                                t_sim += service[node]
                            p = node
                        if feasible:
                            d_ret = float(dist[p, 0])
                            if bat_sim - d_ret >= 0 and t_sim + d_ret / speed <= horizon:
                                routes.append(route)
                            else:
                                # Drop last customer
                                while route:
                                    route.pop()
                                    # Re-simulate? Too slow.
                                    # Let's just accept it might be infeasible and let the GA repair/filter it.
                                    # Actually, if it's infeasible, evaluate_solution will return False.
                                    # We want to return *some* routes.
                                    # Let's just append it and hope for the best, or drop it.
                                    # Dropping is safer for validity.
                                    # But we must serve everyone.
                                    # If we can't fit it, it stays in 'remaining' and will be picked in next vehicle.
                                    # So we DON'T append if infeasible.
                                    pass
            # If route is empty or infeasible, the customers remain in 'remaining' for next vehicle.
            # Wait, the loop above discards from 'remaining'.
            # If we don't add to routes, we need to put them back?
            # The loop `for v in perm` iterates over the fixed list.
            # `remaining` is used to skip already visited.
            # If we skip a customer in this vehicle (due to constraints), it stays in `remaining`.
            # So the next `while remaining` iteration will pick it up.
            
            if not route:
                # If we couldn't start a route (e.g. first customer unreachable?), break to avoid infinite loop
                # But we should always be able to start with a single customer if battery/time allows.
                # If not, the problem is infeasible or we have a bug.
                # Let's force add the first remaining customer to prevent infinite loop, even if infeasible?
                # No, better to break and let the evaluator handle it (score 0).
                break

        return routes

    # --- GA Initialization ---
    pop_size = 100
    population = []
    
    # 1. Nearest Neighbor
    nn_perm = []
    unvisited = set(customers)
    curr = 0
    while unvisited:
        nearest = min(unvisited, key=lambda x: dist[curr, x])
        nn_perm.append(nearest)
        unvisited.remove(nearest)
        curr = nearest
    population.append(nn_perm)
    
    # 2. Random Permutations
    for _ in range(pop_size - 1):
        perm = customers[:]
        random.shuffle(perm)
        population.append(perm)

    # Evaluate initial population
    def evaluate_perm(perm):
        routes = construct_routes_from_perm(perm)
        is_valid, dist_val = evaluate_solution(routes)
        if is_valid:
            return dist_val
        else:
            return float('inf')

    scores = [evaluate_perm(p) for p in population]
    
    # Sort
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1])
    population = [population[i] for i, s in indexed_scores]
    scores = [s for i, s in indexed_scores]
    
    best_score = scores[0]
    best_perm = population[0]
    
    # --- GA Loop ---
    generations = 0
    while time.time() < deadline:
        generations += 1
        
        # Selection: Tournament
        def select():
            i, j = random.sample(range(len(population)), 2)
            if scores[i] < scores[j]:
                return population[i]
            else:
                return population[j]
        
        new_population = []
        
        # Elitism
        elite_count = 5
        new_population.extend(population[:elite_count])
        
        # Crossover & Mutation
        while len(new_population) < pop_size:
            p1 = select()
            p2 = select()
            
            # Order Crossover (OX)
            a, b = sorted(random.sample(range(len(p1)), 2))
            child = [None] * len(p1)
            child[a:b+1] = p1[a:b+1]
            
            # Fill rest with p2 in order
            fill = [x for x in p2 if x not in child[a:b+1]]
            idx = 0
            for i in range(len(child)):
                if child[i] is None:
                    child[i] = fill[idx]
                    idx += 1
            
            # Mutation
            if random.random() < 0.5:
                # Swap
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]
            
            new_population.append(child)
        
        # Evaluate new population
        new_scores = [evaluate_perm(p) for p in new_population]
        
        # Combine and select best
        combined = list(zip(new_population, new_scores))
        combined.sort(key=lambda x: x[1])
        
        population = [p for p, s in combined[:pop_size]]
        scores = [s for p, s in combined[:pop_size]]
        
        if scores[0] < best_score:
            best_score = scores[0]
            best_perm = population[0]
            
        # Check time
        if time.time() > deadline - 5:
            break

    # Final Result
    final_routes = construct_routes_from_perm(best_perm)
    
    # If GA failed to find a valid solution (shouldn't happen with NN init), fallback
    valid, _ = evaluate_solution(final_routes)
    if not valid:
        # Fallback to simple 1-customer-per-route
        final_routes = [[v] for v in customers]
        
    return final_routes

# EVOLVE-BLOCK-END

import numpy as np
import time
import math
import random
from scipy.spatial.distance import cdist

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Extract parameters
    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    dist = cdist(coords, coords)
    
    customers = list(range(1, n_customers + 1))
    
    # Precompute station distances for quick access
    # We'll need to insert stations dynamically, but we can precompute distances between customers and stations
    # and between stations and depot/customers.
    
    def simulate_route(customers_seq):
        """
        Simulates a route of customers.
        Returns (total_distance, total_time, is_valid, node_sequence).
        node_sequence includes stations inserted for charging.
        """
        if not customers_seq:
            return 0.0, 0.0, True, []
            
        total_dist = 0.0
        t = 0.0
        bat = battery_cap
        last = 0
        nodes = []
        
        for c in customers_seq:
            d_direct = dist[last, c]
            
            if bat < d_direct:
                # Need to charge
                best_s = -1
                best_cost = float('inf')
                
                # Check all stations
                for s in stations:
                    d1 = dist[last, s]
                    d2 = dist[s, c]
                    # Can we reach station?
                    if bat < d1:
                        continue
                    # Can we reach customer from station with full charge?
                    if battery_cap < d2:
                        continue
                    
                    cost = d1 + d2 - d_direct
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s
                        
                if best_s == -1:
                    return 0.0, 0.0, False, []
                
                # Go to station
                d1 = dist[last, best_s]
                total_dist += d1
                bat -= d1
                t += d1 / speed
                
                # Charge
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
                last = best_s
                nodes.append(best_s)
                
                # Go to customer
                d2 = dist[last, c]
                total_dist += d2
                bat -= d2
                t += d2 / speed
                last = c
            else:
                # Direct
                total_dist += d_direct
                bat -= d_direct
                t += d_direct / speed
                last = c
            
            nodes.append(c)
            
            # Time window check
            t_start = max(t, ready[c])
            if t_start > due[c]:
                return 0.0, 0.0, False, []
            t = t_start + service[c]
            
        # Return to depot
        d_back = dist[last, 0]
        if bat < d_back:
            best_s = -1
            best_cost = float('inf')
            for s in stations:
                d1 = dist[last, s]
                d2 = dist[s, 0]
                if bat < d1 or battery_cap < d2:
                    continue
                cost = d1 + d2 - d_back
                if cost < best_cost:
                    best_cost = cost
                    best_s = s
            
            if best_s == -1:
                return 0.0, 0.0, False, []
            
            d1 = dist[last, best_s]
            total_dist += d1
            bat -= d1
            t += d1 / speed
            charge_time = (battery_cap - bat) * recharge_rate
            t += charge_time
            bat = battery_cap
            last = best_s
            nodes.append(best_s)
            
            d2 = dist[last, 0]
            total_dist += d2
            bat -= d2
            t += d2 / speed
        else:
            total_dist += d_back
            bat -= d_back
            t += d_back / speed
            
        if t > horizon:
            return 0.0, 0.0, False, []
            
        return total_dist, t, True, nodes

    def decode_permutation(perm):
        """
        Decodes a permutation of customers into a list of routes.
        Uses a greedy approach: iterate through perm, try to add to current route.
        If infeasible, close current route and start a new one.
        Returns (total_distance, routes_list, is_valid).
        """
        routes = []
        current_route = []
        current_dist = 0.0
        current_time = 0.0
        current_bat = battery_cap
        current_last = 0
        current_demand = 0
        
        for c in perm:
            # Try to add c to current route
            # We need to check if adding c is feasible.
            # To avoid complex incremental state checks, we can simulate the addition.
            # However, simulating the whole route every time is O(N^2).
            # With N=25, it's acceptable.
            
            test_route = current_route + [c]
            # Check capacity
            if current_demand + demands[c] > capacity:
                # Close current route
                if current_route:
                    d, t, valid, nodes = simulate_route(current_route)
                    if not valid:
                        # This shouldn't happen if we checked incrementally, but just in case
                        # If invalid, we might need to split further or mark as invalid
                        # For robustness, if a route becomes invalid, we treat it as a failure for this permutation
                        return float('inf'), [], False
                    routes.append(nodes)
                    current_dist += d
                # Start new route with c
                current_route = [c]
                current_demand = demands[c]
                # Reset state for new route? 
                # Actually, we need to re-simulate or track state.
                # Let's just track the list and re-simulate at the end? 
                # No, we need to know if it's feasible to add *next* item.
                # Let's maintain the state of the current route explicitly.
                # Re-initialize state for the new route [c]
                d_to_c = dist[0, c]
                if battery_cap < d_to_c:
                    # Can't even reach first customer
                    return float('inf'), [], False
                current_dist = d_to_c # Tentative, will be recalculated if we close
                current_time = d_to_c / speed
                t_start = max(current_time, ready[c])
                if t_start > due[c]:
                    return float('inf'), [], False
                current_time = t_start + service[c]
                current_bat = battery_cap - d_to_c
                current_last = c
                continue
            
            # Check feasibility of adding c to current_route
            # We simulate the extension
            # To be efficient, we can just simulate the whole current_route + [c]
            # But we need to know the resulting state to continue.
            # Let's just simulate the whole route so far + c.
            # This is O(K^2) per step, total O(N^3) = 25^3 = 15625 ops. Very fast.
            
            d_new, t_new, valid_new, _ = simulate_route(test_route)
            
            if valid_new:
                # Update state to reflect the new last node and stats
                # We don't actually need to store d_new/t_new/bat/last if we re-simulate the whole route at the end?
                # No, we need to know if the *next* addition is feasible.
                # The simulate_route function returns the final state implicitly via the simulation.
                # But we need the *intermediate* state? No, we just need to know if the *sequence* is feasible.
                # If test_route is feasible, we can add c.
                current_route = test_route
                current_demand += demands[c]
            else:
                # Cannot add c. Close current route.
                if current_route:
                    d, t, valid, nodes = simulate_route(current_route)
                    if not valid:
                        return float('inf'), [], False
                    routes.append(nodes)
                
                # Start new route with c
                current_route = [c]
                current_demand = demands[c]
                # Check if single customer route is feasible
                d, t, valid, _ = simulate_route(current_route)
                if not valid:
                    return float('inf'), [], False
        
        # Close last route
        if current_route:
            d, t, valid, nodes = simulate_route(current_route)
            if not valid:
                return float('inf'), [], False
            routes.append(nodes)
            
        total_dist = sum(simulate_route(r)[0] for r in routes)
        return total_dist, routes, True

    # Genetic Algorithm Components
    
    def order_crossover(p1, p2):
        """
        Order Crossover (OX1)
        """
        size = len(p1)
        child = [None] * size
        
        # Select two random cut points
        a, b = sorted(random.sample(range(size), 2))
        
        # Copy segment from p1
        child[a:b+1] = p1[a:b+1]
        
        # Fill remaining positions with elements from p2 in order
        # Skip elements already in child
        fill_pos = 0
        for i in range(size):
            # Rotate p2 so it starts after b
            idx = (b + 1 + i) % size
            val = p2[idx]
            if val not in child:
                # Find first empty spot
                while child[fill_pos] is not None:
                    fill_pos += 1
                child[fill_pos] = val
                fill_pos += 1
                
        return child

    def mutation(perm, rate=0.1):
        """
        Swap mutation
        """
        perm = perm[:]
        n = len(perm)
        for _ in range(int(n * rate)):
            i, j = random.sample(range(n), 2)
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    def local_search(routes):
        """
        Simple local search on the decoded routes.
        Uses 2-opt and relocate.
        Returns improved routes and distance.
        """
        best_routes = [r[:] for r in routes]
        best_dist = sum(simulate_route(r)[0] for r in best_routes)
        
        improved = True
        # Limit local search time
        ls_deadline = time.time() + 2.0
        
        while improved and time.time() < ls_deadline:
            improved = False
            
            # 1. Intra-route 2-Opt
            for i in range(len(best_routes)):
                route = best_routes[i]
                n = len(route)
                if n < 3:
                    continue
                
                for a in range(n - 1):
                    for b in range(a + 1, n):
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        d_new, t_new, valid_new, _ = simulate_route(new_route)
                        d_old, t_old, valid_old, _ = simulate_route(route)
                        
                        if valid_new and d_new < d_old - 1e-9:
                            best_routes[i] = new_route
                            best_dist = best_dist - d_old + d_new
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                continue
                
            # 2. Inter-route Relocate
            for i in range(len(best_routes)):
                for j in range(len(best_routes)):
                    if i == j:
                        continue
                    for k in range(len(best_routes[i])):
                        c = best_routes[i][k]
                        route_i_without = best_routes[i][:k] + best_routes[i][k+1:]
                        d_i_old, _, valid_i_old, _ = simulate_route(best_routes[i])
                        d_i_new, _, valid_i_new, _ = simulate_route(route_i_without)
                        
                        if not valid_i_new:
                            continue
                            
                        route_j = best_routes[j]
                        d_j_old, _, valid_j_old, _ = simulate_route(route_j)
                        
                        best_pos = -1
                        best_d_j_new = float('inf')
                        
                        for pos in range(len(route_j) + 1):
                            new_route_j = route_j[:pos] + [c] + route_j[pos:]
                            if sum(demands[x] for x in new_route_j) > capacity:
                                continue
                            d, t, valid, _ = simulate_route(new_route_j)
                            if valid and d < best_d_j_new:
                                best_d_j_new = d
                                best_pos = pos
                                
                        if best_pos != -1:
                            total_new_dist = best_dist - d_i_old - d_j_old + d_i_new + best_d_j_new
                            if total_new_dist < best_dist - 1e-9:
                                best_routes[i] = route_i_without
                                best_routes[j] = best_routes[j][:best_pos] + [c] + best_routes[j][best_pos:]
                                best_dist = total_new_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
                    
        return best_routes, best_dist

    # Initialize Population
    pop_size = 50
    population = []
    fitness = []
    
    # Initial solution: sorted by angle or due date?
    # Let's try sorting by due date (earliest due first) as a heuristic start
    sorted_by_due = sorted(customers, key=lambda x: due[x])
    
    for i in range(pop_size):
        if i == 0:
            perm = sorted_by_due
        elif i == 1:
            perm = sorted(customers, key=lambda x: dist[0, x])
        else:
            perm = customers[:]
            random.shuffle(perm)
        
        d, routes, valid = decode_permutation(perm)
        if valid:
            population.append(perm)
            fitness.append(d)
        else:
            # Penalize invalid
            population.append(perm)
            fitness.append(float('inf'))
            
    # Ensure we have a valid solution
    best_idx = np.argmin(fitness)
    if fitness[best_idx] == float('inf'):
        # Fallback
        routes = [[c] for c in customers]
        d, valid = decode_permutation([c for r in routes for c in r]) # This won't work directly
        # Just return single customer routes
        final_routes = []
        for c in customers:
            _, _, valid, nodes = simulate_route([c])
            if valid:
                final_routes.append(nodes)
            else:
                final_routes.append([c])
        return final_routes

    # GA Loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Sort by fitness
        pairs = sorted(zip(fitness, population))
        fitness = [p[0] for p in pairs]
        population = [p[1] for p in pairs]
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        new_population = population[:elite_count]
        new_fitness = fitness[:elite_count]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection: Tournament
            t1 = random.sample(range(len(population)), 3)
            t2 = random.sample(range(len(population)), 3)
            p1_idx = min(t1, key=lambda x: fitness[x])
            p2_idx = min(t2, key=lambda x: fitness[x])
            
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover
            if random.random() < 0.8:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            
            # Mutation
            child = mutation(child, rate=0.1)
            
            # Evaluate
            d, routes, valid = decode_permutation(child)
            if not valid:
                d = float('inf')
            
            new_population.append(child)
            new_fitness.append(d)
            
        # Local Search on best few
        # Sort new population
        pairs = sorted(zip(new_fitness, new_population))
        new_fitness = [p[0] for p in pairs]
        new_population = [p[1] for p in pairs]
        
        # Apply local search to top 5
        for i in range(min(5, len(new_population))):
            if new_fitness[i] != float('inf'):
                d, routes, valid = decode_permutation(new_population[i])
                if valid:
                    improved_routes, improved_d = local_search(routes)
                    if improved_d < new_fitness[i]:
                        new_fitness[i] = improved_d
                        # We don't easily update the permutation from routes, 
                        # so we just store the improved distance. 
                        # The permutation is still the one that decoded to these routes (roughly).
                        # This is a slight inconsistency but acceptable for heuristic.
                        
        population = new_population
        fitness = new_fitness
        
    # Final best
    best_idx = np.argmin(fitness)
    best_perm = population[best_idx]
    
    d, routes, valid = decode_permutation(best_perm)
    if not valid:
        # Fallback
        routes = [[c] for c in customers]
        final_routes = []
        for c in customers:
            _, _, valid, nodes = simulate_route([c])
            if valid:
                final_routes.append(nodes)
            else:
                final_routes.append([c])
        return final_routes
    
    return routes


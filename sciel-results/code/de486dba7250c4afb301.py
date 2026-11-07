"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import copy

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["hor horizon"] if 'horizon' in instance else instance.get('horizon', 1000)
    
    # Ensure horizon is available
    if 'horizon' not in instance:
        horizon = 1000
    else:
        horizon = instance['horizon']

    # Precompute distance matrix
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
            dist[i, j] = d
            dist[j, i] = d

    # Fast feasibility check for a single route
    def check_route_fast(route):
        if not route:
            return True, 0.0
        total_dist = 0.0
        t = 0.0
        bat = battery_cap
        prev = 0
        for node in route:
            d = dist[prev, node]
            total_dist += d
            bat -= d
            if bat < 0:
                return False, total_dist
            t += d / speed
            if node in stations:
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
            else:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node]:
                    return False, total_dist
                t += service[node]
            prev = node
        
        d = dist[prev, 0]
        total_dist += d
        bat -= d
        if bat < 0:
            return False, total_dist
        t += d / speed
        if t > horizon:
            return False, total_dist
        return True, total_dist

    def total_distance(routes):
        total = 0.0
        for r in routes:
            if not r:
                continue
            total += dist[0, r[0]]
            for i in range(len(r) - 1):
                total += dist[r[i], r[i + 1]]
            total += dist[r[-1], 0]
        return total

    def route_demand(route):
        return sum(demands[n] for n in route if n not in stations)

    # Generate a diverse initial solution using randomized greedy insertion
    def generate_initial_solution(seed):
        random.seed(seed)
        customers = list(range(1, n_customers + 1))
        random.shuffle(customers)
        
        routes = []
        for c in customers:
            best_route = None
            best_pos = -1
            best_cost = float('inf')
            feasible_found = False
            
            # Try inserting into existing routes
            for i, r in enumerate(routes):
                # Quick demand check
                if route_demand(r) + demands[c] > capacity:
                    continue
                for j in range(len(r) + 1):
                    new_route = r[:j] + [c] + r[j:]
                    feas, d = check_route_fast(new_route)
                    if feas:
                        # Calculate cost increase
                        old_d = dist[0, r[0]] if r else 0
                        for k in range(len(r)-1):
                            old_d += dist[r[k], r[k+1]]
                        if r:
                            old_d += dist[r[-1], 0]
                        
                        new_d = dist[0, new_route[0]]
                        for k in range(len(new_route)-1):
                            new_d += dist[new_route[k], new_route[k+1]]
                        new_d += dist[new_route[-1], 0]
                        
                        cost = new_d - old_d
                        if cost < best_cost:
                            best_cost = cost
                            best_route = i
                            best_pos = j
                            feasible_found = True
            
            if feasible_found:
                r = routes[best_route]
                r.insert(best_pos, c)
            else:
                routes.append([c])
        
        return routes

    # Tabu Search
    def tabu_search(routes, max_iters=5000):
        current_routes = [list(r) for r in routes]
        current_cost = total_distance(current_routes)
        best_routes = [list(r) for r in current_routes]
        best_cost = current_cost
        
        tabu_list = []
        tabu_tenure = 20
        aspiration_cost = best_cost
        
        no_improve_count = 0
        max_no_improve = 100
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            move_type = random.choice(['relocate', 'swap', '2opt', 'oropt'])
            improved = False
            
            if move_type == 'relocate':
                # Move customer from route i to route j
                i = random.randint(0, len(current_routes) - 1)
                if len(current_routes[i]) == 0: continue
                k = random.randint(0, len(current_routes[i]) - 1)
                c = current_routes[i][k]
                
                # Remove c from i
                new_i = current_routes[i][:k] + current_routes[i][k+1:]
                
                # Try inserting into j
                for j in range(len(current_routes)):
                    if i == j: continue
                    for pos in range(len(current_routes[j]) + 1):
                        new_j = current_routes[j][:pos] + [c] + current_routes[j][pos:]
                        
                        # Check feasibility
                        feas_i, _ = check_route_fast(new_i)
                        feas_j, _ = check_route_fast(new_j)
                        if not feas_i or not feas_j: continue
                        if route_demand(new_j) > capacity: continue
                        
                        # Check tabu
                        is_tabu = False
                        for (ti, ti_idx, tj, tj_idx) in tabu_list:
                            if ti == i and ti_idx == k and tj == j and tj_idx == pos:
                                is_tabu = True
                                break
                            
                        new_cost = total_distance(new_i + new_j)
                        
                        # Aspiration: accept if better than best
                        if is_tabu and new_cost >= best_cost:
                            continue
                            
                        if new_cost < current_cost or (is_tabu and new_cost < best_cost):
                            current_routes[i] = new_i
                            current_routes[j] = new_j
                            current_cost = new_cost
                            tabu_list.append((i, k, j, pos))
                            if len(tabu_list) > tabu_tenure:
                                tabu_list.pop(0)
                            
                            if new_cost < best_cost:
                                best_cost = new_cost
                                best_routes = [list(r) for r in current_routes]
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                            improved = True
                            break
                    if improved: break
                if not improved:
                    no_improve_count += 1

            elif move_type == 'swap':
                # Swap customer c1 in route i with c2 in route j
                if len(current_routes) < 2: continue
                i, j = random.sample(range(len(current_routes)), 2)
                if len(current_routes[i]) == 0 or len(current_routes[j]) == 0: continue
                k1 = random.randint(0, len(current_routes[i]) - 1)
                k2 = random.randint(0, len(current_routes[j]) - 1)
                c1 = current_routes[i][k1]
                c2 = current_routes[j][k2]
                
                new_i = current_routes[i][:]
                new_j = current_routes[j][:]
                new_i[k1] = c2
                new_j[k2] = c1
                
                feas_i, _ = check_route_fast(new_i)
                feas_j, _ = check_route_fast(new_j)
                if not feas_i or not feas_j:
                    no_improve_count += 1
                    continue
                if route_demand(new_i) > capacity or route_demand(new_j) > capacity:
                    no_improve_count += 1
                    continue
                    
                is_tabu = False
                for (ti, ti_idx, tj, tj_idx) in tabu_list:
                    if (ti == i and ti_idx == k1 and tj == j and tj_idx == k2) or \
                       (ti == j and ti_idx == k2 and tj == i and tj_idx == k1):
                        is_tabu = True
                        break
                        
                new_cost = total_distance(new_i + new_j)
                
                if is_tabu and new_cost >= best_cost:
                    no_improve_count += 1
                    continue
                
                if new_cost < current_cost or (is_tabu and new_cost < best_cost):
                    current_routes[i] = new_i
                    current_routes[j] = new_j
                    current_cost = new_cost
                    tabu_list.append((i, k1, j, k2))
                    if len(tabu_list) > tabu_tenure:
                        tabu_list.pop(0)
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_routes = [list(r) for r in current_routes]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    improved = True
                else:
                    no_improve_count += 1

            elif move_type == '2opt':
                i = random.randint(0, len(current_routes) - 1)
                r = current_routes[i]
                if len(r) < 2: continue
                a, b = random.sample(range(len(r)), 2)
                if a > b: a, b = b, a
                new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                
                feas, _ = check_route_fast(new_r)
                if not feas:
                    no_improve_count += 1
                    continue
                    
                is_tabu = False
                for (ti, ti_idx, tj, tj_idx) in tabu_list:
                    if ti == i and ti_idx == a and tj == i and tj_idx == b:
                        is_tabu = True
                        break
                        
                new_cost = total_distance(new_r + current_routes[:i] + current_routes[i+1:])
                
                if is_tabu and new_cost >= best_cost:
                    no_improve_count += 1
                    continue
                
                if new_cost < current_cost or (is_tabu and new_cost < best_cost):
                    current_routes[i] = new_r
                    current_cost = new_cost
                    tabu_list.append((i, a, i, b))
                    if len(tabu_list) > tabu_tenure:
                        tabu_list.pop(0)
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_routes = [list(r) for r in current_routes]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    improved = True
                else:
                    no_improve_count += 1

            elif move_type == 'oropt':
                i = random.randint(0, len(current_routes) - 1)
                r = current_routes[i]
                if len(r) < 2: continue
                length = random.choice([1, 2, 3])
                if len(r) < length: continue
                a = random.randint(0, len(r) - length)
                segment = r[a:a+length]
                rest = r[:a] + r[a+length:]
                b = random.randint(0, len(rest))
                new_r = rest[:b] + segment + rest[b:]
                
                if new_r == r: continue
                
                feas, _ = check_route_fast(new_r)
                if not feas:
                    no_improve_count += 1
                    continue
                    
                is_tabu = False
                for (ti, ti_idx, tj, tj_idx) in tabu_list:
                    if ti == i and ti_idx == a and tj == i and tj_idx == b:
                        is_tabu = True
                        break
                        
                new_cost = total_distance(new_r + current_routes[:i] + current_routes[i+1:])
                
                if is_tabu and new_cost >= best_cost:
                    no_improve_count += 1
                    continue
                
                if new_cost < current_cost or (is_tabu and new_cost < best_cost):
                    current_routes[i] = new_r
                    current_cost = new_cost
                    tabu_list.append((i, a, i, b))
                    if len(tabu_list) > tabu_tenure:
                        tabu_list.pop(0)
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_routes = [list(r) for r in current_routes]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    improved = True
                else:
                    no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Perturb: random restart from best or slight perturbation
                # Here we just break and restart from best in outer loop
                break

        return best_routes, best_cost

    # Multi-start strategy
    best_overall_routes = None
    best_overall_cost = float('inf')
    
    # Generate initial solutions and run Tabu Search on them
    n_starts = 20
    for s in range(n_starts):
        if time.time() > deadline - 5:
            break
            
        init_routes = generate_initial_solution(s)
        # Quick local search (2-opt) to polish initial solution
        # (Skipping for speed, Tabu will handle it)
        
        tabu_routes, tabu_cost = tabu_search(init_routes, max_iters=2000)
        
        if tabu_cost < best_overall_cost:
            best_overall_cost = tabu_cost
            best_overall_routes = tabu_routes

    # If we still have time, do a final polish on the best solution
    if best_overall_routes is not None and time.time() < deadline - 10:
        # Run a more aggressive local search on the best solution
        current_routes = [list(r) for r in best_overall_routes]
        current_cost = total_distance(current_routes)
        
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            # Simple 2-opt pass
            improved = False
            for i in range(len(current_routes)):
                r = current_routes[i]
                n = len(r)
                for a in range(n - 1):
                    for b in range(a + 1, n):
                        new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                        if new_r == r: continue
                        feas, d_new = check_route_fast(new_r)
                        if not feas: continue
                        d_old = total_distance([r])
                        if d_new < d_old - 1e-9:
                            current_routes[i] = new_r
                            current_cost = total_distance(current_routes)
                            improved = True
                            break
                    if improved: break
                if improved: break
            if not improved: break
        
        if current_cost < best_overall_cost:
            best_overall_routes = current_routes
            best_overall_cost = current_cost

    if best_overall_routes is None:
        # Fallback: one customer per route
        customers = list(range(1, n_customers + 1))
        customers.sort(key=lambda c: due[c])
        best_overall_routes = [[c] for c in customers]

    return best_overall_routes
# EVOLVE-BLOCK-END

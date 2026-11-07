"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def plan(instance):
    start_time = time.time()
    # Use a tighter deadline to ensure we return well before 120s
    deadline = start_time + 100.0 

    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    diff = coords[:, np.newaxis] - coords[np.newaxis, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dist, 0.0)

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def simulate_route(route):
        """
        Simulate a route starting from depot.
        Returns (total_dist, feasible, end_time).
        """
        if not route:
            return 0.0, True, 0.0

        t = 0.0
        b = battery
        total_d = 0.0
        prev = 0

        for node in route:
            d = dist[prev][node]
            t += d / speed
            b -= d
            total_d += d

            if b < -1e-9:
                return total_d, False, t

            if node in station_set:
                if b < battery:
                    t += (battery - b) * recharge_rate
                b = battery
            else:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return total_d, False, t
                t += service[node]

            prev = node

        d = dist[prev][0]
        t += d / speed
        b -= d
        total_d += d

        if b < -1e-9:
            return total_d, False, t
        if t > horizon + 1e-9:
            return total_d, False, t

        return total_d, True, t

    def total_demand(route):
        return sum(demands[v] for v in route if v not in station_set)

    def construct_greedy():
        unvisited = set(customers)
        routes = []
        
        # Sort by due date, then distance from depot
        cust_sorted = sorted(customers, key=lambda v: (due[v], dist[0][v]))

        while unvisited:
            route = []
            current_node = 0
            t = 0.0
            b = battery
            load = 0

            while unvisited:
                best_cust = None
                best_dist = float('inf')

                # To speed up, we can limit the number of candidates considered
                # but with 25 customers, checking all is fine.
                for c in cust_sorted:
                    if c not in unvisited:
                        continue
                    if load + demands[c] > capacity:
                        continue

                    d_to_c = dist[current_node][c]
                    if d_to_c >= best_dist:
                        continue

                    new_b = b - d_to_c
                    if new_b < -1e-9:
                        continue

                    new_t = t + d_to_c / speed
                    if new_t < ready[c]:
                        new_t = ready[c]
                    if new_t > due[c] + 1e-9:
                        continue
                    new_t += service[c]

                    # Check return to depot
                    d_back = dist[c][0]
                    final_b = new_b - d_back
                    final_t = new_t + d_back / speed
                    if final_b < -1e-9 or final_t > horizon + 1e-9:
                        continue

                    best_dist = d_to_c
                    best_cust = c

                if best_cust is None:
                    break

                d_to_c = dist[current_node][best_cust]
                t += d_to_c / speed
                b -= d_to_c
                if t < ready[best_cust]:
                    t = ready[best_cust]
                t += service[best_cust]
                load += demands[best_cust]
                route.append(best_cust)
                current_node = best_cust
                unvisited.discard(best_cust)

            if route:
                _, feasible, _ = simulate_route(route)
                if feasible:
                    routes.append(route)
                else:
                    # Fallback if the greedy route became infeasible due to rounding or logic error
                    for c in route:
                        routes.append([c])
                    unvisited.update(route)
        return routes

    def two_opt_route(route):
        """Apply 2-opt to a single route."""
        if len(route) < 2:
            return route

        best_route = list(route)
        best_dist, _, _ = simulate_route(best_route)
        n = len(best_route)
        
        # Limit iterations to prevent hanging
        max_passes = 10
        
        for _ in range(max_passes):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(n - 1):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, n):
                    # Reverse segment i+1 to j
                    new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                    nd, feasible, _ = simulate_route(new_route)
                    if feasible and nd < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = nd
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        return best_route

    # 1. Construct initial routes
    routes = construct_greedy()

    # 2. Improve each route with 2-opt
    for i in range(len(routes)):
        if time.time() > deadline - 10.0:
            break
        routes[i] = two_opt_route(routes[i])

    # 3. Simple relocation/or-opt: Try moving a customer to another route if it reduces distance
    # This is simpler than the previous full or-opt and less likely to timeout
    # We do this in a single pass to keep time bounded.
    
    current_total_dist = sum(simulate_route(r)[0] for r in routes)
    
    # We will try to move each customer once
    for i in range(len(routes)):
        if time.time() > deadline - 10.0:
            break
        if len(routes[i]) < 2:
            continue
        
        route_i = routes[i]
        # We iterate backwards to avoid index issues if we were modifying, 
        # but we are not modifying in place here, just evaluating.
        for k in range(len(route_i)):
            if time.time() > deadline - 10.0:
                break
            cust = route_i[k]
            # Remove cust
            route_i_without = route_i[:k] + route_i[k+1:]
            if not route_i_without:
                continue
            
            d_i_new, feas_i, _ = simulate_route(route_i_without)
            if not feas_i:
                continue
            
            # Try inserting into other routes
            best_j = -1
            best_pos = -1
            best_new_dist = float('inf')
            
            d_i_old, _, _ = simulate_route(route_i)
            
            for j in range(len(routes)):
                if i == j:
                    continue
                route_j = routes[j]
                d_j_old, _, _ = simulate_route(route_j)
                
                for pos in range(len(route_j) + 1):
                    new_route_j = route_j[:pos] + [cust] + route_j[pos:]
                    if total_demand(new_route_j) > capacity:
                        continue
                    d_j_new, feas_j, _ = simulate_route(new_route_j)
                    if not feas_j:
                        continue
                    
                    total_new = d_i_new + d_j_new
                    total_old = d_i_old + d_j_old
                    
                    if total_new < total_old - 1e-9 and total_new < best_new_dist:
                        best_new_dist = total_new
                        best_j = j
                        best_pos = pos
            
            if best_j != -1:
                # Perform the move
                routes[i] = route_i_without
                routes[best_j] = routes[best_j][:best_pos] + [cust] + routes[best_j][best_pos:]
                # If we removed the last element, the loop index k is now out of bounds for the modified list,
                # but since we break the inner loop and continue outer, we should be careful.
                # Actually, we are modifying routes[i] which is the list we are iterating over in the outer loop?
                # No, we are iterating over range(len(route_i)). 
                # If we modify routes[i], the next k might be invalid if we shortened it.
                # To be safe, we break the inner loop after a successful move and restart the outer logic for this route?
                # Or just break both loops and restart the whole relocation pass?
                # For simplicity and safety, let's just break the inner loop and continue to the next customer in the *original* list?
                # No, the list has changed.
                # Let's just break the inner loop and assume the next customer index is valid? 
                # If we removed index k, the element at k+1 is now at k.
                # So we should decrement k? Or just break and let the next outer loop iteration handle it?
                # Given the small size, let's just break the inner loop and continue the outer loop.
                # The current customer 'cust' is moved. The next iteration of k will look at the new element at k.
                # This is effectively fine.
                break

    # 4. Final 2-opt pass on all routes
    for i in range(len(routes)):
        if time.time() > deadline - 5.0:
            break
        routes[i] = two_opt_route(routes[i])

    # Verify coverage and feasibility
    all_served = set()
    for r in routes:
        for v in r:
            if v not in station_set:
                all_served.add(v)

    if all_served != set(customers):
        # Fallback
        routes = [[v] for v in customers]

    for r in routes:
        _, feasible, _ = simulate_route(r)
        if not feasible:
            routes = [[v] for v in customers]
            break

    return routes
# EVOLVE-BLOCK-END

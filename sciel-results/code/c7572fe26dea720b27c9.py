"""EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def plan(instance):
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin from 150s limit

    coords = np.array(instance["coords"], dtype=float)
    n_nodes = len(coords)
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

    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist[i, j] = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def simulate_route(route):
        """Simulate a route, return (total_distance, is_feasible)."""
        if not route:
            return 0.0, True
        total_dist = 0.0
        time_t = 0.0
        battery = battery_cap
        load = 0.0

        # From depot to first node
        prev = 0
        for node in route:
            d = dist[prev, node]
            total_dist += d
            time_t += d / speed
            battery -= d

            if battery < -1e-9:
                return total_dist, False

            if node in station_set:
                # Charging station
                charge_time = max(0.0, (battery_cap - battery)) * recharge_rate
                time_t += charge_time
                battery = battery_cap
            else:
                # Customer
                time_t = max(time_t, ready[node])
                if time_t > due[node] + 1e-9:
                    return total_dist, False
                time_t += service[node]
                load += demands[node]
                if load > capacity + 1e-9:
                    return total_dist, False

            prev = node

        # Return to depot
        d = dist[prev, 0]
        total_dist += d
        time_t += d / speed
        battery -= d

        if battery < -1e-9:
            return total_dist, False
        if time_t > horizon + 1e-9:
            return total_dist, False

        return total_dist, True

    def total_distance(routes):
        total = 0.0
        for route in routes:
            if not route:
                continue
            prev = 0
            for node in route:
                total += dist[prev, node]
                prev = node
            total += dist[prev, 0]
        return total

    def is_feasible_all(routes):
        for route in routes:
            _, feasible = simulate_route(route)
            if not feasible:
                return False
        return True

    def greedy_construct(order_seed):
        """Construct routes greedily: sort customers by angle/distance, insert into existing routes."""
        # Sort customers by nearest neighbor from depot for initial ordering
        cust_list = list(customers)
        random.Random(order_seed).shuffle(cust_list)

        routes = []
        unassigned = set(cust_list)
        route_info = []  # (route, last_node, load, time_at_end, battery_at_end, total_dist)

        # Try to assign each customer to an existing route or create a new one
        # Use a simple insertion heuristic
        for _ in range(len(cust_list)):
            if not unassigned:
                break

            # Pick the customer closest to depot among unassigned
            best_cust = min(unassigned, key=lambda c: dist[0, c])

            # Try inserting into existing routes
            best_cost = float('inf')
            best_route_idx = -1
            best_pos = -1

            for ri, route in enumerate(routes):
                if not route:
                    continue
                # Try each position in the route
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [best_cust] + route[pos:]
                    d, feasible = simulate_route(new_route)
                    if feasible and d < best_cost:
                        best_cost = d
                        best_route_idx = ri
                        best_pos = pos

            if best_route_idx >= 0:
                routes[best_route_idx] = routes[best_route_idx][:best_pos] + [best_cust] + routes[best_route_idx][best_pos:]
            else:
                # Create new route
                routes.append([best_cust])

            unassigned.discard(best_cust)

        return routes

    def two_opt_route(route):
        """Apply 2-opt to a single route."""
        if len(route) < 2:
            return route
        improved = True
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    d_old, _ = simulate_route(route)
                    d_new, feasible = simulate_route(new_route)
                    if feasible and d_new < d_old - 1e-9:
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
        return route

    def relocate(routes):
        """Try moving a customer from one route to another."""
        improved = True
        while improved:
            improved = False
            for ri, route in enumerate(routes):
                if len(route) < 1:
                    continue
                for pos in range(len(route)):
                    cust = route[pos]
                    # Remove from current route
                    route_without = route[:pos] + route[pos+1:]
                    d_old, _ = simulate_route(route)
                    d_old_rest, _ = simulate_route(route_without)

                    # Try inserting into other routes
                    best_cost = float('inf')
                    best_rj = -1
                    best_pos2 = -1

                    for rj, route2 in enumerate(routes):
                        if rj == ri:
                            continue
                        for pos2 in range(len(route2) + 1):
                            new_route2 = route2[:pos2] + [cust] + route2[pos2:]
                            d_new2, feasible2 = simulate_route(new_route2)
                            if not feasible2:
                                continue
                            d_total = d_old_rest + d_new2
                            if d_total < best_cost:
                                best_cost = d_total
                                best_rj = rj
                                best_pos2 = pos2

                    if best_rj >= 0 and best_cost < d_old + d_old_rest - 1e-9:
                        routes[ri] = route_without
                        routes[best_rj] = routes[best_rj][:best_pos2] + [cust] + routes[best_rj][best_pos2:]
                        improved = True
                        break
                if improved:
                    break
        return routes

    def swap_customers(routes):
        """Try swapping customers between two routes."""
        improved = True
        while improved:
            improved = False
            for ri in range(len(routes)):
                for rj in range(ri + 1, len(routes)):
                    for pi in range(len(routes[ri])):
                        for pj in range(len(routes[rj])):
                            ci = routes[ri][pi]
                            cj = routes[rj][pj]
                            # Swap
                            routes[ri][pi] = cj
                            routes[rj][pj] = ci
                            if is_feasible_all(routes):
                                d = total_distance(routes)
                                # Check if better - we need old distance
                                # This is expensive; simplify by just checking feasibility and keeping if distance improves
                                # For now, accept if feasible (we'll rely on 2-opt to clean up)
                                improved = True
                            else:
                                # Revert
                                routes[ri][pi] = ci
                                routes[rj][pj] = cj
                    if improved:
                        break
                if improved:
                    break
        return routes

    # Main optimization loop
    best_routes = None
    best_dist = float('inf')

    # Initial construction with multiple seeds
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        routes = greedy_construct(seed)
        if not is_feasible_all(routes):
            continue
        d = total_distance(routes)
        if d < best_dist:
            best_dist = d
            best_routes = [list(r) for r in routes]

    if best_routes is None:
        # Fallback: one customer per route
        best_routes = [[c] for c in customers]
        best_dist = total_distance(best_routes)

    # Local search improvements
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        current_routes = [list(r) for r in best_routes]
        current_dist = best_dist

        # 2-opt on each route
        for i in range(len(current_routes)):
            current_routes[i] = two_opt_route(current_routes[i])

        # Relocate
        current_routes = relocate(current_routes)

        # Re-apply 2-opt after relocate
        for i in range(len(current_routes)):
            current_routes[i] = two_opt_route(current_routes[i])

        if is_feasible_all(current_routes):
            d = total_distance(current_routes)
            if d < best_dist - 1e-9:
                best_dist = d
                best_routes = [list(r) for r in current_routes]
            else:
                # No improvement, try a random perturbation
                if iterations % 5 == 0:
                    # Random restart component: remove 3 random customers and reinsert
                    if len(current_routes) >= 1 and any(len(r) >= 3 for r in current_routes):
                        perturbed = [list(r) for r in current_routes]
                        removed = []
                        for _ in range(3):
                            non_empty = [i for i, r in enumerate(perturbed) if len(r) > 0]
                            if not non_empty:
                                break
                            ri = random.choice(non_empty)
                            pos = random.randrange(len(perturbed[ri]))
                            removed.append(perturbed[ri].pop(pos))
                            if not perturbed[ri]:
                                perturbed.pop(ri)
                                # Adjust indices
                                break
                        # Reinsert removed customers
                        for cust in removed:
                            best_cost = float('inf')
                            best_rj = -1
                            best_pos = -1
                            for rj, route in enumerate(perturbed):
                                for pos in range(len(route) + 1):
                                    new_route = route[:pos] + [cust] + route[pos:]
                                    dd, feasible = simulate_route(new_route)
                                    if feasible and dd < best_cost:
                                        best_cost = dd
                                        best_rj = rj
                                        best_pos = pos
                            if best_rj >= 0:
                                perturbed[best_rj] = perturbed[best_rj][:best_pos] + [cust] + perturbed[best_rj][best_pos:]
                            else:
                                perturbed.append([cust])

                        # Clean up empty routes
                        perturbed = [r for r in perturbed if r]

                        # 2-opt on perturbed
                        for i in range(len(perturbed)):
                            perturbed[i] = two_opt_route(perturbed[i])

                        if is_feasible_all(perturbed):
                            d = total_distance(perturbed)
                            if d < best_dist - 1e-9:
                                best_dist = d
                                best_routes = [list(r) for r in perturbed]

        # Check deadline
        if time.time() > deadline - 5:
            break

    return best_routes
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for evaluator

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

    # Precompute pairwise distances
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist[i, j] = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)

    # Check if a route is feasible
    def check_route(route):
        """Check if route is feasible. Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0
        total_demand = sum(demands[v] for v in route if v <= n_customers)
        if total_demand > capacity + 1e-9:
            return False, 0.0

        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_cap
        total_dist = 0.0

        for node in route:
            # Travel from curr_node to node
            d = dist[curr_node, node]
            curr_time += d / speed
            curr_battery -= d
            total_dist += d
            if curr_battery < -1e-9:
                return False, 0.0

            # If node is a station, recharge
            if node in stations:
                recharge_time = max(0, (battery_cap - curr_battery)) * recharge_rate
                curr_time += recharge_time
                curr_battery = battery_cap
            # If node is a customer, apply time window and service
            elif node <= n_customers:
                curr_time = max(curr_time, ready[node])
                if curr_time > due[node] + 1e-9:
                    return False, 0.0
                curr_time += service[node]
            # Otherwise (shouldn't happen), just pass through

            curr_node = node

        # Return to depot
        d = dist[curr_node, 0]
        curr_time += d / speed
        curr_battery -= d
        total_dist += d
        if curr_battery < -1e-9:
            return False, 0.0
        if curr_time > horizon + 1e-9:
            return False, 0.0

        return True, total_dist

    # Greedy construction with nearest-neighbor and time-window awareness
    def construct_greedy():
        """Construct routes using a greedy approach with feasibility checks."""
        # Sort customers by due time for initial priority
        customer_list = list(range(1, n_customers + 1))
        customer_list.sort(key=lambda v: (due[v], ready[v]))

        routes = []
        unassigned = set(range(1, n_customers + 1))

        while unassigned:
            if time.time() > deadline:
                break

            # Try to build a new route starting from the most urgent unassigned customer
            # Pick the customer with earliest due time among unassigned
            if not unassigned:
                break

            # Sort unassigned by due time
            unsorted_unassigned = sorted(unassigned, key=lambda v: (due[v], ready[v]))
            start_customer = unsorted_unassigned[0]

            route = [start_customer]
            unassigned.remove(start_customer)

            # Try to add more customers to this route
            while True:
                if time.time() > deadline:
                    break

                best_node = -1
                best_penalty = float('inf')

                for cand in unassigned:
                    # Try appending cand to route
                    new_route = route + [cand]
                    feasible, _ = check_route(new_route)
                    if not feasible:
                        continue

                    # Compute a cost: distance increase + time pressure
                    d_increase = dist[route[-1], cand] + dist[cand, 0] - dist[route[-1], 0]
                    # Prefer customers that are close and have later due times
                    penalty = d_increase

                    if penalty < best_penalty:
                        best_penalty = penalty
                        best_node = cand

                if best_node == -1:
                    break

                route.append(best_node)
                unassigned.remove(best_node)

            if route:
                routes.append(route)

        # Assign any remaining unassigned customers as single-route
        for cust in sorted(unassigned, key=lambda v: due[v]):
            routes.append([cust])
            unassigned.discard(cust)

        return routes

    # Relocate move: try moving a customer from one route to another
    def try_relocate(routes):
        """Try to improve routes by relocating customers between routes."""
        n_routes = len(routes)
        improved = False

        for i in range(n_routes):
            if time.time() > deadline:
                break
            if len(routes[i]) < 2:
                continue

            # Try each customer in route i
            route_i = routes[i][:]
            for pos in range(len(route_i)):
                if time.time() > deadline:
                    break
                cust = route_i[pos]

                # Try removing cust from route i
                new_route_i = route_i[:pos] + route_i[pos+1:]

                # Try inserting cust into each other route
                for j in range(n_routes):
                    if j == i:
                        continue
                    route_j = routes[j]
                    for ins_pos in range(len(route_j) + 1):
                        new_route_j = route_j[:ins_pos] + [cust] + route_j[ins_pos:]

                        # Check feasibility
                        feas_i, dist_i = check_route(new_route_i) if new_route_i else (True, 0.0)
                        feas_j, dist_j = check_route(new_route_j)

                        if not (feas_i and feas_j):
                            continue

                        # Compute total distance change
                        old_dist = check_route(route_i)[1] + check_route(route_j)[1]
                        new_dist = dist_i + dist_j

                        if new_dist < old_dist - 1e-9:
                            routes[i] = new_route_i
                            routes[j] = new_route_j
                            improved = True
                            break

                    if time.time() > deadline:
                        break

        # Remove empty routes
        routes = [r for r in routes if r]
        return routes, improved

    # 2-opt within a route
    def two_opt_route(route):
        """Apply 2-opt to a single route."""
        if len(route) < 2:
            return route

        best_route = route[:]
        best_dist = check_route(best_route)[1]

        for i in range(len(best_route) - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(best_route)):
                # Reverse segment i..j
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                feasible, new_dist = check_route(new_route)
                if feasible and new_dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = new_dist

        return best_route

    # Inter-route 2-opt: try reversing segments across two routes
    def inter_route_2opt(route_a, route_b):
        """Try inter-route 2-opt between two routes."""
        if len(route_a) < 1 or len(route_b) < 1:
            return route_a, route_b

        best_a = route_a[:]
        best_b = route_b[:]
        best_dist = check_route(best_a)[1] + check_route(best_b)[1]

        # Try all possible segment swaps
        for i1 in range(len(best_a)):
            if time.time() > deadline:
                break
            for j1 in range(i1, len(best_a)):
                for i2 in range(len(best_b)):
                    if time.time() > deadline:
                        break
                    for j2 in range(i2, len(best_b)):
                        # Swap segments
                        seg_a = best_a[i1:j1+1]
                        seg_b = best_b[i2:j2+1]

                        new_a = best_a[:i1] + seg_b + best_a[j1+1:]
                        new_b = best_b[:i2] + seg_a + best_b[j2+1:]

                        feas_a, dist_a = check_route(new_a) if new_a else (True, 0.0)
                        feas_b, dist_b = check_route(new_b) if new_b else (True, 0.0)

                        if feas_a and feas_b:
                            new_dist = dist_a + dist_b
                            if new_dist < best_dist - 1e-9:
                                best_a = new_a
                                best_b = new_b
                                best_dist = new_dist

        return best_a, best_b

    # Main algorithm
    routes = construct_greedy()

    # Iterative improvement
    max_iters = 20
    for it in range(max_iters):
        if time.time() > deadline:
            break

        # Within-route 2-opt
        changed = False
        for idx, route in enumerate(routes):
            if time.time() > deadline:
                break
            new_route = two_opt_route(route)
            if new_route != route:
                routes[idx] = new_route
                changed = True

        # Relocate
        routes, relocate_improved = try_relocate(routes)
        changed = changed or relocate_improved

        # Inter-route 2-opt on pairs of routes
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(routes)):
                if time.time() > deadline:
                    break
                new_a, new_b = inter_route_2opt(routes[i], routes[j])
                if new_a != routes[i] or new_b != routes[j]:
                    routes[i] = new_a
                    routes[j] = new_b
                    changed = True

        if not changed:
            break

    # Remove empty routes
    routes = [r for r in routes if r]

    # Final verification: ensure all customers are served exactly once
    all_customers_served = set()
    for route in routes:
        for node in route:
            if node <= n_customers:
                all_customers_served.add(node)

    # If any customer is missing, add them as single routes
    missing = set(range(1, n_customers + 1)) - all_customers_served
    for cust in sorted(missing, key=lambda v: due[v]):
        routes.append([cust])

    return routes
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def plan(instance):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

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
    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist[i, j] = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)

    customers = list(range(1, n_customers + 1))

    # Helper: simulate a route (list of node ids), return (feasible, total_distance, arrival_time_at_depot)
    def simulate(route):
        """Simulate route. Returns (feasible, total_dist)."""
        if not route:
            return True, 0.0
        t = 0.0
        batt = battery_cap
        total_dist = 0.0
        prev = 0  # depot
        for node in route:
            d = dist[prev, node]
            batt -= d
            total_dist += d
            t += d / speed
            if batt < -1e-9:
                return False, total_dist
            if node in stations:
                # Recharge
                if batt < 0:
                    batt = 0.0
                charge_time = (batt - batt) * recharge_rate  # Wait, formula: (battery - energy) * recharge_rate
                # Actually: time added = (battery_cap - batt) * recharge_rate
                charge_time = (battery_cap - batt) * recharge_rate
                t += charge_time
                batt = battery_cap
            else:
                # Customer
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return False, total_dist
                t += service[node]
            prev = node
        # Return to depot
        d = dist[prev, 0]
        batt -= d
        total_dist += d
        t += d / speed
        if batt < -1e-9:
            return False, total_dist
        if t > horizon + 1e-9:
            return False, total_dist
        return True, total_dist

    # Constructive: nearest neighbor with time windows, one route at a time
    # Sort customers by due time for tie-breaking
    cust_sorted = sorted(customers, key=lambda v: (due[v], ready[v]))

    def build_route(greedy_order, max_customers=None):
        """Build a route by greedily inserting customers in order."""
        route = []
        t = 0.0
        batt = battery_cap
        total_dist = 0.0
        load = 0.0
        prev = 0
        remaining = list(greedy_order)
        if max_customers is not None:
            remaining = remaining[:max_customers]

        for node in remaining:
            # Try to insert node at current end
            d = dist[prev, node]
            new_batt = batt - d
            new_t = t + d / speed
            if new_batt < -1e-9:
                # Need to go to a station first
                # Find closest station
                best_station = None
                best_dist = float('inf')
                for s in stations:
                    ds = dist[prev, s]
                    if ds < best_dist:
                        best_dist = ds
                        best_station = s
                if best_station is None:
                    continue
                d_to_s = dist[prev, best_station]
                batt_s = batt - d_to_s
                t_s = t + d_to_s / speed
                if batt_s < -1e-9:
                    continue
                charge_time = (battery_cap - max(0, batt_s)) * recharge_rate
                t_s += charge_time
                batt_s = battery_cap
                d_s_to_node = dist[best_station, node]
                batt_node = batt_s - d_s_to_node
                t_node = t_s + d_s_to_node / speed
                if batt_node < -1e-9:
                    continue
                if t_node < ready[node]:
                    t_node = ready[node]
                if t_node > due[node] + 1e-9:
                    continue
                if load + demands[node] > capacity + 1e-9:
                    continue
                # Check if we can still return to depot
                d_back = dist[node, 0]
                batt_back = batt_node - d_back
                t_back = t_node + service[node] + d_back / speed
                if batt_back < -1e-9:
                    continue
                if t_back > horizon + 1e-9:
                    continue
                # Accept: add station and node
                route.append(best_station)
                route.append(node)
                t = t_node + service[node]
                batt = batt_node
                total_dist += d_to_s + d_s_to_node
                load += demands[node]
                prev = node
            else:
                if t < ready[node]:
                    new_t = ready[node]
                if new_t > due[node] + 1e-9:
                    continue
                if load + demands[node] > capacity + 1e-9:
                    continue
                # Check return
                d_back = dist[node, 0]
                batt_back = new_batt - d_back
                t_back = new_t + service[node] + d_back / speed
                if batt_back < -1e-9:
                    continue
                if t_back > horizon + 1e-9:
                    continue
                route.append(node)
                t = new_t + service[node]
                batt = new_batt
                total_dist += d
                load += demands[node]
                prev = node

        return route

    # Build initial solution
    routes = []
    unserved = set(cust_sorted)
    while unserved:
        if time.time() > deadline:
            break
        # Build a route starting from unserved customers
        route = build_route(list(unserved))
        if not route:
            # Force insert one customer
            for c in list(unserved):
                single = [c]
                feasible, _ = simulate(single)
                if feasible:
                    routes.append(single)
                    unserved.discard(c)
                    break
            else:
                # Should not happen
                break
        else:
            # Extract customers from route
            route_customers = [n for n in route if n not in stations]
            for c in route_customers:
                unserved.discard(c)
            routes.append(route)

    # Local search: 2-opt and swap within routes, and between routes
    def route_cost(route):
        if not route:
            return 0.0
        total = dist[0, route[0]]
        for i in range(len(route) - 1):
            total += dist[route[i], route[i+1]]
        total += dist[route[-1], 0]
        return total

    def is_route_feasible(route):
        feasible, _ = simulate(route)
        return feasible

    # Local search iterations
    improved = True
    max_iter = 500
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        improved = False

        # 2-opt within each route
        for ri, route in enumerate(routes):
            if len(route) < 2:
                continue
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if is_route_feasible(new_route):
                        old_cost = route_cost(route)
                        new_cost = route_cost(new_route)
                        if new_cost < old_cost - 1e-9:
                            routes[ri] = new_route
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

        # Swap between routes
        if not improved:
            for ri in range(len(routes)):
                for rj in range(ri + 1, len(routes)):
                    route_i = routes[ri]
                    route_j = routes[rj]
                    for ai in range(len(route_i)):
                        for aj in range(len(route_j)):
                            ni = route_i[ai]
                            nj = route_j[aj]
                            # Swap
                            new_i = route_i.copy()
                            new_j = route_j.copy()
                            new_i[ai], new_j[aj] = nj, ni
                            if is_route_feasible(new_i) and is_route_feasible(new_j):
                                old_cost = route_cost(route_i) + route_cost(route_j)
                                new_cost = route_cost(new_i) + route_cost(new_j)
                                if new_cost < old_cost - 1e-9:
                                    routes[ri] = new_i
                                    routes[rj] = new_j
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        if not improved:
            break

    # Final validation
    all_customers_served = set()
    for route in routes:
        for node in route:
            if node not in stations:
                all_customers_served.add(node)
    if all_customers_served != set(customers):
        # Fallback: one customer per route
        return [[c] for c in customers]

    for route in routes:
        if not is_route_feasible(route):
            return [[c] for c in customers]

    return routes
# EVOLVE-BLOCK-END

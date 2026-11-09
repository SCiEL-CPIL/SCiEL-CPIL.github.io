"""EV routing with time windows and battery; construction + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)

    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = _euclidean(coords[i], coords[j])
            dist[i, j] = d
            dist[j, i] = d

    customers = list(range(1, n_customers + 1))

    # Check if a route is feasible
    def check_route(route):
        """Returns (feasible, total_distance). Route is list of node ids (no depot at ends)."""
        if not route:
            return True, 0.0
        total_dist = 0.0
        time_cur = 0.0
        battery = battery_cap
        prev_node = 0
        for node in route:
            d = dist[prev_node, node]
            time_cur += d / speed
            battery -= d
            if battery < -1e-9:
                return False, 0.0
            if node in customers:
                time_cur = max(time_cur, ready[node])
                if time_cur > due[node] + 1e-9:
                    return False, 0.0
                time_cur += service[node]
            elif node in stations:
                # Recharge: time added = (battery) * recharge_rate / speed? 
                # "a station adds (battery - energy) * recharge_rate time and refills"
                # energy here is current battery. So time added = battery * recharge_rate
                # Wait, re-reading: "adds (battery - energy) * recharge_rate time"
                # If energy is the battery level, then (battery - energy) is 0?
                # Let's re-read carefully: "a station adds (battery - energy) * recharge_rate time and refills"
                # This is ambiguous. Usually recharge time = (full - current) / rate.
                # Let's assume "battery" in the formula means battery_cap and "energy" means current battery.
                # So time added = (battery_cap - battery) * recharge_rate
                time_cur += (battery_cap - battery) * recharge_rate
                battery = battery_cap
            total_dist += d
            prev_node = node
        # Return to depot
        d = dist[prev_node, 0]
        time_cur += d / speed
        battery -= d
        if battery < -1e-9:
            return False, 0.0
        if time_cur > horizon + 1e-9:
            return False, 0.0
        total_dist += d
        return True, total_dist

    # Greedy construction: insert customers into routes
    def construct():
        routes = []
        unvisited = set(customers)
        # Sort by due time
        for cust in sorted(customers, key=lambda x: due[x]):
            best_route = -1
            best_pos = -1
            best_dist = float('inf')
            # Try inserting into existing routes
            for r_idx, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [cust] + route[pos:]
                    # Check capacity
                    total_demand = sum(demands[n] for n in new_route)
                    if total_demand > capacity + 1e-9:
                        continue
                    feasible, d = check_route(new_route)
                    if feasible and d < best_dist:
                        best_dist = d
                        best_route = r_idx
                        best_pos = pos
            # If no feasible insertion, create new route
            if best_route == -1:
                routes.append([cust])
            else:
                routes[best_route].insert(best_pos, cust)
            unvisited.discard(cust)
        return routes

    def total_distance(routes):
        total = 0.0
        for route in routes:
            _, d = check_route(route)
            total += d
        return total

    # Local search: Or-opt and 2-opt
    def local_search(routes, time_limit):
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            # Try moving each customer to a better position
            for r_idx, route in enumerate(routes):
                for i, cust in enumerate(route):
                    # Try removing cust
                    new_route = route[:i] + route[i+1:]
                    # Try inserting into all routes at all positions
                    for r_idx2, route2 in enumerate(routes):
                        for pos in range(len(route2) + 1):
                            if r_idx2 == r_idx:
                                # Don't insert back at same position or adjacent (avoid no-ops)
                                if pos == i or pos == i + 1:
                                    continue
                                new_route2 = route2[:pos] + [cust] + route2[pos:]
                                new_routes = [r[:] for r in routes]
                                new_routes[r_idx] = new_route
                                new_routes[r_idx2] = new_route2
                                # Check feasibility
                                feasible = True
                                for r in new_routes:
                                    f, _ = check_route(r)
                                    if not f:
                                        feasible = False
                                        break
                                if feasible:
                                    new_total = total_distance(new_routes)
                                    old_total = total_distance(routes)
                                    if new_total < old_total - 1e-9:
                                        routes = new_routes
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            # Try 2-opt within each route
            for r_idx, route in enumerate(routes):
                if len(route) < 2:
                    continue
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        feasible, _ = check_route(new_route)
                        if feasible:
                            old_d = 0.0
                            _, old_d = check_route(route)
                            _, new_d = check_route(new_route)
                            if new_d < old_d - 1e-9:
                                routes[r_idx] = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # Initial construction
    routes = construct()
    routes = local_search(routes, deadline)

    # Try a few restarts with different construction orders if time permits
    best_routes = routes
    best_score = total_distance(routes)

    while time.time() < deadline - 5.0:
        # Perturb: shuffle some customers
        perturbed = [r[:] for r in best_routes]
        # Simple perturbation: swap two customers from different routes
        if len(perturbed) >= 2:
            r1 = np.random.randint(len(perturbed))
            r2 = np.random.randint(len(perturbed))
            if r1 != r2 and len(perturbed[r1]) > 0 and len(perturbed[r2]) > 0:
                i1 = np.random.randint(len(perturbed[r1]))
                i2 = np.random.randint(len(perturbed[r2]))
                perturbed[r1][i1], perturbed[r2][i2] = perturbed[r2][i2], perturbed[r1][i1]
        
        # Reconstruct by inserting unvisited? No, just local search from perturbed
        # But we need to ensure all customers are present exactly once
        # Let's just do local search from the perturbed state
        new_routes = local_search(perturbed, deadline - 2.0)
        new_score = total_distance(new_routes)
        if new_score < best_score - 1e-9:
            best_routes = new_routes
            best_score = new_score

    return best_routes
# EVOLVE-BLOCK-END

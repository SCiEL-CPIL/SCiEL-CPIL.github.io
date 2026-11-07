"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def plan(instance):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dist[i, j] = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)

    customers = list(range(1, n_customers + 1))

    # Helper to simulate a route with charging
    def simulate_route(route):
        """
        route: list of node ids (customers and possibly stations)
        Returns (total_distance, is_valid)
        """
        if not route:
            return 0.0, True

        t = 0.0
        battery = battery_max
        total_dist = 0.0
        current_node = 0  # depot

        for node in route:
            d = dist[current_node, node]
            battery -= d
            t += d / speed
            total_dist += d

            if battery < -1e-9:
                return total_dist, False

            # Arrive at node
            if node in customers_set:
                # Customer node
                if t > due[node] + 1e-9:
                    return total_dist, False
                t = max(t, ready[node])
                t += service[node]
                battery -= 0  # no battery use for service, but demand is handled by capacity
            else:
                # Station or depot
                if node in stations_set:
                    # Charge
                    charge_time = max(0, (battery_max - battery)) * recharge_rate
                    t += charge_time
                    battery = battery_max
                # If depot (shouldn't happen mid-route in our representation)
                # Actually, stations are separate nodes

            current_node = node

        # Return to depot
        d = dist[current_node, 0]
        battery -= d
        t += d / speed
        total_dist += d

        if battery < -1e-9:
            return total_dist, False
        if t > horizon + 1e-9:
            return total_dist, False

        return total_dist, True

    customers_set = set(customers)
    stations_set = set(stations)

    # Step 1: Generate initial routes using Clarke-Wright savings
    # Start with each customer in its own route
    routes = [[c] for c in customers]
    route_of = {c: i for i, c in enumerate(customers)}
    route_index = {i: i for i in range(len(customers))}

    # Compute savings for merging two routes
    # Savings of merging route A ending with a and route B starting with b:
    # dist(depot, a) + dist(depot, b) - dist(a, b)
    # We need to consider the order. For simplicity, we'll use a greedy approach.

    # Actually, let's use a simpler approach:
    # 1. Start with individual routes
    # 2. Repeatedly find the best merge (max savings) that is feasible
    # 3. Merge them

    # For efficiency, let's precompute potential savings
    # But with 25 customers, we can afford O(n^2) per iteration

    def get_route_savings(r1, r2):
        """
        Compute savings of merging r1 and r2.
        r1 and r2 are lists of node ids.
        We can merge in two ways:
        1. r1 followed by r2: savings = dist(depot, r1[-1]) + dist(depot, r2[0]) - dist(r1[-1], r2[0])
        2. r2 followed by r1: savings = dist(depot, r2[-1]) + dist(depot, r1[0]) - dist(r2[-1], r1[0])
        Return the max savings and the merged route.
        """
        if not r1 or not r2:
            return 0.0, None

        # Check capacity
        demand_r1 = sum(demands[node] for node in r1 if node in customers_set)
        demand_r2 = sum(demands[node] for node in r2 if node in customers_set)
        if demand_r1 + demand_r2 > capacity:
            return 0.0, None

        # Option 1: r1 + r2
        s1 = dist[0, r1[-1]] + dist[0, r2[0]] - dist[r1[-1], r2[0]]
        merged1 = r1 + r2

        # Option 2: r2 + r1
        s2 = dist[0, r2[-1]] + dist[0, r1[0]] - dist[r2[-1], r1[0]]
        merged2 = r2 + r1

        if s1 >= s2:
            # Check feasibility of merged1
            total_dist, valid = simulate_route(merged1)
            if valid:
                return s1, merged1
        if s2 > s1:
            # Check feasibility of merged2
            total_dist, valid = simulate_route(merged2)
            if valid:
                return s2, merged2

        return 0.0, None

    # Greedy merging
    merged = [False] * len(customers)
    active_routes = {i: routes[i] for i in range(len(customers))}

    for _ in range(len(customers) - 1):  # At most n-1 merges
        best_saving = 0.0
        best_pair = None
        best_merged = None

        for i in range(len(customers)):
            if merged[i]:
                continue
            for j in range(i + 1, len(customers)):
                if merged[j]:
                    continue

                r1 = active_routes[i]
                r2 = active_routes[j]

                s, merged_route = get_route_savings(r1, r2)
                if s > best_saving:
                    best_saving = s
                    best_pair = (i, j)
                    best_merged = merged_route

        if best_pair is None:
            break

        i, j = best_pair
        active_routes[i] = best_merged
        merged[j] = True

    # Collect final routes
    final_routes = []
    for i in range(len(customers)):
        if not merged[i]:
            final_routes.append(active_routes[i])

    # Step 2: Try to improve by inserting charging stations
    # For each route, try inserting stations at various positions
    # This is a local search step

    def try_insert_stations(route):
        """
        Try to improve a route by inserting charging stations.
        Returns the best route found.
        """
        best_route = route
        best_dist, best_valid = simulate_route(route)
        if not best_valid:
            return None

        # Try inserting each station at each position
        for station in stations:
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [station] + route[pos:]
                d, valid = simulate_route(new_route)
                if valid and d < best_dist:
                    best_dist = d
                    best_route = new_route

        return best_route

    # Apply station insertion to each route
    improved_routes = []
    for route in final_routes:
        improved = try_insert_stations(route)
        if improved is not None:
            improved_routes.append(improved)
        else:
            improved_routes.append(route)

    # Step 3: Local search with 2-opt
    # Try to improve each route with 2-opt moves

    def two_opt_route(route):
        """
        Apply 2-opt to a route to improve distance.
        """
        if len(route) < 2:
            return route

        best_route = route
        best_dist, best_valid = simulate_route(route)
        if not best_valid:
            return None

        improved = True
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    # Reverse segment i+1..j
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    d, valid = simulate_route(new_route)
                    if valid and d < best_dist - 1e-9:
                        best_dist = d
                        best_route = new_route
                        improved = True

        return best_route

    # Apply 2-opt to each route
    optimized_routes = []
    for route in improved_routes:
        optimized = two_opt_route(route)
        if optimized is not None:
            optimized_routes.append(optimized)
        else:
            optimized_routes.append(route)

    # Step 4: Cross-route improvements (Or-opt moves)
    # Try moving a customer from one route to another

    def or_opt_move(routes):
        """
        Try Or-opt moves: move a customer (or small segment) from one route to another.
        """
        n = len(routes)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    for pos in range(len(routes[i])):
                        customer = routes[i][pos]
                        # Remove from route i
                        new_route_i = routes[i][:pos] + routes[i][pos+1:]
                        # Try inserting into route j at each position
                        for pos_j in range(len(routes[j]) + 1):
                            new_route_j = routes[j][:pos_j] + [customer] + routes[j][pos_j:]

                            # Check feasibility
                            d_i, valid_i = simulate_route(new_route_i)
                            d_j, valid_j = simulate_route(new_route_j)

                            if valid_i and valid_j:
                                old_d, old_valid = simulate_route(routes[i])
                                old_d2, old_valid2 = simulate_route(routes[j])
                                if old_valid and old_valid2:
                                    if d_i + d_j < old_d + old_d2 - 1e-9:
                                        routes[i] = new_route_i
                                        routes[j] = new_route_j
                                        improved = True

        return routes

    # Apply Or-opt moves
    final_routes = or_opt_move(optimized_routes)

    # Final validation
    total_distance = 0.0
    for route in final_routes:
        d, valid = simulate_route(route)
        if not valid:
            # Fallback: return individual routes
            return [[c] for c in customers]
        total_distance += d

    return final_routes

# EVOLVE-BLOCK-END

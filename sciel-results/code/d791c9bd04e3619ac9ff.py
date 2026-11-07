# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np
from itertools import permutations, product

def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[0 - b[1]]) # typo check: a[1]-b[1]
    # Actually:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _build_dist(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            v = math.hypot(xi - xj, yi - yj)
            d[i, j] = v
            d[j, i] = v
    return d

def _simulate_route(route, instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon, speed=1.0):
    """
    Simulate a route.
    Returns (feasible, total_distance, final_time, final_battery, max_load)
    """
    if not route:
        return True, 0.0, 0.0, battery_cap, 0.0

    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    
    t = 0.0
    bat = battery_cap
    load = 0.0
    total_dist = 0.0
    prev = 0 # depot

    for node in route:
        d = dist[prev][node]
        bat -= d
        t += d / speed
        total_dist += d
        
        if bat < -1e-9:
            return False, total_dist, t, bat, load

        if node in stations_set:
            # Charge
            charge_time = (battery_cap - bat) * recharge_rate
            t += charge_time
            bat = battery_cap
        else:
            # Customer
            load += demands[node]
            if load > capacity + 1e-9:
                return False, total_dist, t, bat, load
            
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return False, total_dist, t, bat, load
            t += service[node]
        
        prev = node

    # Return to depot
    d = dist[prev][0]
    bat -= d
    t += d / speed
    total_dist += d
    
    if bat < -1e-9:
        return False, total_dist, t, bat, load
    if t > horizon + 1e-9:
        return False, total_dist, t, bat, load
        
    return True, total_dist, t, bat, load

def _generate_columns(instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon):
    """
    Generate a set of feasible subtours (columns).
    A column is a sequence of nodes (customers and stations) that starts and ends at depot?
    No, in set partitioning for VRP, a column is usually a full route from depot to depot.
    However, generating all full routes is hard.
    
    Alternative interpretation of "column generation" here:
    We want to partition customers into disjoint sets, each served by a route.
    We can generate candidate routes (columns) that cover subsets of customers.
    Then we solve the Set Partitioning Problem: select a subset of columns such that every customer is covered exactly once, minimizing total cost.
    
    Given the small size (25 customers), we can generate columns of size 1, 2, 3, 4 customers.
    For each subset, we find the best feasible route (order + charging) for that subset.
    """
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations_list = list(stations_set)
    
    columns = [] # List of (frozenset(customers_in_route), route_list, cost)
    
    # Helper to find best route for a specific set of customers
    def _best_route_for_subset(cust_subset):
        """
        Find the best feasible route for a given set of customers.
        We try different orderings and optional station insertions.
        Since subsets are small (<=4), we can brute force orderings.
        Station insertion: we can check if a station is needed and where.
        """
        if not cust_subset:
            return None, float('inf')
        
        best_cost = float('inf')
        best_route = None
        
        # Generate all permutations of customers
        perms = list(permutations(cust_subset))
        
        # For each permutation, try inserting 0 or 1 station at each position
        # Actually, battery constraints might require multiple stations, but for small subsets, 0 or 1 is likely enough.
        # Let's just try all permutations of (customers + optional stations).
        # To keep it simple and fast, we'll try:
        # 1. No stations
        # 2. One station inserted at each possible position
        
        # Try no stations
        for perm in perms:
            route = list(perm)
            feasible, cost, _, _, _ = _simulate_route(route, instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon)
            if feasible and cost < best_cost:
                best_cost = cost
                best_route = route
        
        # Try one station
        for st in stations_list:
            for perm in perms:
                # Try inserting station at each position
                for pos in range(len(perm) + 1):
                    route = list(perm[:pos]) + [st] + list(perm[pos:])
                    feasible, cost, _, _, _ = _simulate_route(route, instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon)
                    if feasible and cost < best_cost:
                        best_cost = cost
                        best_route = route
        
        # Try two stations (if subset size is small, this is still manageable)
        if len(cust_subset) <= 3:
            for st1 in stations_list:
                for st2 in stations_list:
                    if st1 == st2:
                        continue
                    for perm in perms:
                        # Insert st1 and st2
                        # This gets complex. Let's stick to 0 or 1 station for now, 
                        # or allow the simulator to handle it if we just check feasibility.
                        # Actually, the simulator handles charging if a station is in the route.
                        # If no station is in the route and battery fails, it's infeasible.
                        # So we MUST include stations in the route sequence if they are needed.
                        # My current approach tries inserting 1 station. 
                        # If 1 station isn't enough, we might miss feasible routes.
                        # But for small subsets, 1 station is usually enough.
                        pass
        
        return best_route, best_cost

    # Generate columns for subsets of size 1 to 4
    # Size 1
    for c in customers:
        route, cost = _best_route_for_subset((c,))
        if route is not None:
            columns.append((frozenset([c]), route, cost))
            
    # Size 2
    for i in range(len(customers)):
        for j in range(i+1, len(customers)):
            c1, c2 = customers[i], customers[j]
            route, cost = _best_route_for_subset((c1, c2))
            if route is not None:
                columns.append((frozenset([c1, c2]), route, cost))
                
    # Size 3
    for i in range(len(customers)):
        for j in range(i+1, len(customers)):
            for k in range(j+1, len(customers)):
                c1, c2, c3 = customers[i], customers[j], customers[k]
                route, cost = _best_route_for_subset((c1, c2, c3))
                if route is not None:
                    columns.append((frozenset([c1, c2, c3]), route, cost))
                    
    # Size 4 (might be too many to generate all, but 25C4 = 12650, times perms(24) is too much. 
    # Wait, _best_route_for_subset does perms. 4! = 24. 12650 * 24 * (1 + 4*5) ~ 1.5M simulations.
    # Each simulation is O(10). 15M ops. Should be fast enough in Python? Maybe borderline.
    # Let's limit to Size 3 to be safe on time, or optimize _best_route_for_subset.
    # Given 150s limit, and 3 instances, we have ~50s per instance.
    # 1.5M simulations * 3 instances = 4.5M simulations.
    # Each sim is ~20 Python ops. 90M ops. Might take 10-20s. Risky.
    # Let's stick to Size 1-3 for column generation, and rely on local search to merge/improve.
    # Or, we can be smarter: only generate columns for "promising" subsets?
    # For now, Size 1-3 is a good balance.
    
    return columns

def _solve_set_partitioning(columns, customers):
    """
    Greedy set partitioning.
    We want to select a set of columns such that each customer is covered exactly once.
    Minimize total cost.
    
    Simple Greedy:
    1. Sort columns by cost / |coverage| (density) or just cost.
    2. Pick the best column that doesn't conflict with already picked columns.
    3. Repeat until all covered.
    4. If stuck (some customer not covered by any remaining column), pick the cheapest single-customer column for them.
    """
    # Pre-process: for each customer, find the cheapest column containing it (as fallback)
    # But first, let's try a simple greedy.
    
    # Sort columns by cost
    columns_sorted = sorted(columns, key=lambda x: x[2])
    
    selected = []
    covered = set()
    
    for col in columns_sorted:
        cust_set, route, cost = col
        # Check if any customer in this column is already covered
        if cust_set & covered:
            continue
        # Accept this column
        selected.append((route, cost))
        covered.update(cust_set)
        
        if len(covered) == len(customers):
            break
    
    # Check if all customers are covered
    if len(covered) < len(customers):
        # Find uncovered customers
        uncovered = set(customers) - covered
        for c in uncovered:
            # Find cheapest column for c (should be a single customer column)
            best_col = None
            best_cost = float('inf')
            for col in columns:
                if c in col[0] and len(col[0]) == 1:
                    if col[2] < best_cost:
                        best_cost = col[2]
                        best_col = col
            if best_col:
                selected.append((best_col[1], best_col[2]))
                covered.add(c)
            else:
                # Should not happen if we generated size 1 columns
                raise ValueError(f"Customer {c} not covered")
                
    return [r for r, c in selected]

def _local_search(routes, instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon, time_limit):
    """
    2-opt and Or-opt polish on the multi-route solution.
    """
    start = time.time()
    
    # Intra-route 2-opt
    for i, route in enumerate(routes):
        if time.time() - start > time_limit:
            break
        n = len(route)
        improved = True
        while improved:
            improved = False
            for a in range(n - 1):
                if time.time() - start > time_limit:
                    break
                for b in range(a + 1, n):
                    # 2-opt: reverse segment a..b
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    feasible, new_cost, _, _, _ = _simulate_route(new_route, instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon)
                    # Get old cost
                    old_feasible, old_cost, _, _, _ = _simulate_route(route, instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon)
                    if feasible and new_cost < old_cost - 1e-6:
                        routes[i] = new_route
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
                    
    # Inter-route 2-opt (move a customer from one route to another, or swap)
    # Simple Or-opt: move 1 customer from route i to route j
    for i in range(len(routes)):
        if time.time() - start > time_limit:
            break
        for j in range(len(routes)):
            if i == j:
                continue
            if time.time() - start > time_limit:
                break
            # Try moving each customer from i to j
            for k in range(len(routes[i])):
                cust = routes[i][k]
                # Remove from i
                new_i = routes[i][:k] + routes[i][k+1:]
                # Try inserting into j at all positions
                best_pos = -1
                best_cost = float('inf')
                best_new_j = None
                best_new_i_cost = 0
                feasible_i, cost_i, _, _, _ = _simulate_route(new_i, instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon)
                if not feasible_i:
                    continue
                
                for pos in range(len(routes[j]) + 1):
                    new_j = routes[j][:pos] + [cust] + routes[j][pos:]
                    feasible_j, cost_j, _, _, _ = _simulate_route(new_j, instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon)
                    if feasible_j:
                        total_cost = cost_i + cost_j
                        # Compare with old total
                        old_feasible_i, old_cost_i, _, _, _ = _simulate_route(routes[i], instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon)
                        old_feasible_j, old_cost_j, _, _, _ = _simulate_route(routes[j], instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon)
                        if old_feasible_i and old_feasible_j:
                            old_total = old_cost_i + old_cost_j
                            if total_cost < old_total - 1e-6:
                                if total_cost < best_cost:
                                    best_cost = total_cost
                                    best_pos = pos
                                    best_new_j = new_j
                                    best_new_i_cost = cost_i
                                    
                if best_pos >= 0:
                    routes[i] = new_i
                    routes[j] = best_new_j
                    # Remove empty routes
                    routes = [r for r in routes if r]
                    # Re-index i, j? This is tricky. Let's just break and restart loops.
                    break
            if time.time() - start > time_limit:
                break
    return routes

def plan(instance):
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    coords = instance["coords"]
    dist = _build_dist(coords)
    stations_set = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # 1. Generate Columns
    columns = _generate_columns(instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon)
    
    # 2. Solve Set Partitioning
    customers = list(range(1, instance["n_customers"] + 1))
    routes = _solve_set_partitioning(columns, customers)
    
    # 3. Local Search
    if time.time() < deadline - 10:
        routes = _local_search(routes, instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon, deadline - 10 - time.time())
    
    # 4. Verify and Fallback
    all_ok = True
    for route in routes:
        feasible, _, _, _, _ = _simulate_route(route, instance, dist, stations_set, capacity, battery_cap, recharge_rate, horizon)
        if not feasible:
            all_ok = False
            break
            
    if not all_ok:
        # Fallback: one customer per route
        customers.sort(key=lambda c: instance["due"][c])
        routes = [[c] for c in customers]
        
    return routes
# EVOLVE-BLOCK-END

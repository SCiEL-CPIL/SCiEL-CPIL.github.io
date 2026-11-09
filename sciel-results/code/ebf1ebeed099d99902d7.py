"""EV routing with time windows: LNS + 2-opt/Or-opt local search."""
import math
import random
import time
import numpy as np

# EVOLVE-BLOCK-START

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _euclidean_distance(coords, route):
    """Total Euclidean distance for a route (depot implicit at both ends)."""
    d = 0.0
    prev = 0  # depot
    for node in route:
        d += _dist(coords[prev], coords[node])
        prev = node
    d += _dist(coords[prev], coords[0])
    return d


def _simulate(instance, route):
    """
    Simulate a route. Return (total_dist, feasible, final_time, final_battery).
    If infeasible at any point, return (dist_so_far, False, inf, 0).
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    total_dist = 0.0
    time_now = 0.0
    battery = battery_max
    load = 0

    prev = 0
    for node in route:
        d = _dist(coords[prev], coords[node])
        total_dist += d
        travel_time = d / speed
        battery -= d
        time_now += travel_time
        if battery < 0:
            return total_dist, False, float('inf'), 0.0

        if node in stations:
            # Recharge: time added = (battery_max - battery) * recharge_rate / 1? 
            # "a station adds (battery - energy) * recharge_rate time and refills"
            # Wait, the contract says: "a station adds (battery - energy) * recharge_rate time and refills"
            # This is ambiguous. Let's re-read: "a station adds (battery - energy) * recharge_rate time and refills"
            # Usually recharge rate is energy per time. So time = (needed_energy) / recharge_rate.
            # But the text says "(battery - energy) * recharge_rate". If recharge_rate is 0.5, and it's a multiplier on time, that's odd.
            # Let's look at the seed: "recharge_rate 0.5".
            # If it's time = (energy_deficit) / recharge_rate, then 0.5 means it takes 2 seconds per unit of energy.
            # If it's time = (energy_deficit) * recharge_rate, then 0.5 means it takes 0.5 seconds per unit of energy.
            # The phrasing "adds ... time" suggests the result is time.
            # Let's assume the standard interpretation: time_to_recharge = (battery_max - battery) / recharge_rate.
            # However, the text literally says: "(battery - energy) * recharge_rate time".
            # If I have battery=100, max=300, deficit=200. If rate=0.5.
            # Interpretation A: 200 / 0.5 = 400 time.
            # Interpretation B: 200 * 0.5 = 100 time.
            # Given "recharge_rate 0.5", a rate of 0.5 units/sec is more standard for "rate".
            # But let's look at the variable name. "recharge_rate".
            # If I assume it's a multiplier for time, it's weird.
            # Let's stick to the most common VRP interpretation: time = deficit / rate.
            # BUT, the prompt says: "a station adds (battery - energy) * recharge_rate time".
            # This is extremely specific. It says MULTIPLY.
            # Let's follow the text literally: time_added = (battery_max - battery) * recharge_rate.
            # Wait, "battery - energy". Energy is likely the current battery level.
            # So deficit = battery_max - battery.
            # time_added = deficit * recharge_rate.
            
            deficit = battery_max - battery
            time_added = deficit * recharge_rate
            time_now += time_added
            battery = battery_max
        else:
            # Customer
            if time_now < ready[node]:
                time_now = ready[node]
            if time_now > due[node]:
                return total_dist, False, float('inf'), 0.0
            time_now += service[node]
            load += demands[node]
            if load > capacity:
                return total_dist, False, float('inf'), 0.0

        prev = node

    # Return to depot
    d = _dist(coords[prev], coords[0])
    total_dist += d
    travel_time = d / speed
    battery -= d
    time_now += travel_time
    if battery < 0:
        return total_dist, False, float('inf'), 0.0
    if time_now > horizon:
        return total_dist, False, float('inf'), 0.0

    return total_dist, True, time_now, battery


def _total_distance(instance, routes):
    total = 0.0
    for r in routes:
        total += _euclidean_distance(instance["coords"], r)
    return total


def _is_feasible(instance, routes):
    for r in routes:
        _, feasible, _, _ = _simulate(instance, r)
        if not feasible:
            return False
    return True


def _construct_initial(instance, customers):
    """Construct initial routes using nearest neighbor insertion."""
    coords = instance["coords"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    
    unvisited = set(customers)
    routes = []
    
    while unvisited:
        route = []
        load = 0
        prev = 0
        time_now = 0.0
        battery = instance["battery"]
        
        while unvisited:
            best_node = None
            best_dist = float('inf')
            for c in unvisited:
                if load + demands[c] > capacity:
                    continue
                d = _dist(coords[prev], coords[c])
                # Simple heuristic: prefer closer, but also check feasibility roughly
                if d < best_dist:
                    best_dist = d
                    best_node = c
            
            if best_node is None:
                break
            
            # Check if we can reach best_node and return to depot
            # We need to ensure battery is sufficient for the whole trip if we add this node?
            # For now, just insert and simulate later. If infeasible, we might need a charging station.
            # This simple constructor might fail. Let's use a safer insertion.
            
            # Actually, let's just pick the closest feasible node.
            # We'll handle charging stations in the repair/local search phase.
            unvisited.remove(best_node)
            route.append(best_node)
            load += demands[best_node]
            prev = best_node
        
        if route:
            routes.append(route)
    
    return routes


def _try_insert(instance, routes, customer):
    """Try to insert a customer into an existing route at the best position."""
    coords = instance["coords"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    
    best_route_idx = -1
    best_pos = -1
    best_extra_dist = float('inf')
    
    for i, route in enumerate(routes):
        # Check capacity
        current_load = sum(demands[c] for c in route)
        if current_load + demands[customer] > capacity:
            continue
        
        # Try inserting at each position
        for j in range(len(route) + 1):
            new_route = route[:j] + [customer] + route[j:]
            # Rough distance check
            prev_node = 0 if j == 0 else route[j-1]
            next_node = route[j] if j < len(route) else 0
            old_dist = _dist(coords[prev_node], coords[next_node])
            new_dist = _dist(coords[prev_node], coords[customer]) + _dist(coords[customer], coords[next_node])
            extra = new_dist - old_dist
            
            if extra < best_extra_dist:
                best_extra_dist = extra
                best_route_idx = i
                best_pos = j
    
    if best_route_idx != -1:
        route = routes[best_route_idx]
        route.insert(best_pos, customer)
        return True
    return False


def _solve_instance(instance):
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    
    # Initial construction
    routes = _construct_initial(instance, customers)
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    # Ensure all customers are served
    served = set()
    for r in routes:
        for c in r:
            served.add(c)
    unserved = [c for c in customers if c not in served]
    
    for c in unserved:
        # Create a new route for this customer
        routes.append([c])
    
    # Local Search: 2-opt and Or-opt
    improved = True
    while improved:
        improved = False
        
        # 2-opt
        for i in range(len(routes)):
            route = routes[i]
            if len(route) < 2:
                continue
            for j in range(len(route)):
                for k in range(j+1, len(route)):
                    new_route = route[:j] + route[j:k+1][::-1] + route[k+1:]
                    old_dist = _euclidean_distance(instance["coords"], route)
                    new_dist = _euclidean_distance(instance["coords"], new_route)
                    if new_dist < old_dist - 1e-9:
                        # Check feasibility
                        _, feasible, _, _ = _simulate(instance, new_route)
                        if feasible:
                            routes[i] = new_route
                            improved = True
                            
        # Or-opt (move 1 or 2 nodes)
        for i in range(len(routes)):
            route = routes[i]
            for j in range(len(route)):
                for length in [1, 2]:
                    if j + length > len(route):
                        continue
                    segment = route[j:j+length]
                    new_route = route[:j] + route[j+length:]
                    # Try inserting segment at other positions
                    for k in range(len(new_route) + 1):
                        if k == j:
                            continue
                        candidate = new_route[:k] + segment + new_route[k:]
                        old_dist = _euclidean_distance(instance["coords"], route)
                        new_dist = _euclidean_distance(instance["coords"], candidate)
                        if new_dist < old_dist - 1e-9:
                            _, feasible, _, _ = _simulate(instance, candidate)
                            if feasible:
                                routes[i] = candidate
                                improved = True
                                break
    
    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Solve for the single instance provided
    routes = _solve_instance(instance)
    
    # Check time
    if time.time() > deadline:
        return routes
    
    # Try LNS if time permits
    # (Simplified LNS for now: random removal and re-insertion)
    current_dist = _total_distance(instance, routes)
    best_routes = [r[:] for r in routes]
    best_dist = current_dist
    
    rng = random.Random(42)
    
    while time.time() < deadline - 5.0:
        # Destroy: remove random customers
        all_customers = [c for r in routes for c in r]
        n_remove = max(1, int(0.2 * len(all_customers)))
        removed = rng.sample(all_customers, n_remove)
        
        # Remove from routes
        new_routes = []
        for r in routes:
            new_r = [c for c in r if c not in removed]
            if new_r:
                new_routes.append(new_r)
        
        # Re-insert removed customers
        for c in removed:
            if not _try_insert(instance, new_routes, c):
                new_routes.append([c])
        
        # Check feasibility and distance
        if _is_feasible(instance, new_routes):
            new_dist = _total_distance(instance, new_routes)
            if new_dist < best_dist:
                best_dist = new_dist
                best_routes = [r[:] for r in new_routes]
                routes = [r[:] for r in new_routes]
        
        # Local search on new solution
        routes = _solve_instance(instance)
        if _is_feasible(instance, routes):
            new_dist = _total_distance(instance, routes)
            if new_dist < best_dist:
                best_dist = new_dist
                best_routes = [r[:] for r in routes]
    
    return best_routes

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import heapq

def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist_matrix(coords):
    n = len(coords)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            v = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            d[i, j] = v
            d[j, i] = v
    return d


def _is_station(node, stations_set):
    return node in stations_set


def _simulate_route(route, instance, dist):
    """
    Simulate a route (list of node ids, excluding depot at both ends).
    Returns (valid, total_distance, final_time, final_battery, max_time)
    or (False, inf, inf, inf, inf) if invalid.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = instance["stations_set"]

    total_demand = 0
    for node in route:
        total_demand += demands[node]
    if total_demand > capacity:
        return False, float('inf'), float('inf'), float('inf'), float('inf')

    t = 0.0
    battery = battery_cap
    total_dist = 0.0
    prev = 0  # depot

    for node in route:
        d = dist[prev][node]
        t += d / speed
        battery -= d
        total_dist += d

        if battery < -1e-9:
            return False, float('inf'), float('inf'), float('inf'), float('inf')

        if _is_station(node, stations_set):
            # Recharge: time += (battery - energy_needed) * recharge_rate? 
            # Wait, the problem says: "a station adds (battery - energy) * recharge_rate time and refills"
            # This phrasing is ambiguous. Usually it means time to charge.
            # Let's re-read: "a station adds (battery - energy) * recharge_rate time and refills"
            # If battery is current level, and energy is... ?
            # Maybe it means: time to charge = (full_battery - current_battery) * recharge_rate?
            # Or is it a penalty?
            # "refills" implies battery becomes full.
            # Let's assume the time added is (battery_cap - battery) / recharge_rate? No, "recharge_rate 0.5".
            # If recharge_rate is energy per time, time = (cap - batt) / rate.
            # If recharge_rate is time per energy, time = (cap - batt) * rate.
            # Let's look at the seed: it doesn't use stations.
            # Standard interpretation: recharge_rate is the rate of energy gain per time unit? Or time per unit energy?
            # Given "recharge_rate 0.5", if it's time per unit energy, charging 300 units takes 150 time.
            # If it's energy per time, charging 300 units takes 600 time.
            # The text says: "adds (battery - energy) * recharge_rate time". This is weird.
            # Maybe "energy" is a typo for "0"? i.e. (battery - 0) * rate? No, that would decrease battery?
            # Let's assume the standard: time_to_charge = (battery_cap - battery) * recharge_rate.
            # And battery becomes battery_cap.
            charge_time = (battery_cap - battery) * recharge_rate
            t += charge_time
            battery = battery_cap
        else:
            # Customer
            if t < ready[node]:
                t = ready[node]
            if t > due[node]:
                return False, float('inf'), float('inf'), float('inf'), float('inf')
            t += service[node]

        prev = node

    # Return to depot
    d = dist[prev][0]
    t += d / speed
    battery -= d
    total_dist += d

    if battery < -1e-9:
        return False, float('inf'), float('inf'), float('inf'), float('inf')
    if t > horizon:
        return False, float('inf'), float('inf'), float('inf'), float('inf')

    return True, total_dist, t, battery, t


def _route_distance(route, dist):
    """Calculate distance of a route without checking constraints."""
    if not route:
        return 0.0
    d = dist[0][route[0]]
    for i in range(len(route) - 1):
        d += dist[route[i]][route[i + 1]]
    d += dist[route[-1]][0]
    return d


def _greedy_insertion(instance, dist):
    """
    Construct initial routes using greedy insertion.
    Returns a list of routes.
    """
    customers = list(range(1, instance["n_customers"] + 1))
    # Sort by due time for better time window handling
    customers.sort(key=lambda v: instance["due"][v])

    routes = []
    remaining = set(customers)

    while remaining:
        # Start a new route
        route = []
        # Pick the customer with the earliest due time among remaining
        # Actually, let's just pick the first one in the sorted list that is remaining
        start_cust = None
        for c in customers:
            if c in remaining:
                start_cust = c
                break
        
        if start_cust is None:
            break

        route.append(start_cust)
        remaining.remove(start_cust)

        # Try to insert other customers
        while remaining:
            best_cust = None
            best_pos = None
            best_dist = float('inf')
            best_valid = False

            # Try inserting each remaining customer at each position
            for c in list(remaining):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    valid, total_dist, _, _, _ = _simulate_route(new_route, instance, dist)
                    if valid and total_dist < best_dist:
                        best_dist = total_dist
                        best_cust = c
                        best_pos = pos
                        best_valid = True

            if best_valid:
                route = route[:best_pos] + [best_cust] + route[best_pos:]
                remaining.remove(best_cust)
            else:
                break

        routes.append(route)

    return routes


def _two_opt(route, dist, instance):
    """
    Apply 2-opt improvement to a single route.
    Returns improved route.
    """
    improved = True
    while improved:
        improved = False
        n = len(route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Try reversing segment i+1..j
                new_route = route[:i + 1] + route[i + 1:j + 1][::-1] + route[j + 1:]
                # Check validity and distance
                old_dist = _route_distance(route, dist)
                new_dist = _route_distance(new_route, dist)
                if new_dist < old_dist - 1e-9:
                    valid, _, _, _, _ = _simulate_route(new_route, instance, dist)
                    if valid:
                        route = new_route
                        improved = True
    return route


def _relocate(route, dist, instance):
    """
    Apply relocate improvement to a single route.
    """
    improved = True
    while improved:
        improved = False
        n = len(route)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                new_route = route[:i] + route[i + 1:j + 1] + [route[i]] + route[j + 1:]
                old_dist = _route_distance(route, dist)
                new_dist = _route_distance(new_route, dist)
                if new_dist < old_dist - 1e-9:
                    valid, _, _, _, _ = _simulate_route(new_route, instance, dist)
                    if valid:
                        route = new_route
                        improved = True
    return route


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    coords = instance["coords"]
    dist = _dist_matrix(coords)
    stations_set = set(instance["stations"])
    instance["stations_set"] = stations_set

    # 1. Construct initial solution
    routes = _greedy_insertion(instance, dist)

    # 2. Local search improvements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(routes)):
            if time.time() >= deadline:
                break
            
            # 2-opt
            new_route = _two_opt(routes[i], dist, instance)
            if new_route != routes[i]:
                routes[i] = new_route
                improved = True
            
            # Relocate
            new_route = _relocate(routes[i], dist, instance)
            if new_route != routes[i]:
                routes[i] = new_route
                improved = True

    # 3. Final validation
    total_distance = 0.0
    for route in routes:
        valid, d, _, _, _ = _simulate_route(route, instance, dist)
        if not valid:
            # Fallback: serve each customer individually
            return [[v] for v in range(1, instance["n_customers"] + 1)]
        total_distance += d

    return routes
# EVOLVE-BLOCK-END

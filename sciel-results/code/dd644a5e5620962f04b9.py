# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt
from itertools import combinations

def _dist(coords, i, j):
    dx = coords[i, 0] - coords[j, 0]
    dy = coords[i, 1] - coords[j, 1]
    return sqrt(dx * dx + dy * dy)

def _solve_instance(instance, time_limit):
    coords = np.array(instance["coords"])
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distances
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = _dist(coords, i, j)
            dist[i, j] = d
            dist[j, i] = d
    
    customers = list(range(1, n_customers + 1))
    
    # Sort customers by due time for initial ordering
    customers_sorted = sorted(customers, key=lambda v: due[v])
    
    # Initial solution: one customer per route
    initial_routes = [[v] for v in customers_sorted]
    initial_cost = sum(dist[0, r[0]] + dist[r[0], 0] for r in initial_routes)
    
    best_routes = initial_routes
    best_cost = initial_cost
    
    start_time = time.time()
    
    # Local search: try to merge routes or improve within routes
    # We'll use a simple 2-opt like improvement + route merging
    
    def check_route(route):
        """Check if a route is feasible. Returns True if feasible."""
        if not route:
            return True
        load = sum(demands[v] for v in route)
        if load > capacity:
            return False
        
        t = 0.0
        b = battery
        prev = 0
        for v in route:
            d = dist[prev, v]
            t += d / speed
            b -= d
            if b < -1e-9:
                return False
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return False
            t += service[v]
            b += 0  # no recharge at customer
            prev = v
        
        # Return to depot
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -1e-9:
            return False
        if t > horizon + 1e-9:
            return False
        return True
    
    def route_cost(route):
        """Calculate distance cost of a route."""
        if not route:
            return 0.0
        cost = 0.0
        prev = 0
        for v in route:
            cost += dist[prev, v]
            prev = v
        cost += dist[prev, 0]
        return cost
    
    def total_cost(routes):
        return sum(route_cost(r) for r in routes)
    
    # Try 2-opt within each route
    def two_opt_improve(routes):
        improved = True
        while improved:
            improved = False
            for ri in range(len(routes)):
                route = routes[ri]
                if len(route) < 2:
                    continue
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                        if check_route(new_route):
                            old_c = route_cost(route)
                            new_c = route_cost(new_route)
                            if new_c < old_c - 1e-9:
                                routes[ri] = new_route
                                improved = True
        return routes
    
    # Try to merge two routes
    def try_merge(routes, i, j):
        """Try to merge routes[i] and routes[j]. Returns merged route or None."""
        r1 = routes[i]
        r2 = routes[j]
        load = sum(demands[v] for v in r1) + sum(demands[v] for v in r2)
        if load > capacity:
            return None
        
        # Try different orderings: r1+r2, r2+r1
        for order in ([r1, r2], [r2, r1]):
            merged = order[0] + order[1]
            if check_route(merged):
                return merged
        return None
    
    # Local search loop
    iteration = 0
    while time.time() - start_time < time_limit * 0.8:
        iteration += 1
        routes = [list(r) for r in best_routes]
        
        # 2-opt improvement
        routes = two_opt_improve(routes)
        
        # Try merging pairs
        merged = False
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if time.time() - start_time > time_limit * 0.8:
                    break
                merged_route = try_merge(routes, i, j)
                if merged_route is not None:
                    new_routes = [r for k, r in enumerate(routes) if k != i and k != j]
                    new_routes.append(merged_route)
                    new_cost = total_cost(new_routes)
                    if new_cost < best_cost - 1e-9:
                        best_routes = new_routes
                        best_cost = new_cost
                        merged = True
                        break
            if merged:
                break
        
        if not merged:
            # Check if current routes are better
            cur_cost = total_cost(routes)
            if cur_cost < best_cost - 1e-9:
                best_routes = routes
                best_cost = cur_cost
    
    return best_routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Solve the instance
    routes = _solve_instance(instance, 110)
    
    # Validate that all customers are served exactly once
    served = []
    for r in routes:
        served.extend(r)
    expected = list(range(1, instance["n_customers"] + 1))
    if sorted(served) != sorted(expected):
        # Fallback to one-per-route
        customers = list(range(1, instance["n_customers"] + 1))
        routes = [[v] for v in customers]
    
    return routes
# EVOLVE-BLOCK-END

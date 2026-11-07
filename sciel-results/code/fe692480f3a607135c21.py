import heapq
import numpy as np
from math import sqrt

def plan(instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = 1 + n_customers + len(stations)
    cust_nodes = list(range(1, n_customers + 1))
    station_set = set(stations)
    
    D = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            D[i,j] = sqrt(dx*dx + dy*dy)
            D[j,i] = D[i,j]
    
    def is_customer(node):
        return 1 <= node <= n_customers
    
    def is_station(node):
        return node in station_set
    
    def check_route(route):
        time = 0.0
        battery = float(battery_cap)
        load = 0
        prev = 0
        total_dist = 0.0
        for node in route:
            d = D[prev, node]
            if battery - d < -1e-9:
                return False, 0.0
            time += d / speed
            battery -= d
            total_dist += d
            if is_customer(node):
                if time > due[node] + 1e-9:
                    return False, 0.0
                time = max(time, ready[node])
                time += service[node]
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, 0.0
            elif is_station(node):
                energy_needed = battery_cap - battery
                if energy_needed > 1e-9:
                    time += energy_needed * recharge_rate
                battery = battery_cap
            prev = node
        d = D[prev, 0]
        if battery - d < -1e-9:
            return False, 0.0
        time += d / speed
        total_dist += d
        if time > horizon + 1e-9:
            return False, 0.0
        return True, total_dist
    
    # Build initial solution with greedy insertion
    def greedy_insertion():
        customers_sorted = sorted(cust_nodes, key=lambda v: due[v])
        routes = []
        for cust in customers_sorted:
            best_idx = -1
            best_pos = -1
            best_cost = float('inf')
            for r_idx, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [cust] + route[pos:]
                    feasible, cost = check_route(new_route)
                    if feasible and cost < best_cost:
                        best_cost = cost
                        best_idx = r_idx
                        best_pos = pos
            if best_idx >= 0:
                routes[best_idx] = routes[best_idx][:best_pos] + [cust] + routes[best_idx][best_pos:]
            else:
                routes.append([cust])
        return routes
    
    def two_opt(routes):
        improved = True
        it = 0
        while improved and it < 50:
            improved = False
            it += 1
            for r in range(len(routes)):
                route = routes[r]
                n = len(route)
                if n < 2:
                    continue
                for i in range(n-1):
                    for j in range(i+1, n):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        f, c = check_route(new_route)
                        if f:
                            oc = check_route(route)[1]
                            if c < oc - 1e-9:
                                routes[r] = new_route
                                improved = True
        return routes
    
    def relocate(routes):
        improved = True
        it = 0
        while improved and it < 30:
            improved = False
            it += 1
            for r1 in range(len(routes)):
                for pos in range(len(routes[r1])):
                    cust = routes[r1][pos]
                    for r2 in range(len(routes)):
                        if r1 == r2:
                            continue
                        for pos2 in range(len(routes[r2]) + 1):
                            new_r1 = routes[r1][:pos] + routes[r1][pos+1:]
                            new_r2 = routes[r2][:pos2] + [cust] + routes[r2][pos2:]
                            f1, c1 = check_route(new_r1)
                            f2, c2 = check_route(new_r2)
                            if f1 and f2:
                                oc = check_route(routes[r1])[1] + check_route(routes[r2])[1]
                                if c1 + c2 < oc - 1e-9:
                                    routes[r1] = new_r1
                                    routes[r2] = new_r2
                                    improved = True
        return routes
    
    routes = greedy_insertion()
    routes = two_opt(routes)
    routes = relocate(routes)
    
    served = set()
    for route in routes:
        for node in route:
            if is_customer(node):
                served.add(node)
    missing = set(cust_nodes) - served
    for cust in missing:
        routes.append([cust])
    
    return routes

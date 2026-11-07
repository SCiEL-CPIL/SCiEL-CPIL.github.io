import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 148.0

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    customers = list(range(1, n_cust + 1))

    def check_route(route_nodes):
        if not route_nodes:
            return True, 0.0
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        for node in route_nodes:
            d = dist[prev, node]
            t += d / speed
            b -= d
            if b < -_EPS:
                return False, 0.0
            if node in customers:
                t = max(t, ready[node])
                if t > due[node] + _EPS:
                    return False, 0.0
                t += service[node]
                load += demands[node]
                if load > capacity + _EPS:
                    return False, 0.0
            else:
                deficit = battery_cap - b
                if deficit > 0:
                    t += deficit * recharge_rate
                b = battery_cap
            d_total += d
            prev = node
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -_EPS:
            return False, 0.0
        if t > horizon + _EPS:
            return False, 0.0
        d_total += d
        return True, d_total

    def nearest_insertion(customers_set, start_route=None):
        """Build a route using nearest insertion heuristic."""
        if not start_route:
            start_route = []
        
        route = start_route[:]
        remaining = set(customers_set)
        
        # Find first customer to insert - closest to depot with most time slack
        best_c = None
        best_score = float('inf')
        for c in remaining:
            d_to_depot = dist[0, c]
            slack = due[c] - ready[c]
            score = d_to_depot / max(slack, 1.0)
            if score < best_score:
                best_score = score
                best_c = c
        
        if best_c is None:
            return route
        
        route.append(best_c)
        remaining.remove(best_c)
        
        # Iteratively insert nearest customer
        while remaining:
            best_c = None
            best_pos = None
            best_cost = float('inf')
            
            for c in remaining:
                # Find best position to insert c
                for i in range(len(route) + 1):
                    if i == 0:
                        prev_node = 0
                    else:
                        prev_node = route[i-1]
                    if i == len(route):
                        next_node = 0
                    else:
                        next_node = route[i]
                    
                    extra_dist = dist[prev_node, c] + dist[c, next_node] - dist[prev_node, next_node]
                    if extra_dist < best_cost - _EPS:
                        # Check if this insertion is feasible
                        test_route = route[:i] + [c] + route[i:]
                        valid, _ = check_route(test_route)
                        if valid:
                            best_cost = extra_dist
                            best_c = c
                            best_pos = i
            
            if best_c is None:
                # Try inserting the closest remaining customer
                for c in remaining:
                    for i in range(len(route) + 1):
                        if i == 0:
                            prev_node = 0
                        else:
                            prev_node = route[i-1]
                        if i == len(route):
                            next_node = 0
                        else:
                            next_node = route[i]
                        
                        test_route = route[:i] + [c] + route[i:]
                        valid, _ = check_route(test_route)
                        if valid:
                            best_c = c
                            best_pos = i
                            break
                    if best_c is not None:
                        break
            
            if best_c is None:
                break
            
            route.insert(best_pos, best_c)
            remaining.remove(best_c)
        
        return route

    def add_station_if_needed(route):
        """Add charging stations if battery would go negative."""
        t = 0.0
        b = battery_cap
        new_route = []
        prev = 0
        
        for node in route:
            d = dist[prev, node]
            
            # Check if we need a station before going to this node
            if b - d < 0:
                # Find the station closest to prev
                best_station = None
                best_d = float('inf')
                for s in stations:
                    ds = dist[prev, s]
                    if ds < best_d and b >= ds:
                        best_d = ds
                        best_station = s
                
                if best_station is not None:
                    # Travel to station
                    t += dist[prev, best_station] / speed
                    b -= dist[prev, best_station]
                    # Recharge
                    deficit = battery_cap - b
                    if deficit > 0:
                        t += deficit * recharge_rate
                    b = battery_cap
                    new_route.append(best_station)
                    prev = best_station
                    d = dist[prev, node]
            
            t += d / speed
            b -= d
            if b < -_EPS:
                return None
            
            if node in customers:
                t = max(t, ready[node])
                if t > due[node] + _EPS:
                    return None
                t += service[node]
            
            new_route.append(node)
            prev = node
        
        # Check return to depot
        d = dist[prev, 0]
        if b - d < 0:
            # Need station before returning
            best_station = None
            best_d = float('inf')
            for s in stations:
                ds = dist[prev, s]
                if ds < best_d and b >= ds:
                    best_d = ds
                    best_station = s
            
            if best_station is not None:
                t += dist[prev, best_station] / speed
                b -= dist[prev, best_station]
                deficit = battery_cap - b
                if deficit > 0:
                    t += deficit * recharge_rate
                b = battery_cap
                new_route.append(best_station)
                d = dist[best_station, 0]
            else:
                return None
        
        t += d / speed
        b -= d
        if b < -_EPS or t > horizon + _EPS:
            return None
        
        return new_route

    def optimize_route(route_nodes, time_limit=2.0):
        """Optimize a single route using 2-opt and relocate."""
        if len(route_nodes) <= 1:
            return route_nodes

        current_route = route_nodes[:]
        current_valid, current_dist = check_route(current_route)
        if not current_valid:
            return current_route

        best_route = current_route[:]
        best_dist = current_dist

        # 2-opt
        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            n = len(current_route)
            for i in range(n - 1):
                if time.time() - t0 > time_limit:
                    break
                for j in range(i + 1, n):
                    u = current_route[i-1] if i > 0 else 0
                    v = current_route[i]
                    w = current_route[j]
                    x = current_route[j+1] if j + 1 < n else 0

                    old_d = dist[u, v] + dist[w, x]
                    new_d = dist[u, w] + dist[v, x]

                    if new_d < old_d - _EPS:
                        current_route[i:j+1] = current_route[i:j+1][::-1]
                        valid, d = check_route(current_route)
                        if valid and d < best_dist:
                            best_dist = d
                            best_route = current_route[:]
                        else:
                            current_route[i:j+1] = current_route[i:j+1][::-1]
                        improved = True

        # Relocate
        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            n = len(current_route)
            for i in range(n):
                if time.time() - t0 > time_limit:
                    break
                c = current_route[i]
                temp_route = current_route[:i] + current_route[i+1:]
                if not temp_route:
                    continue
                for j in range(len(temp_route) + 1):
                    if j == i or (j > i and j == i + 1):
                        continue
                    new_route = temp_route[:j] + [c] + temp_route[j:]
                    valid, d = check_route(new_route)
                    if valid and d < best_dist - _EPS:
                        best_dist = d
                        best_route = new_route
                        current_route = new_route
                        improved = True
                        break

        return best_route

    def build_routes_nearest_insertion():
        """Build all routes using nearest insertion."""
        routes = []
        remaining = set(customers)
        
        while remaining:
            route = nearest_insertion(remaining)
            if not route:
                # Fallback: pick any customer
                c = next(iter(remaining))
                route = [c]
            
            # Try to add stations if needed
            full_route = add_station_if_needed(route)
            if full_route is None:
                # Try without the last customer
                if len(route) > 1:
                    last = route.pop()
                    full_route = add_station_if_needed(route)
                    if full_route is None:
                        route.append(last)
                        full_route = add_station_if_needed(route)
                    else:
                        # Add last as separate route later
                        remaining.add(last)
                else:
                    full_route = add_station_if_needed(route)
            
            if full_route is None:
                # Even single customer can't be done? Should not happen
                full_route = route
            
            # Optimize
            full_route = optimize_route(full_route)
            routes.append(full_route)
            
            for c in full_route:
                if c in customers:
                    remaining.discard(c)
        
        return routes

    def cluster_customers(k, n_restarts=10):
        """Cluster customers into k groups using k-means with multiple restarts."""
        if k >= len(customers):
            return [[c] for c in customers]

        best_clusters = None
        best_dist = float('inf')

        for restart in range(n_restarts):
            if time.time() > DEADLINE - 3:
                break

            centroids = np.array([coords[c] for c in random.sample(customers, k)])

            for _ in range(30):
                labels = np.zeros(len(customers), dtype=int)
                for idx, c in enumerate(customers):
                    d_to_centroids = np.sqrt(np.sum((centroids - coords[c])**2, axis=1))
                    labels[idx] = np.argmin(d_to_centroids)

                new_centroids = np.zeros_like(centroids)
                counts = np.zeros(k)
                for idx, c in enumerate(customers):
                    new_centroids[labels[idx]] += coords[c]
                    counts[labels[idx]] += 1

                for i in range(k):
                    if counts[i] > 0:
                        new_centroids[i] /= counts[i]
                    else:
                        new_centroids[i] = coords[random.choice(customers)]

                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

            clusters = [[] for _ in range(k)]
            for idx, c in enumerate(customers):
                clusters[labels[idx]].append(c)
            clusters = [cl for cl in clusters if cl]

            # Compute total intra-cluster distance
            total_d = 0
            for cl in clusters:
                for c1 in cl:
                    for c2 in cl:
                        if c1 != c2:
                            total_d += dist[c1, c2]

            if total_d < best_dist:
                best_dist = total_d
                best_clusters = clusters

        return best_clusters if best_clusters else [[c] for c in customers]

    def build_solution_from_clusters(clusters):
        """Build routes from clusters."""
        routes = []
        for cl in clusters:
            if not cl:
                continue
            
            angles = [math.atan2(coords[c][1], coords[c][0]) for c in cl]
            cl_sorted = [c for _, c in sorted(zip(angles, cl))]
            
            route = cl_sorted[:]
            full_route = add_station_if_needed(route)
            if full_route is None:
                # Try building with nearest insertion
                full_route = nearest_insertion(set(cl))
                full_route = add_station_if_needed(full_route)
            
            if full_route is None:
                # Fallback to singles
                for c in cl:
                    sr = [c]
                    fr = add_station_if_needed(sr)
                    routes.append(fr if fr else sr)
            else:
                full_route = optimize_route(full_route)
                routes.append(full_route)

        return routes

    def inter_route_relocate(routes, time_limit=5.0):
        """Move customers between routes to improve total distance."""
        if len(routes) <= 1:
            return routes

        best_routes = [r[:] for r in routes]
        best_cost = sum(check_route(r)[1] for r in best_routes)

        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            for i in range(len(best_routes)):
                if time.time() - t0 > time_limit:
                    break
                for j in range(len(best_routes[i])):
                    c = best_routes[i][j]
                    # Remove c from route i
                    new_i = best_routes[i][:j] + best_routes[i][j+1:]
                    valid_i, d_i = check_route(new_i) if new_i else (True, 0.0)
                    if not valid_i:
                        continue

                    # Try inserting into each other route
                    for k in range(len(best_routes)):
                        if k == i:
                            continue
                        for pos in range(len(best_routes[k]) + 1):
                            new_k = best_routes[k][:pos] + [c] + best_routes[k][pos:]
                            valid_k, d_k = check_route(new_k)
                            if valid_k:
                                total_d = d_i + d_k
                                # Subtract old distances
                                old_total = sum(check_route(r)[1] for r in best_routes)
                                if total_d < old_total - _EPS:
                                    best_routes[i] = new_i
                                    best_routes[k] = new_k
                                    best_cost = old_total - (check_route(best_routes[i])[1] + check_route(best_routes[k])[1]) + total_d
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        return best_routes

    # Main strategy
    best_routes = None
    best_cost = float('inf')

    # Method 1: Nearest insertion
    if time.time() < DEADLINE - 5:
        routes = build_routes_nearest_insertion()
        # Ensure all customers covered
        served = set()
        for r in routes:
            for c in r:
                served.add(c)
        missing = set(customers) - served
        for c in missing:
            sr = [c]
            fr = add_station_if_needed(sr)
            routes.append(fr if fr else sr)
        
        total_d = 0
        valid_all = True
        for r in routes:
            valid, d = check_route(r)
            if not valid:
                valid_all = False
                break
            total_d += d
        
        if valid_all and total_d < best_cost:
            best_cost = total_d
            best_routes = routes[:]

    # Method 2: K-means clustering with various k
    for k in range(3, 9):
        if time.time() > DEADLINE - 5:
            break
        
        clusters = cluster_customers(k, n_restarts=5)
        routes = build_solution_from_clusters(clusters)
        
        served = set()
        for r in routes:
            for c in r:
                served.add(c)
        missing = set(customers) - served
        for c in missing:
            sr = [c]
            fr = add_station_if_needed(sr)
            routes.append(fr if fr else sr)
        
        total_d = 0
        valid_all = True
        for r in routes:
            valid, d = check_route(r)
            if not valid:
                valid_all = False
                break
            total_d += d
        
        if valid_all and total_d < best_cost:
            best_cost = total_d
            best_routes = routes[:]

    # Inter-route optimization
    if best_routes is not None and time.time() < DEADLINE - 5:
        best_routes = inter_route_relocate(best_routes, time_limit=DEADLINE - time.time() - 2)
        # Re-verify
        served = set()
        for r in best_routes:
            for c in r:
                served.add(c)
        missing = set(customers) - served
        for c in missing:
            sr = [c]
            fr = add_station_if_needed(sr)
            best_routes.append(fr if fr else sr)

    # Final verification
    if best_routes is None:
        best_routes = [[c] for c in customers]

    final_routes = []
    served = set()
    for r in best_routes:
        valid, d = check_route(r)
        if valid:
            final_routes.append(r)
            for c in r:
                served.add(c)

    missing = set(customers) - served
    for c in missing:
        sr = [c]
        fr = add_station_if_needed(sr)
        if fr:
            final_routes.append(fr)
        else:
            final_routes.append(sr)

    return final_routes

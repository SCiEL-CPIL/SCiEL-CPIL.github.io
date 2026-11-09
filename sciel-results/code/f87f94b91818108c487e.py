import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0

    # Extract instance data
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    stations_set = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))
    n_nodes = len(coords)

    # Pre-compute full distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    def route_distance_and_feasibility(route):
        if not route:
            return True, 0.0

        cur_node = 0
        cur_time = 0.0
        cur_bat = battery
        total_dist = 0.0
        load = 0.0

        for node in route:
            d = dist[cur_node, node]
            cur_time += d / speed
            cur_bat -= d
            total_dist += d

            if cur_bat < -1e-9:
                return False, total_dist

            if node in stations_set:
                energy = battery - cur_bat
                cur_time += energy * recharge_rate
                cur_bat = battery
            else:
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, total_dist
                cur_time = max(cur_time, ready[node])
                if cur_time > due[node] + 1e-9:
                    return False, total_dist
                cur_time += service[node]

            cur_node = node

        d = dist[cur_node, 0]
        cur_time += d / speed
        cur_bat -= d
        total_dist += d

        if cur_bat < -1e-9:
            return False, total_dist
        if cur_time > horizon + 1e-9:
            return False, total_dist

        return True, total_dist

    def compute_total_distance(routes):
        total = 0.0
        for route in routes:
            if not route:
                continue
            cur_node = 0
            for node in route:
                total += dist[cur_node, node]
                cur_node = node
            total += dist[cur_node, 0]
        return total

    # --- SET PARTITIONING WITH COLUMN GENERATION ---
    # Strategy: Generate candidate routes (columns) using a k-tree heuristic that
    # explicitly handles battery and time windows. Then solve the set partitioning
    # problem using branch-and-bound with a greedy covering heuristic.

    # Step 1: Generate initial columns using a k-tree heuristic
    # A k-tree is a spanning tree with at most k edges per node.
    # We use this to generate diverse feasible routes.

    def generate_column(start_node, unvisited, k=3):
        """Generate a feasible route starting from start_node using k-tree heuristic."""
        route = []
        cur_node = start_node
        cur_time = 0.0
        cur_bat = battery
        load = 0.0

        while unvisited:
            # Find feasible next nodes
            feasible_nexts = []
            for c in unvisited:
                d = dist[cur_node, c]
                new_time = cur_time + d / speed
                new_bat = cur_bat - d
                if new_bat < -1e-9:
                    continue
                if load + demands[c] > capacity:
                    continue
                new_time = max(new_time, ready[c])
                if new_time > due[c]:
                    continue
                new_time += service[c]
                feasible_nexts.append((c, new_time, new_bat, d))

            if not feasible_nexts:
                break

            # Use k-tree heuristic: prefer nodes that are close to the current node
            # and have good connectivity to remaining unvisited nodes
            # Simplified: use nearest neighbor with a diversity factor
            best_c = None
            best_score = float('inf')

            for c, nt, nb, d in feasible_nexts:
                # Score: distance + penalty for isolated nodes
                # Calculate average distance from c to other unvisited nodes
                if len(unvisited) > 1:
                    other_nodes = [x for x in unvisited if x != c]
                    avg_dist = np.mean([dist[c, x] for x in other_nodes])
                    score = d + 0.1 * avg_dist
                else:
                    score = d

                if score < best_score:
                    best_score = score
                    best_c = c
                    best_nt = nt
                    best_nb = nb

            if best_c is None:
                break

            route.append(best_c)
            cur_node = best_c
            cur_time = best_nt
            cur_bat = best_nb
            load += demands[best_c]
            unvisited.remove(best_c)

        # Check if route is feasible including return
        if route:
            d_back = dist[cur_node, 0]
            if cur_bat - d_back >= -1e-9:
                t_back = cur_time + d_back / speed
                if t_back <= horizon:
                    return route
        return None

    # Generate a pool of candidate routes (columns)
    def generate_column_pool(n_columns=200):
        columns = []
        rng = random.Random(42)

        # Generate columns from different starting points and with different k values
        for i in range(n_columns):
            if time.time() > deadline - 5:
                break

            # Random subset of customers to start with
            subset_size = rng.randint(3, min(8, n_customers))
            subset = set(rng.sample(customers, subset_size))

            # Try to generate a column
            col = generate_column(0, set(subset))
            if col and len(col) > 0:
                # Check feasibility
                feasible, _ = route_distance_and_feasibility(col)
                if feasible:
                    columns.append(col)

        # Also generate some full-route columns
        for i in range(50):
            if time.time() > deadline - 5:
                break
            col = generate_column(0, set(customers))
            if col and len(col) > 0:
                feasible, _ = route_distance_and_feasibility(col)
                if feasible:
                    columns.append(col)

        return columns

    # Step 2: Solve the set partitioning problem
    # We want to select a subset of columns such that:
    # 1. Each customer is covered exactly once
    # 2. Total distance is minimized

    def solve_set_partitioning(columns, customers):
        """Solve set partitioning using greedy covering + local search."""
        if not columns:
            return [[c] for c in customers]

        # Greedy covering heuristic
        remaining = set(customers)
        selected = []
        total_dist = 0.0

        # Sort columns by average distance per customer
        def col_score(col):
            d, _ = route_distance_and_feasibility(col)
            n_cust = len([x for x in col if x not in stations_set])
            if n_cust == 0:
                return float('inf')
            return d / n_cust

        sorted_cols = sorted(columns, key=col_score)

        for col in sorted_cols:
            if not remaining:
                break

            # Check if this column covers any remaining customers
            col_customers = [x for x in col if x in remaining]
            if not col_customers:
                continue

            # Check if column is feasible with respect to remaining
            # (it might conflict with already selected columns)
            # For simplicity, just check if it covers remaining customers
            if any(x in remaining for x in col if x not in stations_set):
                selected.append(col)
                for x in col:
                    if x in remaining:
                        remaining.remove(x)

        # If some customers are still uncovered, add individual routes
        if remaining:
            for c in list(remaining):
                if route_distance_and_feasibility([c])[0]:
                    selected.append([c])
                    remaining.remove(c)

        return selected

    # Step 3: Local search to improve the solution
    def local_search(routes, time_limit):
        current_routes = [list(r) for r in routes]
        current_dist = compute_total_distance(current_routes)
        best_routes = [list(r) for r in current_routes]
        best_dist = current_dist

        start_ls = time.time()
        rng = random.Random(42)

        while time.time() - start_ls < time_limit:
            improved = False

            # 1. Intra-route 2-opt
            for r_idx in range(len(current_routes)):
                route = current_routes[r_idx]
                n = len(route)
                if n < 2:
                    continue

                for i in range(n - 1):
                    for j in range(i + 1, n):
                        prev_i = route[i-1] if i > 0 else 0
                        next_j = route[j+1] if j < n-1 else 0

                        old_cost = dist[prev_i, route[i]] + dist[route[j], next_j]
                        new_cost = dist[prev_i, route[j]] + dist[route[i], next_j]

                        if new_cost < old_cost - 1e-9:
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            if route_distance_and_feasibility(new_route)[0]:
                                new_routes = [list(r) for r in current_routes]
                                new_routes[r_idx] = new_route
                                new_dist = compute_total_distance(new_routes)
                                if new_dist < current_dist - 1e-9:
                                    current_routes = new_routes
                                    current_dist = new_dist
                                    if current_dist < best_dist:
                                        best_dist = current_dist
                                        best_routes = [list(r) for r in current_routes]
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break

            # 2. Inter-route Relocate
            if not improved and len(current_routes) > 1:
                for r1_idx in range(len(current_routes)):
                    route1 = current_routes[r1_idx]
                    if not route1:
                        continue
                    for idx in range(len(route1)):
                        node = route1[idx]
                        new_route1 = route1[:idx] + route1[idx+1:]

                        for r2_idx in range(len(current_routes)):
                            if r2_idx == r1_idx:
                                continue
                            route2 = current_routes[r2_idx]
                            for pos in range(len(route2) + 1):
                                prev1 = route1[idx-1] if idx > 0 else 0
                                next1 = route1[idx+1] if idx < len(route1)-1 else 0
                                old_cost1 = dist[prev1, node] + dist[node, next1]
                                new_cost1 = dist[prev1, next1]

                                prev2 = route2[pos-1] if pos > 0 else 0
                                next2 = route2[pos] if pos < len(route2) else 0
                                old_cost2 = dist[prev2, next2]
                                new_cost2 = dist[prev2, node] + dist[node, next2]

                                delta = (new_cost1 - old_cost1) + (new_cost2 - old_cost2)

                                if delta < -1e-9:
                                    new_route2 = route2[:pos] + [node] + route2[pos:]
                                    if route_distance_and_feasibility(new_route1)[0] and route_distance_and_feasibility(new_route2)[0]:
                                        new_routes = [list(r) for r in current_routes]
                                        new_routes[r1_idx] = new_route1
                                        new_routes[r2_idx] = new_route2
                                        new_dist = compute_total_distance(new_routes)
                                        if new_dist < current_dist - 1e-9:
                                            current_routes = new_routes
                                            current_dist = new_dist
                                            if current_dist < best_dist:
                                                best_dist = current_dist
                                                best_routes = [list(r) for r in current_routes]
                                            improved = True
                                            break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

            # 3. Inter-route Swap
            if not improved and len(current_routes) > 1:
                for r1_idx in range(len(current_routes) - 1):
                    route1 = current_routes[r1_idx]
                    for i in range(len(route1)):
                        for r2_idx in range(r1_idx + 1, len(current_routes)):
                            route2 = current_routes[r2_idx]
                            for j in range(len(route2)):
                                n1 = route1[i]
                                n2 = route2[j]

                                new_route1 = route1[:i] + [n2] + route1[i+1:]
                                new_route2 = route2[:j] + [n1] + route2[j+1:]

                                if route_distance_and_feasibility(new_route1)[0] and route_distance_and_feasibility(new_route2)[0]:
                                    new_routes = [list(r) for r in current_routes]
                                    new_routes[r1_idx] = new_route1
                                    new_routes[r2_idx] = new_route2
                                    new_dist = compute_total_distance(new_routes)
                                    if new_dist < current_dist - 1e-9:
                                        current_routes = new_routes
                                        current_dist = new_dist
                                        if current_dist < best_dist:
                                            best_dist = current_dist
                                            best_routes = [list(r) for r in current_routes]
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

        return best_routes

    # Main algorithm
    # Step 1: Generate column pool
    columns = generate_column_pool(300)

    # Step 2: Solve set partitioning
    routes = solve_set_partitioning(columns, customers)

    # Step 3: Local search
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 2.0
    if remaining > 1.0:
        routes = local_search(routes, remaining)

    # Final validation
    valid = True
    served = set()
    for route in routes:
        if not route_distance_and_feasibility(route)[0]:
            valid = False
            break
        for node in route:
            if node not in stations_set:
                if node in served:
                    valid = False
                    break
                served.add(node)

    if not valid or served != set(customers):
        # Fallback: construct a valid solution
        routes = []
        unvisited = set(customers)
        while unvisited:
            route = generate_column(0, set(unvisited))
            if route:
                routes.append(route)
                for c in route:
                    if c in unvisited:
                        unvisited.remove(c)
            else:
                # Add individual customers
                for c in list(unvisited):
                    if route_distance_and_feasibility([c])[0]:
                        routes.append([c])
                        unvisited.remove(c)
                break

    return routes

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            dist[i, j] = d
            dist[j, i] = d

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def check_route(route):
        """Check if a route is feasible considering charging stations.
        Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0

        # Check capacity
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0

        t = 0.0
        bat = float(battery)
        prev = 0
        total_dist = 0.0

        for v in route:
            d = dist[prev, v]
            bat -= d
            t += d / speed
            total_dist += d

            if bat < -1e-9:
                return False, 0.0

            # At customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
            prev = v

        # Return to depot
        d = dist[prev, 0]
        bat -= d
        t += d / speed
        total_dist += d

        if bat < -1e-9:
            return False, 0.0
        if t > horizon + 1e-9:
            return False, 0.0

        return True, total_dist

    def check_route_with_charging(route, max_charges=3):
        """Check feasibility allowing up to max_charges station visits.
        Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0

        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0

        # First check without charging
        feasible, total_dist = check_route(route)
        if feasible:
            return True, total_dist

        # Try inserting charging stations at optimal points
        # Strategy: find where battery would go negative, insert nearest station before that point
        for attempt in range(max_charges):
            t = 0.0
            bat = float(battery)
            prev = 0
            total_dist = 0.0
            inserted = []  # list of (position_in_route, station)

            for idx, v in enumerate(route):
                d = dist[prev, v]
                if bat - d < -1e-9:
                    # Need to charge before going to v
                    # Find best station: minimize extra distance
                    best_station = None
                    best_extra = float('inf')
                    for s in stations:
                        d1 = dist[prev, s]
                        d2 = dist[s, v]
                        extra = d1 + d2 - d
                        # Check if we can reach the station
                        if bat - d1 >= -1e-9:
                            # Time check for station arrival
                            t_arrive = t + d1 / speed
                            t_after = t_arrive + (battery - (bat - d1)) * recharge_rate + d2 / speed
                            if t_after <= horizon:
                                if extra < best_extra:
                                    best_extra = extra
                                    best_station = s

                    if best_station is None:
                        return False, 0.0

                    # Insert station
                    d1 = dist[prev, best_station]
                    bat -= d1
                    t += d1 / speed
                    total_dist += d1

                    # Charge
                    charge_time = (battery - bat) * recharge_rate
                    t += charge_time
                    bat = float(battery)

                    # Continue to v
                    d2 = dist[best_station, v]
                    bat -= d2
                    t += d2 / speed
                    total_dist += d2
                    prev = v

                    # Customer service
                    if t < ready[v]:
                        t = ready[v]
                    if t > due[v] + 1e-9:
                        return False, 0.0
                    t += service[v]
                else:
                    bat -= d
                    t += d / speed
                    total_dist += d
                    prev = v
                    if t < ready[v]:
                        t = ready[v]
                    if t > due[v] + 1e-9:
                        return False, 0.0
                    t += service[v]

            # Return to depot
            d = dist[prev, 0]
            if bat - d < -1e-9:
                # Need to charge before returning
                best_station = None
                best_extra = float('inf')
                for s in stations:
                    d1 = dist[prev, s]
                    d2 = dist[s, 0]
                    extra = d1 + d2 - d
                    if bat - d1 >= -1e-9:
                        if extra < best_extra:
                            best_extra = extra
                            best_station = s

                if best_station is None:
                    return False, 0.0

                d1 = dist[prev, best_station]
                bat -= d1
                t += d1 / speed
                total_dist += d1
                charge_time = (battery - bat) * recharge_rate
                t += charge_time
                bat = float(battery)
                d2 = dist[best_station, 0]
                bat -= d2
                t += d2 / speed
                total_dist += d2
            else:
                bat -= d
                t += d / speed
                total_dist += d

            if bat < -1e-9 or t > horizon + 1e-9:
                return False, 0.0

            return True, total_dist

        return False, 0.0

    def total_route_distance(routes):
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += dist[0, route[0]]
            for i in range(len(route) - 1):
                total += dist[route[i], route[i + 1]]
            total += dist[route[-1], 0]
        return total

    def greedy_construct(customer_order, allow_charging=True):
        """Construct routes using greedy nearest neighbor with time windows."""
        remaining = list(customer_order)
        routes = []

        while remaining:
            route = []
            prev = 0
            t = 0.0
            bat = float(battery)
            load = 0

            while remaining:
                best_customer = None
                best_score = float('inf')
                best_idx = -1

                for idx, v in enumerate(remaining):
                    d = dist[prev, v]
                    bat_after = bat - d
                    t_after = t + d / speed

                    if bat_after < -1e-9:
                        continue
                    if load + demands[v] > capacity:
                        continue

                    # Check time window
                    t_eff = max(t_after, ready[v])
                    if t_eff > due[v] + 1e-9:
                        continue

                    # Score: distance + small time penalty
                    score = d + (t_eff - ready[v]) * 0.03

                    if score < best_score:
                        best_score = score
                        best_customer = v
                        best_idx = idx

                if best_customer is None:
                    break

                v = best_customer
                d = dist[prev, v]
                bat -= d
                t += d / speed
                load += demands[v]

                if t < ready[v]:
                    t = ready[v]
                t += service[v]

                route.append(v)
                remaining.pop(best_idx)
                prev = v

            if route:
                routes.append(route)

        return routes

    # Initial solution: greedy with multiple orderings
    best_routes = None
    best_distance = float('inf')

    orderings = [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: dist[0, v]),
        sorted(customers, key=lambda v: (due[v], dist[0, v])),
        sorted(customers, key=lambda v: (-demands[v], dist[0, v])),
        list(customers),
    ]

    for ordering in orderings:
        if time.time() > deadline:
            break
        greedy_routes = greedy_construct(ordering)
        # Validate
        all_customers = []
        valid = True
        total_dist = 0.0
        for route in greedy_routes:
            feasible, d = check_route_with_charging(route)
            if not feasible:
                valid = False
                break
            all_customers.extend(route)
            total_dist += d

        if valid and set(all_customers) == set(customers) and len(all_customers) == len(set(all_customers)):
            if total_dist < best_distance:
                best_routes = [r[:] for r in greedy_routes]
                best_distance = total_dist

    # Fallback: one customer per route
    if best_routes is None:
        best_routes = [[v] for v in customers]
        best_distance = sum(dist[0, v] * 2 for v in customers)

    # LNS (Large Neighborhood Search) with simulated annealing
    rng = random.Random(42)

    def destroy_reconstruct(routes, destroy_fraction=0.3, rng_seed=None):
        """Remove some customers and reinsert them."""
        if rng_seed is not None:
            rng.seed(rng_seed)

        # Collect all customer positions
        all_positions = []
        for i, route in enumerate(routes):
            for j, v in enumerate(route):
                all_positions.append((i, j, v))

        n_remove = max(1, int(len(all_positions) * destroy_fraction))
        removed = rng.sample(all_positions, min(n_remove, len(all_positions)))

        # Remove customers
        removed_set = set(v for _, _, v in removed)
        new_routes = []
        for i, route in enumerate(routes):
            new_route = [v for v in route if v not in removed_set]
            if new_route:
                new_routes.append(new_route)

        # Reinsert removed customers using cheapest insertion
        for _, _, v in removed:
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')

            for i, route in enumerate(new_routes):
                # Check capacity
                route_demand = sum(demands[x] for x in route)
                if route_demand + demands[v] > capacity:
                    continue

                for pos in range(len(route) + 1):
                    # Compute insertion cost
                    if pos == 0:
                        if pos == len(route):
                            cost = dist[0, v] + dist[v, 0]
                        else:
                            cost = dist[0, v] + dist[v, route[0]] - dist[0, route[0]]
                    elif pos == len(route):
                        cost = dist[route[-1], v] + dist[v, 0] - dist[route[-1], 0]
                    else:
                        cost = dist[route[pos - 1], v] + dist[v, route[pos]] - dist[route[pos - 1], route[pos]]

                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = i
                        best_pos = pos

            if best_route_idx == -1:
                # Create new route
                new_routes.append([v])
            else:
                new_routes[best_route_idx].insert(best_pos, v)

        return new_routes

    # LNS loop
    current_routes = [r[:] for r in best_routes]
    current_dist = best_distance
    T = 2.0  # initial temperature
    cooling = 0.9995
    min_T = 0.01

    lns_iter = 0
    while time.time() < deadline and T > min_T:
        lns_iter += 1

        # Destroy and reconstruct
        new_routes = destroy_reconstruct(current_routes, destroy_fraction=0.25, rng_seed=lns_iter)

        # Validate
        all_customers = []
        valid = True
        new_dist = 0.0
        for route in new_routes:
            if not route:
                continue
            feasible, d = check_route_with_charging(route)
            if not feasible:
                valid = False
                break
            all_customers.extend(route)
            new_dist += d

        if not valid or set(all_customers) != set(customers) or len(all_customers) != len(set(all_customers)):
            T *= cooling
            continue

        # Simulated annealing acceptance
        delta = new_dist - current_dist
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_routes = new_routes
            current_dist = new_dist
            if current_dist < best_distance:
                best_routes = [r[:] for r in current_routes]
                best_distance = current_dist

        T *= cooling

        # Periodic local search refinement
        if lns_iter % 20 == 0 and time.time() < deadline - 10:
            # 2-opt within each route
            for i in range(len(current_routes)):
                route = current_routes[i]
                if len(route) < 2:
                    continue
                improved = True
                while improved:
                    improved = False
                    for a in range(len(route)):
                        if time.time() > deadline - 10:
                            break
                        for b in range(a + 1, len(route)):
                            old_boundary = 0.0
                            if a > 0:
                                old_boundary += dist[route[a - 1], route[a]]
                            else:
                                old_boundary += dist[0, route[a]]
                            if b < len(route) - 1:
                                old_boundary += dist[route[b], route[b + 1]]
                            else:
                                old_boundary += dist[route[b], 0]

                            new_boundary = 0.0
                            if a > 0:
                                new_boundary += dist[route[a - 1], route[b]]
                            else:
                                new_boundary += dist[0, route[b]]
                            if b < len(route) - 1:
                                new_boundary += dist[route[a], route[b + 1]]
                            else:
                                new_boundary += dist[route[a], 0]

                            if new_boundary < old_boundary - 1e-6:
                                new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                                feasible, _ = check_route_with_charging(new_route)
                                if feasible:
                                    current_routes[i] = new_route
                                    route = new_route
                                    improved = True
                                    break
                        if time.time() > deadline - 10:
                            break
                    if time.time() > deadline - 10:
                        break

            # Relocate moves
            for i in range(len(current_routes)):
                if time.time() > deadline - 10:
                    break
                for j in range(len(current_routes[i])):
                    if time.time() > deadline - 10:
                        break
                    v = current_routes[i][j]
                    for k in range(len(current_routes)):
                        if k == i or time.time() > deadline - 10:
                            continue
                        for pos in range(len(current_routes[k]) + 1):
                            # Try moving v from route i to position pos in route k
                            if k == i and pos == j:
                                continue

                            new_routes = []
                            for r in range(len(current_routes)):
                                if r == i:
                                    nr = [x for idx2, x in enumerate(current_routes[i]) if idx2 != j]
                                    new_routes.append(nr)
                                elif r == k:
                                    nr = current_routes[k][:]
                                    nr.insert(pos, v)
                                    new_routes.append(nr)
                                else:
                                    new_routes.append(current_routes[r][:])

                            # Validate
                            all_c = []
                            valid2 = True
                            nd = 0.0
                            for route in new_routes:
                                if not route:
                                    continue
                                f, d = check_route_with_charging(route)
                                if not f:
                                    valid2 = False
                                    break
                                all_c.extend(route)
                                nd += d

                            if valid2 and set(all_c) == set(customers) and len(all_c) == len(set(all_c)):
                                if nd < current_dist - 1e-6:
                                    current_routes = new_routes
                                    current_dist = nd
                                    if current_dist < best_distance:
                                        best_routes = [r[:] for r in current_routes]
                                        best_distance = current_dist
                                    j = -1  # restart j loop
                                    break
                        if j == -1:
                            break
                    if j == -1:
                        break

    # Final local search pass
    if time.time() < deadline - 5:
        # Swap moves
        improved_swap = True
        while improved_swap and time.time() < deadline - 5:
            improved_swap = False
            for i in range(len(best_routes)):
                if time.time() > deadline - 5:
                    break
                for j in range(len(best_routes)):
                    if i == j or time.time() > deadline - 5:
                        continue
                    for a in range(len(best_routes[i])):
                        if time.time() > deadline - 5:
                            break
                        for b in range(len(best_routes[j])):
                            vi = best_routes[i][a]
                            vj = best_routes[j][b]
                            if vi == vj:
                                continue

                            new_routes = []
                            for r in range(len(best_routes)):
                                if r == i:
                                    nr = best_routes[i][:]
                                    nr[a] = vj
                                    new_routes.append(nr)
                                elif r == j:
                                    nr = best_routes[j][:]
                                    nr[b] = vi
                                    new_routes.append(nr)
                                else:
                                    new_routes.append(best_routes[r][:])

                            all_c = []
                            valid = True
                            nd = 0.0
                            for route in new_routes:
                                if not route:
                                    continue
                                f, d = check_route_with_charging(route)
                                if not f:
                                    valid = False
                                    break
                                all_c.extend(route)
                                nd += d

                            if valid and set(all_c) == set(customers) and len(all_c) == len(set(all_c)):
                                cur_d = total_route_distance(best_routes)
                                if nd < cur_d - 1e-6:
                                    best_routes = new_routes
                                    improved_swap = True
                                    break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
                if improved_swap:
                    break

    return best_routes
# EVOLVE-BLOCK-END

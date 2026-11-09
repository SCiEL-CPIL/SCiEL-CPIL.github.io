"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random


def _dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _route_state(route, D, instance):
    """Return (time, battery, dist, feasible, demand) for a route."""
    if not route:
        return 0.0, instance.get("battery", 300), 0.0, True, 0
    speed = instance["speed"]
    battery_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = instance["_stations_set"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    t = 0.0
    bat = float(battery_cap)
    dist = 0.0
    demand = 0
    prev = 0
    for v in route:
        d = D[prev, v]
        t += d
        bat -= d
        dist += d
        if bat < -1e-9:
            return t, bat, dist, False, demand
        if v in stations:
            energy_needed = battery_cap - bat
            t += energy_needed * recharge_rate
            bat = float(battery_cap)
        else:
            demand += demands[v]
            if demand > instance["capacity"] + 1e-9:
                return t, bat, dist, False, demand
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return t, bat, dist, False, demand
            t += service[v]
        prev = v
    d = D[prev, 0]
    t += d
    bat -= d
    dist += d
    if bat < -1e-9:
        return t, bat, dist, False, demand
    if t > horizon + 1e-9:
        return t, bat, dist, False, demand
    return t, bat, dist, True, demand


def _route_feasible(route, D, instance):
    if not route:
        return True
    _, _, _, feas, _ = _route_state(route, D, instance)
    return feas


def _route_dist(route, D):
    if not route:
        return 0.0
    total = D[0, route[0]]
    for i in range(len(route) - 1):
        total += D[route[i], route[i + 1]]
    total += D[route[-1], 0]
    return total


def _total_dist(routes, D):
    return sum(_route_dist(r, D) for r in routes)


def _construct_clustering(instance, D, seed=42):
    """Cluster customers by angle from depot, then greedily build routes per cluster."""
    rng = random.Random(seed)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    demands = instance["demands"]
    cap = instance["capacity"]
    coords = instance["coords"]
    due = instance["due"]

    # Compute angle from depot for each customer
    depot_x, depot_y = coords[0]
    angles = {}
    for c in customers:
        dx = coords[c][0] - depot_x
        dy = coords[c][1] - depot_y
        angles[c] = math.atan2(dy, dx)

    # Sort by angle
    sorted_by_angle = sorted(customers, key=lambda c: angles[c])

    # Create clusters: group consecutive customers with similar angles
    # Use a simple approach: divide into roughly equal angular sectors
    n_clusters = max(3, n_customers // 6)  # aim for ~6 customers per route
    cluster_size = n_customers // n_clusters
    clusters = []
    for i in range(0, n_customers, cluster_size):
        clusters.append(sorted_by_angle[i:i + cluster_size])

    # Build routes from clusters
    routes = []
    for cluster in clusters:
        # Sort cluster by due date
        cluster_sorted = sorted(cluster, key=lambda c: due[c])
        route = []
        route_demand = 0
        for c in cluster_sorted:
            if route_demand + demands[c] > cap:
                break
            # Try to insert
            best_pos = None
            best_cost = float('inf')
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                if not _route_feasible(new_route, D, instance):
                    continue
                if pos == 0:
                    old_cost = D[0, route[0]] if route else 0
                    new_cost = D[0, c]
                elif pos == len(route):
                    old_cost = D[route[-1], 0]
                    new_cost = D[c, 0]
                else:
                    old_cost = D[route[pos-1], route[pos]]
                    new_cost = D[route[pos-1], c] + D[c, route[pos]]
                delta = new_cost - old_cost
                if delta < best_cost:
                    best_cost = delta
                    best_pos = pos
            if best_pos is not None:
                route.insert(best_pos, c)
                route_demand += demands[c]
        if route:
            routes.append(route)

    # Add any remaining customers
    assigned = set()
    for r in routes:
        assigned.update(r)
    remaining = [c for c in customers if c not in assigned]
    for c in remaining:
        best_pos = None
        best_route_idx = -1
        best_cost = float('inf')
        for ri, route in enumerate(routes):
            route_demand = sum(demands[x] for x in route)
            if route_demand + demands[c] > cap:
                continue
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                if not _route_feasible(new_route, D, instance):
                    continue
                if pos == 0:
                    old_cost = D[0, route[0]] if route else 0
                    new_cost = D[0, c]
                elif pos == len(route):
                    old_cost = D[route[-1], 0]
                    new_cost = D[c, 0]
                else:
                    old_cost = D[route[pos-1], route[pos]]
                    new_cost = D[route[pos-1], c] + D[c, route[pos]]
                delta = new_cost - old_cost
                if delta < best_cost:
                    best_cost = delta
                    best_pos = pos
                    best_route_idx = ri
        if best_route_idx >= 0:
            routes[best_route_idx].insert(best_pos, c)
        else:
            routes.append([c])

    return routes


def _construct_greedy(instance, D, seed=0):
    """Greedy insertion construction for EVRP with randomization."""
    rng = random.Random(seed)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    demands = instance["demands"]
    cap = instance["capacity"]
    due = instance["due"]

    # Random perturbation of sort order
    if seed > 0:
        # Add noise to due dates for sorting
        noisy_due = {c: due[c] + rng.uniform(-20, 20) for c in customers}
        sorted_customers = sorted(customers, key=lambda v: noisy_due[v])
    else:
        sorted_customers = sorted(customers, key=lambda v: due[v])

    routes = []
    route_demands = []

    for v in sorted_customers:
        best_pos = None
        best_route_idx = -1
        best_cost = float('inf')

        for ri, route in enumerate(routes):
            if route_demands[ri] + demands[v] > cap + 1e-9:
                continue

            for pos in range(len(route) + 1):
                new_route = route[:pos] + [v] + route[pos:]
                if not _route_feasible(new_route, D, instance):
                    continue
                if pos == 0:
                    old_cost = D[0, route[0]] if route else 0
                    new_cost = D[0, v]
                elif pos == len(route):
                    old_cost = D[route[-1], 0]
                    new_cost = D[v, 0]
                else:
                    old_cost = D[route[pos-1], route[pos]]
                    new_cost = D[route[pos-1], v] + D[v, route[pos]]
                delta_cost = new_cost - old_cost
                if delta_cost < best_cost:
                    best_cost = delta_cost
                    best_pos = pos
                    best_route_idx = ri

        if best_route_idx >= 0:
            routes[best_route_idx].insert(best_pos, v)
            route_demands[best_route_idx] += demands[v]
        else:
            routes.append([v])
            route_demands.append(demands[v])

    return routes


def _2opt(route, D, instance, max_iter=300):
    """2-opt with feasibility check."""
    if len(route) < 2:
        return route
    best = route[:]
    n = len(best)

    for _ in range(max_iter):
        improved = False
        for i in range(n - 1):
            a = best[i - 1] if i > 0 else 0
            b = best[i]
            for j in range(i + 1, n):
                c = best[j]
                d = best[j + 1] if j + 1 < n else 0
                old_seg = D[a, b] + D[c, d]
                new_seg = D[a, c] + D[b, d]
                if new_seg >= old_seg - 1e-9:
                    continue
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if not _route_feasible(new_route, D, instance):
                    continue
                best = new_route
                n = len(best)
                improved = True
                break
            if improved:
                break
        if not improved:
            break
    return best


def _or_opt(route, D, instance, max_iter=80):
    """Or-opt: relocate single nodes."""
    if len(route) < 3:
        return route
    best = route[:]
    n = len(best)
    best_dist = _route_dist(best, D)

    for _ in range(max_iter):
        improved = False
        for i in range(n):
            node = best[i]
            remaining = best[:i] + best[i + 1:]
            for j in range(len(remaining) + 1):
                if j == i or j == i - 1:
                    continue
                candidate = remaining[:j] + [node] + remaining[j:]
                if candidate == best:
                    continue
                if not _route_feasible(candidate, D, instance):
                    continue
                cand_dist = _route_dist(candidate, D)
                if cand_dist < best_dist - 1e-9:
                    best = candidate
                    best_dist = cand_dist
                    n = len(best)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _relocate_inter(routes, D, instance, max_iters=30):
    """Inter-route relocate moves."""
    for _ in range(max_iters):
        improved = False
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for j in range(len(routes)):
                if i == j:
                    continue
                for ai in range(len(routes[i])):
                    v = routes[i][ai]
                    new_i = routes[i][:]
                    new_i.pop(ai)
                    for aj in range(len(routes[j]) + 1):
                        new_j = routes[j][:]
                        new_j.insert(aj, v)
                        if not _route_feasible(new_i, D, instance):
                            continue
                        if not _route_feasible(new_j, D, instance):
                            continue
                        old_d = _route_dist(routes[i], D) + _route_dist(routes[j], D)
                        new_d = _route_dist(new_i, D) + _route_dist(new_j, D)
                        if new_d < old_d - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
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
    return routes


def _swap_inter(routes, D, instance, max_iters=15):
    """Inter-route swap moves."""
    for _ in range(max_iters):
        improved = False
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for j in range(i + 1, len(routes)):
                if not routes[j]:
                    continue
                for ai in range(len(routes[i])):
                    for aj in range(len(routes[j])):
                        vi = routes[i][ai]
                        vj = routes[j][aj]
                        new_i = routes[i][:]
                        new_j = routes[j][:]
                        new_i[ai] = vj
                        new_j[aj] = vi
                        if not _route_feasible(new_i, D, instance):
                            continue
                        if not _route_feasible(new_j, D, instance):
                            continue
                        old_d = _route_dist(routes[i], D) + _route_dist(routes[j], D)
                        new_d = _route_dist(new_i, D) + _route_dist(new_j, D)
                        if new_d < old_d - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
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
    return routes


def _simulated_annealing(routes, D, instance, start_time, deadline, rng):
    """Simulated annealing with multiple move types."""
    current_routes = [r[:] for r in routes]
    current_cost = _total_dist(current_routes, D)
    best_routes = [r[:] for r in current_routes]
    best_cost = current_cost

    T = 50.0
    T_min = 0.1
    cooling = 0.995
    max_steps = 5000

    for step in range(max_steps):
        if time.time() > deadline - 10:
            break

        # Choose a random move type
        move_type = rng.choice(['swap', 'relocate', '2opt'])

        if move_type == 'swap':
            # Swap two customers between routes or within a route
            if len(current_routes) < 2 and len(current_routes[0]) < 2:
                continue
            if len(current_routes) >= 2:
                ri = rng.randint(0, len(current_routes) - 1)
                rj = rng.randint(0, len(current_routes) - 1)
                if ri == rj:
                    continue
                ai = rng.randint(0, len(current_routes[ri]) - 1)
                aj = rng.randint(0, len(current_routes[rj]) - 1)
                new_i = current_routes[ri][:]
                new_j = current_routes[rj][:]
                vi = new_i[ai]
                vj = new_j[aj]
                new_i[ai] = vj
                new_j[aj] = vi
                if not _route_feasible(new_i, D, instance) or not _route_feasible(new_j, D, instance):
                    continue
                new_cost = _total_dist(current_routes, D)
                delta = (_route_dist(new_i, D) + _route_dist(new_j, D)) - (_route_dist(current_routes[ri], D) + _route_dist(current_routes[rj], D))
            else:
                r = current_routes[0]
                if len(r) < 2:
                    continue
                ai = rng.randint(0, len(r) - 1)
                aj = rng.randint(0, len(r) - 1)
                if ai == aj:
                    continue
                new_r = r[:]
                new_r[ai], new_r[aj] = new_r[aj], new_r[ai]
                if not _route_feasible(new_r, D, instance):
                    continue
                delta = _route_dist(new_r, D) - _route_dist(r, D)
                new_i = new_r
                new_j = None
                ri = 0
                rj = None

            if new_j is not None:
                old_i = current_routes[ri][:]
                old_j = current_routes[rj][:]
                current_routes[ri] = new_i
                current_routes[rj] = new_j
                new_cost = _total_dist(current_routes, D)
            else:
                old_r = current_routes[0][:]
                current_routes[0] = new_i
                new_cost = _total_dist(current_routes, D)

            if delta < 0 or rng.random() < math.exp(-delta / max(T, T_min)):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_routes = [r[:] for r in current_routes]
            else:
                if new_j is not None:
                    current_routes[ri] = old_i
                    current_routes[rj] = old_j
                else:
                    current_routes[0] = old_r

        elif move_type == 'relocate':
            # Relocate a customer from one route to another
            if len(current_routes) < 2:
                continue
            ri = rng.randint(0, len(current_routes) - 1)
            rj = rng.randint(0, len(current_routes) - 1)
            if ri == rj:
                continue
            if not current_routes[ri]:
                continue
            ai = rng.randint(0, len(current_routes[ri]) - 1)
            aj = rng.randint(0, len(current_routes[rj]) + 1)
            v = current_routes[ri][ai]
            new_i = current_routes[ri][:]
            new_i.pop(ai)
            new_j = current_routes[rj][:]
            new_j.insert(aj, v)
            if not _route_feasible(new_i, D, instance) or not _route_feasible(new_j, D, instance):
                continue
            old_cost_part = _route_dist(current_routes[ri], D) + _route_dist(current_routes[rj], D)
            new_cost_part = _route_dist(new_i, D) + _route_dist(new_j, D)
            delta = new_cost_part - old_cost_part
            if delta < 0 or rng.random() < math.exp(-delta / max(T, T_min)):
                old_i = current_routes[ri][:]
                old_j = current_routes[rj][:]
                current_routes[ri] = new_i
                current_routes[rj] = new_j
                new_cost = _total_dist(current_routes, D)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_routes = [r[:] for r in current_routes]
            else:
                pass

        else:  # 2opt
            # 2-opt within a random route
            ri = rng.randint(0, len(current_routes) - 1)
            if len(current_routes[ri]) < 2:
                continue
            n = len(current_routes[ri])
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            old_route = current_routes[ri][:]
            new_route = old_route[:i] + old_route[i:j+1][::-1] + old_route[j+1:]
            if not _route_feasible(new_route, D, instance):
                continue
            delta = _route_dist(new_route, D) - _route_dist(old_route, D)
            if delta < 0 or rng.random() < math.exp(-delta / max(T, T_min)):
                current_routes[ri] = new_route
                new_cost = _total_dist(current_routes, D)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_routes = [r[:] for r in current_routes]

        T = max(T_min, T * cooling)

    return best_routes, best_cost


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Precompute
    coords = instance["coords"]
    D = _dist_matrix(coords)
    instance["_stations_set"] = set(instance.get("stations", []))

    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))

    # Generate diverse initial solutions
    best_routes = None
    best_cost = float('inf')

    # Solution 1: Clustering-based
    routes1 = _construct_clustering(instance, D, seed=42)
    cost1 = _total_dist(routes1, D)
    if cost1 < best_cost:
        best_cost = cost1
        best_routes = [r[:] for r in routes1]

    # Solution 2: Greedy with seed 0
    routes2 = _construct_greedy(instance, D, seed=0)
    cost2 = _total_dist(routes2, D)
    if cost2 < best_cost:
        best_cost = cost2
        best_routes = [r[:] for r in routes2]

    # Solution 3: Greedy with seed 1
    routes3 = _construct_greedy(instance, D, seed=1)
    cost3 = _total_dist(routes3, D)
    if cost3 < best_cost:
        best_cost = cost3
        best_routes = [r[:] for r in routes3]

    # Solution 4: Clustering with different seed
    routes4 = _construct_clustering(instance, D, seed=123)
    cost4 = _total_dist(routes4, D)
    if cost4 < best_cost:
        best_cost = cost4
        best_routes = [r[:] for r in routes4]

    # Optimize the best solution with local search
    if time.time() < deadline - 60:
        for i in range(len(best_routes)):
            best_routes[i] = _2opt(best_routes[i], D, instance, max_iter=200)
            best_routes[i] = _or_opt(best_routes[i], D, instance, max_iter=60)

        # Inter-route optimization
        if time.time() < deadline - 40:
            best_routes = _relocate_inter(best_routes, D, instance, max_iters=25)
            best_routes = _swap_inter(best_routes, D, instance, max_iters=12)

        # Polish
        if time.time() < deadline - 20:
            for i in range(len(best_routes)):
                best_routes[i] = _2opt(best_routes[i], D, instance, max_iter=100)
                best_routes[i] = _or_opt(best_routes[i], D, instance, max_iter=40)

    # Simulated annealing on the best solution
    if time.time() < deadline - 15:
        rng = random.Random(42)
        best_routes, best_cost = _simulated_annealing(
            best_routes, D, instance, start_time, deadline, rng
        )

    # Final feasibility check and cleanup
    final_routes = []
    for r in best_routes:
        if not r:
            continue
        if _route_feasible(r, D, instance):
            final_routes.append(r)
        else:
            # Fallback: serve individually
            for node in r:
                final_routes.append([node])

    # Ensure all customers are covered
    assigned = set()
    for r in final_routes:
        assigned.update(r)
    missing = [c for c in customers if c not in assigned]
    for c in missing:
        final_routes.append([c])

    return final_routes
# EVOLVE-BLOCK-END

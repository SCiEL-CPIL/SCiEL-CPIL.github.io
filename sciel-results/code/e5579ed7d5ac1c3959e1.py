"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _route_time_and_battery(route, D, instance):
    """Return (total_time, total_dist, min_battery, feasible) for a route."""
    speed = instance.get("speed", 1)
    battery_cap = instance.get("battery", 300)
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance.get("horizon", 1000)
    stations = set(instance.get("stations", []))
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    t = 0.0
    bat = battery_cap
    dist = 0.0
    prev = 0
    min_bat = battery_cap
    for v in route:
        d = D[prev, v]
        t += d / speed
        bat -= d
        dist += d
        if bat < -1e-9:
            return t, dist, bat, False
        min_bat = min(min_bat, bat)
        if v in stations:
            energy_needed = battery_cap - bat
            t += energy_needed * recharge_rate
            bat = battery_cap
        else:
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return t, dist, bat, False
            t += service[v]
        prev = v
    d = D[prev, 0]
    t += d / speed
    bat -= d
    dist += d
    if bat < -1e-9:
        return t, dist, bat, False
    min_bat = min(min_bat, bat)
    if t > horizon + 1e-9:
        return t, dist, bat, False
    return t, dist, min_bat, True


def _route_feasible(route, D, instance):
    if not route:
        return True
    cap = instance.get("capacity", 100)
    demands = instance["demands"]
    if sum(demands[v] for v in route) > cap + 1e-9:
        return False
    _, _, _, feas = _route_time_and_battery(route, D, instance)
    return feas


def _route_dist(route, D):
    if not route:
        return 0.0
    total = D[0, route[0]]
    for i in range(len(route) - 1):
        total += D[route[i], route[i + 1]]
    total += D[route[-1], 0]
    return total


def _construct_greedy(instance, D, deadline):
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: instance["due"][v])

    routes = []
    unassigned = set(customers)
    battery_cap = instance.get("battery", 300)
    speed = instance.get("speed", 1)
    recharge_rate = instance.get("recharge_rate", 0.5)
    cap = instance.get("capacity", 100)
    horizon = instance.get("horizon", 1000)
    stations = set(instance.get("stations", []))
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    while unassigned:
        if time.time() > deadline:
            break
        route = []
        prev = 0
        t = 0.0
        bat = battery_cap
        demand = 0

        while unassigned:
            if time.time() > deadline:
                break
            best_v = None
            best_cost = float('inf')
            for v in unassigned:
                d = D[prev, v]
                new_t = t + d / speed
                new_bat = bat - d
                if new_bat < -1e-9:
                    continue
                if demand + demands[v] > cap + 1e-9:
                    continue
                if v in stations:
                    new_t += (battery_cap - new_bat) * recharge_rate
                    new_bat = battery_cap
                else:
                    new_t = max(new_t, ready[v])
                    if new_t > due[v] + 1e-9:
                        continue
                    new_t += service[v]
                ret_d = D[v, 0]
                ret_t = new_t + ret_d / speed
                ret_bat = new_bat - ret_d
                if ret_t > horizon + 1e-9 or ret_bat < -1e-9:
                    continue
                cost = d + 0.01 * due[v]
                if cost < best_cost:
                    best_cost = cost
                    best_v = v
            if best_v is None:
                break
            d = D[prev, best_v]
            t += d / speed
            bat -= d
            demand += demands[best_v]
            if best_v in stations:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            else:
                t = max(t, ready[best_v])
                t += service[best_v]
            route.append(best_v)
            prev = best_v
            unassigned.discard(best_v)

        if route:
            routes.append(route)
        else:
            v = min(unassigned, key=lambda x: due[x])
            routes.append([v])
            unassigned.discard(v)

    return routes


def _2opt(route, D, instance, max_iter=200):
    if len(route) < 2:
        return route
    best = route[:]

    for _ in range(max_iter):
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                a = best[i - 1] if i > 0 else 0
                b = best[i]
                c = best[j]
                d = best[j + 1] if j + 1 < len(best) else 0
                old_seg = D[a, b] + D[c, d]
                new_seg = D[a, c] + D[b, d]
                if new_seg >= old_seg - 1e-9:
                    continue
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if not _route_feasible(new_route, D, instance):
                    continue
                best = new_route
                improved = True
                break
            if improved:
                break
        if not improved:
            break
    return best


def _or_opt(route, D, instance, max_iter=100):
    if len(route) < 3:
        return route
    best = route[:]

    for _ in range(max_iter):
        improved = False
        for i in range(len(best)):
            node = best[i]
            for j in range(len(best)):
                if j == i or j == i - 1:
                    continue
                new_route = best[:i] + best[i + 1:]
                ins_j = j if i > j else j - 1
                if ins_j < 0:
                    ins_j = 0
                if ins_j > len(new_route):
                    ins_j = len(new_route)
                candidate = new_route[:ins_j] + [node] + new_route[ins_j:]
                if candidate == best:
                    continue
                if not _route_feasible(candidate, D, instance):
                    continue
                if _route_dist(candidate, D) < _route_dist(best, D) - 1e-9:
                    best = candidate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _inter_route_swap(routes, D, instance, max_iters=30):
    for _ in range(max_iters):
        improved = False
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
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


def _inter_route_relocate(routes, D, instance, max_iters=30):
    for _ in range(max_iters):
        improved = False
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
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


def _cluster_by_angle(instance, D):
    """Sweep-angle based clustering for initial route assignment."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    depot = (0, 0)
    
    # Compute angle from depot for each customer
    angles = {}
    for v in customers:
        x, y = instance["coords"][v]
        angles[v] = math.atan2(y, x)
    
    # Sort by angle
    sorted_customers = sorted(customers, key=lambda v: angles[v])
    
    # Split into roughly equal clusters
    n_routes = max(1, len(customers) // 8)  # aim for ~8 customers per route
    cluster_size = len(customers) // n_routes
    remainder = len(customers) % n_routes
    
    routes = []
    idx = 0
    for r in range(n_routes):
        size = cluster_size + (1 if r < remainder else 0)
        routes.append(sorted_customers[idx:idx + size])
        idx += size
    
    return routes


def _construct_clustered(instance, D, deadline):
    """Build routes using angle-based clustering, then fill with greedy."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    battery_cap = instance.get("battery", 300)
    speed = instance.get("speed", 1)
    recharge_rate = instance.get("recharge_rate", 0.5)
    cap = instance.get("capacity", 100)
    horizon = instance.get("horizon", 1000)
    stations = set(instance.get("stations", []))
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    
    # Get initial clusters
    clusters = _cluster_by_angle(instance, D)
    
    routes = []
    unassigned = set(customers)
    
    for cluster in clusters:
        if time.time() > deadline:
            break
        # Sort cluster by due time
        cluster_sorted = sorted(cluster, key=lambda v: due[v])
        
        route = []
        prev = 0
        t = 0.0
        bat = battery_cap
        demand = 0
        
        for v in cluster_sorted:
            if v not in unassigned:
                continue
            d = D[prev, v]
            new_t = t + d / speed
            new_bat = bat - d
            if new_bat < -1e-9:
                continue
            if demand + demands[v] > cap + 1e-9:
                continue
            if v in stations:
                new_t += (battery_cap - new_bat) * recharge_rate
                new_bat = battery_cap
            else:
                new_t = max(new_t, ready[v])
                if new_t > due[v] + 1e-9:
                    continue
                new_t += service[v]
            ret_d = D[v, 0]
            ret_t = new_t + ret_d / speed
            ret_bat = new_bat - ret_d
            if ret_t > horizon + 1e-9 or ret_bat < -1e-9:
                continue
            
            # Accept
            t += d / speed
            bat -= d
            demand += demands[v]
            if v in stations:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            else:
                t = max(t, ready[v])
                t += service[v]
            route.append(v)
            prev = v
            unassigned.discard(v)
        
        if route:
            routes.append(route)
    
    # Assign remaining customers greedily
    while unassigned:
        if time.time() > deadline:
            break
        best_route_idx = -1
        best_cost = float('inf')
        best_v = None
        
        for r_idx, route in enumerate(routes):
            prev = route[-1] if route else 0
            # Get current time and battery at end of route
            t_curr, bat_curr = _get_route_end_state(route, D, instance)
            
            for v in unassigned:
                d = D[prev, v]
                new_t = t_curr + d / speed
                new_bat = bat_curr - d
                if new_bat < -1e-9:
                    continue
                if sum(demands[x] for x in route) + demands[v] > cap + 1e-9:
                    continue
                if v in stations:
                    new_t += (battery_cap - new_bat) * recharge_rate
                    new_bat = battery_cap
                else:
                    new_t = max(new_t, ready[v])
                    if new_t > due[v] + 1e-9:
                        continue
                    new_t += service[v]
                ret_d = D[v, 0]
                ret_t = new_t + ret_d / speed
                ret_bat = new_bat - ret_d
                if ret_t > horizon + 1e-9 or ret_bat < -1e-9:
                    continue
                cost = d
                if cost < best_cost:
                    best_cost = cost
                    best_route_idx = r_idx
                    best_v = v
        
        if best_v is None:
            # Create new route
            v = min(unassigned, key=lambda x: due[x])
            routes.append([v])
            unassigned.discard(v)
        else:
            route = routes[best_route_idx]
            prev = route[-1] if route else 0
            t_curr, bat_curr = _get_route_end_state(route, D, instance)
            
            d = D[prev, best_v]
            t_curr += d / speed
            bat_curr -= d
            if best_v in stations:
                t_curr += (battery_cap - bat_curr) * recharge_rate
                bat_curr = battery_cap
            else:
                t_curr = max(t_curr, ready[best_v])
                t_curr += service[best_v]
            route.append(best_v)
            unassigned.discard(best_v)
    
    return routes


def _get_route_end_state(route, D, instance):
    """Get time and battery at end of route."""
    if not route:
        return 0.0, instance.get("battery", 300)
    
    speed = instance.get("speed", 1)
    battery_cap = instance.get("battery", 300)
    recharge_rate = instance.get("recharge_rate", 0.5)
    stations = set(instance.get("stations", []))
    ready = instance["ready"]
    service = instance["service"]
    
    t = 0.0
    bat = battery_cap
    prev = 0
    for v in route:
        d = D[prev, v]
        t += d / speed
        bat -= d
        if v in stations:
            t += (battery_cap - bat) * recharge_rate
            bat = battery_cap
        else:
            t = max(t, ready[v])
            t += service[v]
        prev = v
    return t, bat


def _simulated_annealing(routes, D, instance, deadline, initial_temp=100.0, cooling=0.995):
    """Simulated annealing to improve routes."""
    best_routes = [r[:] for r in routes]
    best_dist = sum(_route_dist(r, D) for r in best_routes)
    current_routes = [r[:] for r in routes]
    current_dist = best_dist
    
    temp = initial_temp
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    
    while time.time() < deadline and temp > 0.1:
        # Generate neighbor
        new_routes = [r[:] for r in current_routes]
        
        # Random move: swap, relocate, or 2-opt
        move_type = np.random.choice(['swap', 'relocate', '2opt'])
        
        if move_type == 'swap':
            # Swap two customers from different routes
            if len(new_routes) >= 2:
                r1, r2 = np.random.choice(len(new_routes), 2, replace=False)
                if new_routes[r1] and new_routes[r2]:
                    i1 = np.random.randint(len(new_routes[r1]))
                    i2 = np.random.randint(len(new_routes[r2]))
                    new_routes[r1][i1], new_routes[r2][i2] = new_routes[r2][i2], new_routes[r1][i1]
        
        elif move_type == 'relocate':
            # Move one customer to another route
            if len(new_routes) >= 2:
                r1 = np.random.randint(len(new_routes))
                if not new_routes[r1]:
                    continue
                i1 = np.random.randint(len(new_routes[r1]))
                v = new_routes[r1].pop(i1)
                r2 = np.random.choice([x for x in range(len(new_routes)) if x != r1])
                i2 = np.random.randint(len(new_routes[r2]) + 1)
                new_routes[r2].insert(i2, v)
        
        else:  # 2opt
            # 2-opt within one route
            if new_routes:
                r_idx = np.random.randint(len(new_routes))
                route = new_routes[r_idx]
                if len(route) >= 2:
                    i = np.random.randint(len(route) - 1)
                    j = np.random.randint(i + 1, len(route))
                    route[i:j+1] = route[i:j+1][::-1]
        
        # Check feasibility
        feasible = all(_route_feasible(r, D, instance) for r in new_routes)
        if not feasible:
            temp *= cooling
            continue
        
        new_dist = sum(_route_dist(r, D) for r in new_routes)
        delta = new_dist - current_dist
        
        # Acceptance criterion
        if delta < 0 or np.random.random() < math.exp(-delta / max(temp, 1e-9)):
            current_routes = new_routes
            current_dist = new_dist
            
            if current_dist < best_dist:
                best_dist = current_dist
                best_routes = [r[:] for r in current_routes]
        
        temp *= cooling
    
    return best_routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    coords = instance["coords"]
    D = _dist_matrix(coords)

    # Try multiple construction methods and pick the best
    best_routes = None
    best_dist = float('inf')
    
    # Method 1: Due-time sorted greedy
    routes1 = _construct_greedy(instance, D, deadline)
    dist1 = sum(_route_dist(r, D) for r in routes1)
    if dist1 < best_dist:
        best_dist = dist1
        best_routes = [r[:] for r in routes1]
    
    # Method 2: Angle-based clustering
    if time.time() < deadline:
        routes2 = _construct_clustered(instance, D, deadline)
        dist2 = sum(_route_dist(r, D) for r in routes2)
        if dist2 < best_dist:
            best_dist = dist2
            best_routes = [r[:] for r in routes2]
    
    if best_routes is None:
        return []
    
    # Local search: 2-opt and Or-opt within routes
    for r_idx in range(len(best_routes)):
        if time.time() > deadline:
            break
        best_routes[r_idx] = _2opt(best_routes[r_idx], D, instance)
        best_routes[r_idx] = _or_opt(best_routes[r_idx], D, instance)
    
    if time.time() > deadline:
        return best_routes
    
    # Inter-route improvements
    best_routes = _inter_route_swap(best_routes, D, instance)
    best_routes = _inter_route_relocate(best_routes, D, instance)
    
    # Final 2-opt pass
    for r_idx in range(len(best_routes)):
        if time.time() > deadline:
            break
        best_routes[r_idx] = _2opt(best_routes[r_idx], D, instance)
    
    # Simulated annealing if time permits
    if time.time() < deadline - 10:
        sa_routes = _simulated_annealing(best_routes, D, instance, deadline - 5)
        sa_dist = sum(_route_dist(r, D) for r in sa_routes)
        if sa_dist < best_dist:
            best_routes = sa_routes
    
    return best_routes
# EVOLVE-BLOCK-END

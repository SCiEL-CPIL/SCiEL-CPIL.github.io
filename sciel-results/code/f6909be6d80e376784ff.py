"""EV routing with time windows: NN construction + Iterated Local Search."""
# EVOLVE-BLOCK-START
import math
import time
import random

_EPS = 1e-9


def _build(instance):
    coords = instance["coords"]
    n = len(coords)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        row = D[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = coords[j]
            row[j] = math.hypot(xi - xj, yi - yj)
    return D


def _simulate(route, instance, D, stations_set):
    """Return (ok, total_distance)."""
    if not route:
        return True, 0.0
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    load = 0.0
    battery = float(battery_max)
    t = 0.0
    prev = 0
    total = 0.0

    for node in route:
        d = D[prev][node]
        if d > battery + _EPS:
            return False, total
        battery -= d
        t += d / speed
        total += d
        prev = node
        if node in stations_set:
            if battery < 0.0:
                return False, total
            charge = battery_max - battery
            if charge > 0.0:
                t += charge * recharge
            battery = float(battery_max)
            continue
        load += demands[node]
        if load > capacity + _EPS:
            return False, total
        if t < ready[node]:
            t = float(ready[node])
        if t > due[node] + _EPS:
            return False, total
        t += service[node]
        if t > horizon + _EPS:
            return False, total

    d = D[prev][0]
    if d > battery + _EPS:
        return False, total
    battery -= d
    t += d / speed
    total += d
    if t > horizon + _EPS:
        return False, total
    return True, total


def _route_dist(route, D):
    if not route:
        return 0.0
    total = D[0][route[0]]
    for i in range(1, len(route)):
        total += D[route[i - 1]][route[i]]
    total += D[route[-1]][0]
    return total


def _total_distance(routes, D):
    return sum(_route_dist(r, D) for r in routes)


def _routes_valid(routes, instance, D, stations_set):
    seen = set()
    for r in routes:
        ok, _ = _simulate(r, instance, D, stations_set)
        if not ok:
            return False
        for node in r:
            if node in stations_set or node == 0:
                continue
            if node in seen:
                return False
            seen.add(node)
    n_cust = instance["n_customers"]
    if len(seen) != n_cust:
        return False
    return True


def _nn_route(instance, D, unvisited, stations_set, rng):
    """Build a feasible route via nearest neighbor, inserting charging stops."""
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    route = []
    load = 0.0
    battery = float(battery_max)
    t = 0.0
    cur = 0
    remaining = set(unvisited)
    if not remaining:
        return []

    def feasible_from(node, battery, t, load):
        d = D[cur][node]
        if d > battery + _EPS:
            return False
        nb = battery - d
        nt = t + d / speed
        if node in stations_set:
            return nb >= -_EPS
        nl = load + demands[node]
        if nl > capacity + _EPS:
            return False
        if nt < ready[node]:
            nt = float(ready[node])
        if nt > due[node] + _EPS:
            return False
        nt += service[node]
        return nt <= horizon + _EPS

    # nearest feasible customer
    first = None
    best_d = None
    for c in remaining:
        if not feasible_from(c, battery, t, load):
            continue
        dd = D[cur][c]
        if best_d is None or dd < best_d:
            best_d = dd
            first = c
    if first is None:
        return []

    def do_customer(node):
        nonlocal battery, t, cur, load
        d = D[cur][node]
        battery -= d
        t += d / speed
        cur = node
        load += demands[node]
        if t < ready[node]:
            t = float(ready[node])
        t += service[node]
        route.append(node)
        remaining.discard(node)

    def do_station(node):
        nonlocal battery, t, cur
        d = D[cur][node]
        battery -= d
        t += d / speed
        cur = node
        charge = battery_max - battery
        if charge > 0.0:
            t += charge * recharge
        battery = float(battery_max)
        route.append(node)

    do_customer(first)

    def can_return():
        d = D[cur][0]
        return d <= battery + _EPS

    while remaining:
        nextc = None
        best_d = None
        for c in remaining:
            if not feasible_from(c, battery, t, load):
                continue
            dd = D[cur][c]
            if best_d is None or dd < best_d:
                best_d = dd
                nextc = c
        if nextc is not None:
            do_customer(nextc)
        else:
            if can_return():
                break
            best_st = None
            best_sd = None
            for s in instance["stations"]:
                dd = D[cur][s]
                if dd <= battery + _EPS:
                    if best_sd is None or dd < best_sd:
                        best_sd = dd
                        best_st = s
            if best_st is None:
                break
            battery2 = battery - best_sd
            t2 = t + best_sd / speed
            charge = battery_max - battery2
            t2 += charge * recharge
            battery2 = float(battery_max)
            dret = D[best_st][0]
            if dret <= battery2 + _EPS and t2 + dret / speed <= horizon + _EPS:
                do_station(best_st)
                break
            found = False
            bestc = None
            bcd = None
            for c in remaining:
                d = D[best_st][c]
                if d > battery2 + _EPS:
                    continue
                nt = t2 + d / speed
                nl = load + demands[c]
                if nl > capacity + _EPS:
                    continue
                if nt < ready[c]:
                    nt = float(ready[c])
                if nt > due[c] + _EPS:
                    continue
                nt += service[c]
                if nt > horizon + _EPS:
                    continue
                if bcd is None or d < bcd:
                    bcd = d
                    bestc = c
            if bestc is not None:
                do_station(best_st)
                continue
            else:
                do_station(best_st)
                break
    return route


def _construct(instance, D, stations_set, deadline, rng):
    customers = list(range(1, instance["n_customers"] + 1))
    customers.sort(key=lambda v: (instance["due"][v], v))
    unvisited = set(customers)
    routes = []
    while unvisited:
        if time.time() >= deadline:
            break
        r = _nn_route(instance, D, unvisited, stations_set, rng)
        if not r:
            c = min(unvisited, key=lambda v: (D[0][v], v))
            r = [c]
        ok, _ = _simulate(r, instance, D, stations_set)
        if not ok:
            # Simple fix: try inserting stations
            fixed = None
            for s in instance["stations"]:
                for p in range(len(r) + 1):
                    cand = r[:p] + [s] + r[p:]
                    ok2, _ = _simulate(cand, instance, D, stations_set)
                    if ok2:
                        fixed = cand
                        break
                if fixed:
                    break
            if fixed:
                r = fixed
            else:
                # Fallback: split into singletons if necessary? 
                # For now, keep as is and hope local search fixes it, 
                # or just accept invalid and let ILS handle? 
                # No, ILS needs valid start. 
                # If NN failed, try single customer route.
                r = [r[0]] if r else [c]
                # Remove others from unvisited? No, we need to serve everyone.
                # If we can't serve them in this route, we must start a new route.
                # The loop `while unvisited` handles this by starting a new route if `r` is small.
                # But `r` might contain multiple customers that are infeasible together.
                # So we should only keep the first customer in `r` if it's infeasible?
                # Better: if infeasible, reduce `r` to just the first customer.
                # And re-add the rest to unvisited.
                removed = set(r[1:])
                unvisited.update(removed)
                r = [r[0]]
        unvisited -= set(r)
        routes.append(r)
    return routes


def _local_optimize(routes, instance, D, stations_set, deadline):
    """Apply 2-opt and Or-opt improvements."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_routes = len(routes)
        
        # 2-opt within routes
        for ri in range(n_routes):
            r = routes[ri]
            m = len(r)
            if m < 3:
                continue
            for i in range(0, m - 1):
                for j in range(i + 1, m):
                    cand = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                    ok, _ = _simulate(cand, instance, D, stations_set)
                    if ok and _route_dist(cand, D) < _route_dist(r, D) - _EPS:
                        routes[ri] = cand
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # Or-opt within routes (1-3 moves)
        if not improved:
            for ri in range(n_routes):
                r = routes[ri]
                m = len(r)
                if m < 2:
                    continue
                for seg_len in [1, 2, 3]:
                    for i in range(m - seg_len + 1):
                        seg = r[i:i + seg_len]
                        if any(n in stations_set for n in seg):
                            continue
                        rest = r[:i] + r[i + seg_len:]
                        for j in range(len(rest) + 1):
                            if j == i or j == i + seg_len:
                                continue
                            cand = rest[:j] + seg + rest[j:]
                            ok, _ = _simulate(cand, instance, D, stations_set)
                            if ok and _route_dist(cand, D) < _route_dist(r, D) - _EPS:
                                routes[ri] = cand
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Relocate between routes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_routes = len(routes)
        for a in range(n_routes):
            ra = routes[a]
            for pos in range(len(ra)):
                node = ra[pos]
                if node in stations_set:
                    continue
                rest_a = ra[:pos] + ra[pos + 1:]
                for b in range(n_routes):
                    if b == a:
                        continue
                    rb = routes[b]
                    for q in range(len(rb) + 1):
                        cand_b = rb[:q] + [node] + rb[q:]
                        okb, _ = _simulate(cand_b, instance, D, stations_set)
                        oka, _ = _simulate(rest_a, instance, D, stations_set)
                        if oka and okb:
                            new_total = _route_dist(cand_b, D) + _route_dist(rest_a, D)
                            old_total = _route_dist(ra, D) + _route_dist(rb, D)
                            if new_total < old_total - _EPS:
                                routes[a] = rest_a
                                routes[b] = cand_b
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    routes = [r for r in routes if r]
    return routes


def _perturb(routes, instance, D, stations_set, rng):
    """Remove 3-5 random customers and re-insert them at best positions."""
    n_remove = rng.randint(3, 5)
    removed = []
    new_routes = [r[:] for r in routes]
    
    # Collect all customers
    all_customers = []
    for r in new_routes:
        for node in r:
            if node not in stations_set:
                all_customers.append(node)
    
    if not all_customers:
        return new_routes

    # Randomly select customers to remove
    to_remove = rng.sample(all_customers, min(n_remove, len(all_customers)))
    to_remove_set = set(to_remove)
    
    for i in range(len(new_routes)):
        new_routes[i] = [n for n in new_routes[i] if n not in to_remove_set]
    
    # Re-insert
    for node in to_remove:
        best_pos = None
        best_dist = float('inf')
        best_route_idx = -1
        best_insert_pos = -1
        
        for ri, r in enumerate(new_routes):
            # Try inserting into each position
            for q in range(len(r) + 1):
                cand = r[:q] + [node] + r[q:]
                ok, dist = _simulate(cand, instance, D, stations_set)
                if ok:
                    # Calculate total distance change? 
                    # Just use distance of new route + rest of solution
                    # Approximation: just check if feasible and pick min dist of this route
                    # Better: calculate total delta
                    # For simplicity, pick min dist of the modified route
                    if dist < best_dist:
                        best_dist = dist
                        best_route_idx = ri
                        best_insert_pos = q
                        best_pos = cand
        
        if best_route_idx != -1:
            new_routes[best_route_idx] = best_pos
        else:
            # If can't insert, create new route
            new_routes.append([node])
            
    return new_routes


def plan(instance):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin
    stations_set = set(instance["stations"])
    D = _build(instance)
    
    rng = random.Random(42)
    
    # 1. Construction
    routes = _construct(instance, D, stations_set, deadline, rng)
    
    if not _routes_valid(routes, instance, D, stations_set):
        # Fallback to singletons if construction failed badly
        routes = [[v] for v in range(1, instance["n_customers"] + 1)]
    
    # 2. Initial Local Optimization
    routes = _local_optimize(routes, instance, D, stations_set, deadline)
    
    best_routes = [r[:] for r in routes]
    best_dist = _total_distance(best_routes, D)
    
    # 3. Iterated Local Search
    current_routes = [r[:] for r in routes]
    current_dist = best_dist
    
    T = 50.0  # Initial temperature
    cooling = 0.99
    
    while time.time() < deadline - 5:  # Stop 5s before deadline
        # Perturb
        perturbed = _perturb(current_routes, instance, D, stations_set, rng)
        
        # Local Optimize the perturbed solution
        improved = _local_optimize(perturbed, instance, D, stations_set, deadline)
        
        new_dist = _total_distance(improved, D)
        
        # Acceptance criterion
        delta = new_dist - current_dist
        if delta < 0 or (T > 1e-6 and rng.random() < math.exp(-delta / T)):
            current_routes = improved
            current_dist = new_dist
            
            if current_dist < best_dist:
                best_dist = current_dist
                best_routes = [r[:] for r in current_routes]
        
        T *= cooling
    
    # Final validity check
    if not _routes_valid(best_routes, instance, D, stations_set):
        best_routes = [[v] for v in range(1, instance["n_customers"] + 1)]
        
    return best_routes
# EVOLVE-BLOCK-END

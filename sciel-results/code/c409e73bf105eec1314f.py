# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _check_route(route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon):
    """Check if a route is feasible. Returns (feasible, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0
    
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return False, 0.0, 0.0
    
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0  # depot
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        time_t += d / speed
        total_dist += d
        
        if battery < -1e-9:
            return False, 0.0, 0.0
        
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    time_t += d / speed
    total_dist += d
    
    if battery < -1e-9:
        return False, 0.0, 0.0
    if time_t > horizon + 1e-9:
        return False, 0.0, 0.0
    
    return True, total_dist, time_t


def _route_distance(route, dist):
    """Quick distance calculation for a route."""
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for v in route:
        total += dist[prev][v]
        prev = v
    total += dist[prev][0]
    return total


def _cluster_customers(coords, n_customers, n_clusters=5):
    """Simple k-means-like clustering to group spatially close customers."""
    customers = list(range(1, n_customers + 1))
    random.shuffle(customers)
    
    # Initialize centroids
    centroids = [np.array(coords[c], dtype=float) for c in customers[:n_clusters]]
    assignments = {}
    
    for _ in range(10):
        new_assignments = {}
        for c in customers:
            best_cluster = 0
            best_dist = float('inf')
            for i, cent in enumerate(centroids):
                d = np.linalg.norm(np.array(coords[c]) - cent)
                if d < best_dist:
                    best_dist = d
                    best_cluster = i
            new_assignments[c] = best_cluster
        
        if new_assignments == assignments:
            break
        assignments = new_assignments
        
        # Update centroids
        for i in range(n_clusters):
            members = [c for c in customers if assignments[c] == i]
            if members:
                centroids[i] = np.mean([np.array(coords[c]) for c in members], axis=0)
    
    clusters = [[] for _ in range(n_clusters)]
    for c in customers:
        clusters[assignments[c]].append(c)
    
    return clusters


def _construct_cluster_routes(instance, dist, time_limit=5):
    """Construct initial routes using spatial clustering + time-window-aware insertion."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Cluster customers spatially
    n_clusters = min(5, n_customers)
    clusters = _cluster_customers(coords, n_customers, n_clusters)
    
    routes = []
    
    for cluster in clusters:
        if not cluster:
            continue
        
        # Sort by due date within cluster
        cluster.sort(key=lambda c: due[c])
        
        current_route = []
        current_demand = 0
        
        for c in cluster:
            if time.time() > deadline:
                break
            
            if current_demand + demands[c] > capacity:
                if current_route:
                    routes.append(current_route)
                    current_route = []
                    current_demand = 0
                else:
                    routes.append([c])
                    continue
            
            # Try to insert c into current_route at best feasible position
            best_pos = -1
            best_cost = float('inf')
            
            for pos in range(len(current_route) + 1):
                new_route = current_route[:pos] + [c] + current_route[pos:]
                feasible, d, t = _check_route(new_route, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
                if feasible and d < best_cost:
                    best_cost = d
                    best_pos = pos
            
            if best_pos != -1:
                current_route = current_route[:best_pos] + [c] + current_route[best_pos:]
                current_demand += demands[c]
            else:
                # Can't insert, start new route with just this customer
                if current_route:
                    routes.append(current_route)
                current_route = [c]
                current_demand = demands[c]
        
        if current_route:
            routes.append(current_route)
    
    # Check for any missing customers
    all_served = set()
    for r in routes:
        all_served.update(r)
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        routes.append([c])
    
    return routes


class _RouteState:
    """Maintains prefix/suffix feasibility info for fast insertion checks."""
    
    def __init__(self, route, dist, instance):
        self.route = list(route)
        self.dist = dist
        self.instance = instance
        self.battery_max = instance["battery"]
        self.capacity = instance["capacity"]
        self.speed = instance["speed"]
        self.recharge_rate = instance["recharge_rate"]
        self.horizon = instance["horizon"]
        self.n_customers = instance["n_customers"]
        self.demands = instance["demands"]
        self.ready = instance["ready"]
        self.due = instance["due"]
        self.service = instance["service"]
        
        self.total_dist = 0.0
        self.total_demand = 0
        self._recompute()
    
    def _recompute(self):
        """Recompute all prefix/suffix states."""
        n = len(self.route)
        d = self.dist
        
        # Prefix: state after visiting route[0..i]
        # prefix_time[i] = time at end of service/charge at node i
        # prefix_batt[i] = battery after leaving node i (i.e., at start of travel to i+1)
        # prefix_dist[i] = cumulative distance up to and including edge to node i
        
        self.prefix_time = [0.0] * n
        self.prefix_batt = [0.0] * n
        self.prefix_dist = [0.0] * n
        
        battery = self.battery_max
        t = 0.0
        total_d = 0.0
        total_dem = 0
        
        for i, v in enumerate(self.route):
            prev = 0 if i == 0 else self.route[i-1]
            dd = d[prev][v]
            battery -= dd
            t += dd / self.speed
            total_d += dd
            
            if v <= self.n_customers:
                total_dem += self.demands[v]
                if t < self.ready[v]:
                    t = self.ready[v]
                # t <= due[v] must hold for feasibility
                t += self.service[v]
            else:
                energy_needed = self.battery_max - battery
                t += energy_needed * self.recharge_rate
                battery = self.battery_max
            
            self.prefix_time[i] = t
            self.prefix_batt[i] = battery
            self.prefix_dist[i] = total_d
        
        # Suffix: state before arriving at node i from depot side
        # We need: if we insert at position pos, the travel from route[pos-1] to new node,
        # then from new node to route[pos], etc.
        # suffix_time[i] = time needed from node i to depot (including service at i)
        # suffix_batt[i] = battery needed at node i to reach depot
        
        self.suffix_time = [0.0] * n
        self.suffix_batt = [0.0] * n
        self.suffix_dist = [0.0] * n
        
        # Compute from end
        # suffix at last node: travel to depot
        # We'll compute "cost to finish" from each position
        
        # Let's define: from position i (at node route[i] after service), 
        # what's the time and battery to get back to depot?
        # This is tricky with charging stations. Let's just store the full route check.
        
        self.total_dist = total_d + d[self.route[-1]][0] if n > 0 else 0.0
        self.total_demand = total_dem
        
        # For fast insertion, we'll just do full check but with early termination
        # Actually, let's precompute a "forward" and "backward" pass
    
    def check_insertion(self, customer, pos):
        """Quick check if inserting customer at pos is feasible. Returns (feasible, new_dist)."""
        route = self.route
        d = self.dist
        n = len(route)
        
        # New route would be route[:pos] + [customer] + route[pos:]
        # Check demand
        if self.total_demand + self.demands[customer] > self.capacity:
            return False, float('inf')
        
        # Estimate distance change
        if n == 0:
            new_dist = d[0][customer] + d[customer][0]
        else:
            if pos == 0:
                old_edge = d[0][route[0]]
                new_edges = d[0][customer] + d[customer][route[0]]
            elif pos == n:
                old_edge = d[route[-1]][0]
                new_edges = d[route[-1]][customer] + d[customer][0]
            else:
                old_edge = d[route[pos-1]][route[pos]]
                new_edges = d[route[pos-1]][customer] + d[customer][route[pos]]
            delta = new_edges - old_edge
            new_dist = self.total_dist + delta
        
        # Quick feasibility: check if new_dist is reasonable and battery allows
        # For full check, we'd need to simulate. Let's do a lightweight simulation.
        battery = self.battery_max
        t = 0.0
        prev = 0
        
        for i, v in enumerate(route):
            if i == pos:
                # Insert customer before v
                dd = d[prev][customer]
                battery -= dd
                t += dd / self.speed
                if battery < -1e-9:
                    return False, float('inf')
                if t < self.ready[customer]:
                    t = self.ready[customer]
                if t > self.due[customer] + 1e-9:
                    return False, float('inf')
                t += self.service[customer]
                prev = customer
                # Now continue to v
                dd2 = d[customer][v]
                battery -= dd2
                t += dd2 / self.speed
                if battery < -1e-9:
                    return False, float('inf')
                # Continue with v
                if v <= self.n_customers:
                    if t < self.ready[v]:
                        t = self.ready[v]
                    if t > self.due[v] + 1e-9:
                        return False, float('inf')
                    t += self.service[v]
                else:
                    energy_needed = self.battery_max - battery
                    t += energy_needed * self.recharge_rate
                    battery = self.battery_max
                prev = v
            else:
                dd = d[prev][v]
                battery -= dd
                t += dd / self.speed
                if battery < -1e-9:
                    return False, float('inf')
                
                if v <= self.n_customers:
                    if t < self.ready[v]:
                        t = self.ready[v]
                    if t > self.due[v] + 1e-9:
                        return False, float('inf')
                    t += self.service[v]
                else:
                    energy_needed = self.battery_max - battery
                    t += energy_needed * self.recharge_rate
                    battery = self.battery_max
                prev = v
        
        # Check return to depot
        dd = d[prev][0]
        battery -= dd
        t += dd / self.speed
        if battery < -1e-9:
            return False, float('inf')
        if t > self.horizon + 1e-9:
            return False, float('inf')
        
        return True, new_dist


def _local_search_fast(routes, dist, instance, time_limit=30):
    """Fast local search using RouteState for quick insertion checks."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Create RouteState for each route
    states = [_RouteState(r, dist, instance) for r in routes]
    
    def total_distance():
        return sum(s.total_dist for s in states if s.route)
    
    best_total = total_distance()
    
    while time.time() < deadline - 1:
        improved = False
        
        # Relocate: move one customer from route i to best position in route j
        for i in range(len(states)):
            if not states[i].route:
                continue
            ri = states[i].route
            for pos_i in range(len(ri)):
                c = ri[pos_i]
                if c > instance["n_customers"]:
                    continue
                
                # Try inserting c into each other route
                for j in range(len(states)):
                    if i == j:
                        continue
                    rj = states[j].route
                    best_pos = -1
                    best_cost = float('inf')
                    
                    for pos_j in range(len(rj) + 1):
                        feasible, new_dist = states[j].check_insertion(c, pos_j)
                        if feasible:
                            # Cost = (new_dist_j - old_dist_j) + (new_dist_i - old_dist_i)
                            # Approximate: just use new_dist_j + adjusted new_dist_i
                            # For simplicity, compute full new distances
                            new_route_j = rj[:pos_j] + [c] + rj[pos_j:]
                            new_route_i = ri[:pos_i] + ri[pos_i+1:]
                            
                            # Quick distance estimate
                            if new_route_i:
                                d_i = _route_distance(new_route_i, dist)
                            else:
                                d_i = 0.0
                            d_j = _route_distance(new_route_j, dist)
                            
                            total_new = d_i + d_j
                            total_old = states[i].total_dist + states[j].total_dist
                            
                            if total_new < best_cost:
                                best_cost = total_new
                                best_pos = pos_j
                    
                    if best_pos != -1:
                        total_old = states[i].total_dist + states[j].total_dist
                        if best_cost < total_old - 1e-9:
                            # Apply move
                            new_route_i = ri[:pos_i] + ri[pos_i+1:]
                            new_route_j = rj[:best_pos] + [c] + rj[best_pos:]
                            
                            # Verify feasibility
                            feas_i, d_i, t_i = _check_route(new_route_i, dist, instance, 
                                instance["battery"], instance["capacity"], instance["speed"], 
                                instance["recharge_rate"], instance["horizon"])
                            feas_j, d_j, t_j = _check_route(new_route_j, dist, instance, 
                                instance["battery"], instance["capacity"], instance["speed"], 
                                instance["recharge_rate"], instance["horizon"])
                            
                            if feas_i and feas_j:
                                states[i].route = new_route_i
                                states[i].total_dist = d_i
                                states[j].route = new_route_j
                                states[j].total_dist = d_j
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            # Clean up empty routes
            states = [s for s in states if s.route]
            continue
        
        # Or-Opt within route: move 1-2 consecutive customers
        for i in range(len(states)):
            r = states[i].route
            n = len(r)
            if n < 3:
                continue
            
            for L in [1, 2]:
                if n < L + 1:
                    continue
                for start in range(n - L):
                    moving = r[start:start+L]
                    # Only move if all are customers
                    if any(m > instance["n_customers"] for m in moving):
                        continue
                    
                    rest = r[:start] + r[start+L:]
                    best_pos = -1
                    best_cost = float('inf')
                    
                    for pos in range(len(rest) + 1):
                        # Skip same position
                        if pos == start or (start > 0 and pos == start + L) or (start == 0 and pos == L):
                            continue
                        
                        new_route = rest[:pos] + moving + rest[pos:]
                        if new_route == r:
                            continue
                        
                        d_new = _route_distance(new_route, dist)
                        if d_new < best_cost:
                            best_cost = d_new
                            best_pos = pos
                    
                    if best_pos != -1 and best_cost < states[i].total_dist - 1e-9:
                        new_route = rest[:best_pos] + moving + rest[best_pos:]
                        feas, d, t = _check_route(new_route, dist, instance, 
                            instance["battery"], instance["capacity"], instance["speed"], 
                            instance["recharge_rate"], instance["horizon"])
                        if feas:
                            states[i].route = new_route
                            states[i].total_dist = d
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # 2-Opt within route
        for i in range(len(states)):
            r = states[i].route
            n = len(r)
            if n < 2:
                continue
            
            found = False
            for a in range(n - 1):
                if found:
                    break
                for b in range(a + 1, n):
                    new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                    if new_r == r:
                        continue
                    
                    d_new = _route_distance(new_r, dist)
                    if d_new < states[i].total_dist - 1e-9:
                        feas, d, t = _check_route(new_r, dist, instance, 
                            instance["battery"], instance["capacity"], instance["speed"], 
                            instance["recharge_rate"], instance["horizon"])
                        if feas:
                            states[i].route = new_r
                            states[i].total_dist = d
                            improved = True
                            found = True
                            break
            if improved:
                break
        
        if not improved:
            break
    
    return [s.route for s in states if s.route]


def _solve_instance(instance, time_limit=45):
    """Solve a single instance."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _dist_matrix(coords)
    
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_customers = instance["n_customers"]
    
    # 1. Construct initial routes using clustering
    construct_time = min(3, time_limit * 0.1)
    routes = _construct_cluster_routes(instance, dist, construct_time)
    
    # Ensure all customers are served
    all_served = set()
    for r in routes:
        all_served.update(r)
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        routes.append([c])
    
    # 2. Local Search
    ls_time = deadline - time.time() - 2
    if ls_time > 1:
        routes = _local_search_fast(routes, dist, instance, ls_time)
    
    # Final verification
    valid_routes = []
    for r in routes:
        feasible, _, _ = _check_route(r, dist, instance, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
    
    # Check all customers are served
    all_served = set()
    for r in valid_routes:
        all_served.update(r)
    
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    for c in missing:
        valid_routes.append([c])
    
    return valid_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    
    # 3 instances, 150s total, leave 15s buffer -> 45s per instance
    per_instance_time = 45
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END

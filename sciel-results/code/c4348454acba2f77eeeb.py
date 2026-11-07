import math
import time
import random
import heapq
import numpy as np
from scipy.spatial.distance import cdist

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin within 150s kill

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

    # Precompute distance matrix
    n_nodes = len(coords)
    dist = cdist(coords, coords)
    
    customers = list(range(1, n_cust + 1))
    
    # Helper to evaluate a route: returns (valid, distance, time)
    def check_route(route_nodes):
        if not route_nodes:
            return True, 0.0, 0.0
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
                return False, 0.0, 0.0
            
            # Charging station logic
            if node in stations:
                # "station adds (battery - energy) * recharge_rate time and refills"
                # Interpretation: time_to_charge = (battery_cap - current_battery) * recharge_rate
                # Then battery is refilled to battery_cap.
                time_to_charge = (battery_cap - b) * recharge_rate
                t += time_to_charge
                b = battery_cap
            else:
                # Customer logic
                t = max(t, ready[node])
                if t > due[node] + _EPS:
                    return False, 0.0, 0.0
                t += service[node]
                load += demands[node]
                if load > capacity + _EPS:
                    return False, 0.0, 0.0
            
            d_total += d
            prev = node
        
        # Return to depot
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -_EPS:
            return False, 0.0, 0.0
        if t > horizon + _EPS:
            return False, 0.0, 0.0
        d_total += d
        return True, d_total, t

    def evaluate_routes(routes):
        """Returns (valid, total_distance) for a list of routes."""
        total_d = 0.0
        for r in routes:
            if not r: continue
            valid, d, _ = check_route(r)
            if not valid:
                return False, 0.0
            total_d += d
        return True, total_d

    # --- Strategy: Constrained Shortest Path on State-Space Graph ---
    # Nodes in graph: (customer_set, current_node, battery_level, time_bucket)
    # This is too large for 25 customers. 
    # However, the prompt suggests: "nodes represent (customer, time_bucket, battery_level)".
    # This implies a single-vehicle tour or a simplified state space.
    # With 25 customers, a full state-space search for the entire tour is infeasible.
    # But we can use this for LOCAL REPAIR or for building routes greedily using Dijkstra-like logic.
    
    # Alternative interpretation: Use Dijkstra to find the best way to visit a SET of customers 
    # given a starting state, or to optimize a single route given a set of customers.
    
    # Let's try a different structural approach:
    # 1. Cluster customers into groups using K-Means (K determined by capacity).
    # 2. For each cluster, find the optimal single-vehicle tour visiting those customers 
    #    (plus potentially stations) using a Dijkstra-based exact solver on a reduced state space 
    #    if the cluster size is small (e.g., <= 8). If larger, use heuristic.
    # 3. Since Dijkstra on state space (visited_mask, node, battery, time) is exponential in customers, 
    #    we can only do this for small subsets.
    
    # Let's refine:
    # The "Constrained Shortest Path" idea is powerful for small instances.
    # With 25 customers, we can't solve the whole thing exactly.
    # But we can use it to optimize each route independently after clustering.
    
    # Step 1: Clustering
    # Estimate number of routes
    total_demand = np.sum(demands[1:n_cust+1])
    K = max(3, int(np.ceil(total_demand / capacity)))
    
    # Use K-Means to partition customers into K clusters
    # Simple K-Means implementation
    rng = random.Random(42)
    
    # Initialize centroids randomly from customer coordinates
    cust_coords = coords[1:n_cust+1]
    centroids = np.array([cust_coords[rng.randrange(n_cust)] for _ in range(K)])
    
    for _ in range(10): # 10 iterations of K-Means
        # Assign customers to nearest centroid
        distances = cdist(cust_coords, centroids)
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = []
        for k in range(K):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centroids.append(np.mean(cust_coords[mask], axis=0))
            else:
                # If empty, re-initialize to a random customer
                new_centroids.append(cust_coords[rng.randrange(n_cust)])
        new_centroids = np.array(new_centroids)
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # Create clusters
    clusters = [[] for _ in range(K)]
    for i, label in enumerate(labels):
        clusters[label].append(i + 1) # customer id
    
    # Step 2: Optimize each cluster's route using Dijkstra on state space
    # State: (visited_mask, current_node, battery, time)
    # Since cluster size is small (approx 25/K), we can afford a small state space.
    # If K=5, cluster size ~5. 2^5 = 32 masks. Nodes ~5+stations+depot.
    # Battery and time are continuous. We need to discretize or use exact arithmetic.
    # Exact arithmetic with floats is tricky for Dijkstra keys.
    # We can use (distance, time, battery) as a tuple for the heap, but Dijkstra requires a single scalar cost.
    # We want to minimize distance. But constraints are on time and battery.
    # This is a multi-objective or constrained shortest path.
    # We can define cost = distance, and prune states if time > horizon or battery < 0.
    # Also, we need to ensure we return to depot.
    
    def optimize_cluster(cluster_customers):
        """
        Find optimal route for a set of customers using Dijkstra on state space.
        Returns list of node ids (excluding depot at start/end).
        """
        if not cluster_customers:
            return []
        
        # Nodes involved: 0 (depot), cluster_customers, and stations?
        # Stations can be visited multiple times. This complicates the state space.
        # If we allow arbitrary station visits, the state space explodes.
        # Assumption: We can only visit a station if it helps. 
        # For small clusters, we might not need stations if the battery is enough.
        # Let's check if the cluster can be served without stations first.
        
        # Simple heuristic check: max distance in cluster + return
        # If battery_cap is sufficient, no station needed.
        
        # For Dijkstra, let's define state as (visited_mask, current_node, battery, time)
        # We discretize battery and time? No, that loses precision.
        # Instead, we use a label-setting algorithm.
        # Labels: (current_node, battery, time, distance, path)
        # We keep the best distance for each (visited_mask, current_node, battery, time) bucket?
        # Too many buckets.
        
        # Alternative: Since the cluster is small, we can use Held-Karp (TSP DP) 
        # with additional state for battery and time.
        # DP[mask][i] = list of (time, battery, dist) labels for reaching node i having visited mask.
        # We prune dominated labels: if label A has time <= B, battery >= B, dist <= B, then B is dominated.
        
        n_c = len(cluster_customers)
        if n_c > 8:
            # Too large for exact DP with labels. Fall back to heuristic.
            return heuristic_route(cluster_customers)
        
        # Nodes in this subproblem: 0 (depot), 1..n_c (customers)
        # Map customer ids to local indices 1..n_c
        cust_map = {c: i+1 for i, c in enumerate(cluster_customers)}
        
        # DP table: dp[mask][i] = list of labels (time, battery, dist)
        # mask: bitmask of visited customers
        # i: current node (0 for depot, 1..n_c for customers)
        
        dp = [[[] for _ in range(n_c + 1)] for _ in range(1 << n_c)]
        
        # Start at depot (node 0)
        # Initial state: mask=0, node=0, time=0, battery=battery_cap, dist=0
        dp[0][0] = [(0.0, battery_cap, 0.0)]
        
        # Iterate over masks
        for mask in range(1 << n_c):
            for i in range(n_c + 1):
                if not dp[mask][i]:
                    continue
                
                # For each label at (mask, i), try to go to next customer j
                for (t, b, d) in dp[mask][i]:
                    # Try to go to any unvisited customer j
                    for j in range(1, n_c + 1):
                        if mask & (1 << (j - 1)):
                            continue
                        
                        node_j = cluster_customers[j - 1]
                        d_ij = dist[i, node_j] # i is local index, need to map to global?
                        # Wait, dist is global. i is local index.
                        # If i=0, global is 0. If i>0, global is cluster_customers[i-1].
                        global_i = 0 if i == 0 else cluster_customers[i - 1]
                        d_ij = dist[global_i, node_j]
                        
                        new_t = t + d_ij / speed
                        new_b = b - d_ij
                        if new_b < -_EPS:
                            continue
                        
                        # Check time window for customer j
                        # Arrival time is new_t. Service starts at max(new_t, ready[node_j]).
                        # Must be <= due[node_j].
                        start_service = max(new_t, ready[node_j])
                        if start_service > due[node_j] + _EPS:
                            continue
                        
                        end_service = start_service + service[node_j]
                        
                        new_mask = mask | (1 << (j - 1))
                        
                        # Check if this label is dominated by existing labels at (new_mask, j)
                        # Label: (end_service, new_b, d + d_ij)
                        # Dominance: label A dominates B if A.time <= B.time, A.battery >= B.battery, A.dist <= B.dist
                        new_label = (end_service, new_b, d + d_ij)
                        
                        dominated = False
                        for existing in dp[new_mask][j]:
                            if existing[0] <= new_label[0] + _EPS and existing[1] >= new_label[1] - _EPS and existing[2] <= new_label[2] + _EPS:
                                dominated = True
                                break
                        
                        if not dominated:
                            # Add new label and prune any labels dominated by it
                            dp[new_mask][j].append(new_label)
                            # Prune
                            new_dp = []
                            for existing in dp[new_mask][j]:
                                if existing[0] >= new_label[0] - _EPS and existing[1] <= new_label[1] + _EPS and existing[2] >= new_label[2] - _EPS:
                                    continue
                                new_dp.append(existing)
                            dp[new_mask][j] = new_dp
        
        # Find best label that returns to depot
        best_dist = float('inf')
        best_path = None
        full_mask = (1 << n_c) - 1
        
        # We need to track the path. The DP above doesn't store paths.
        # For small n_c, we can store the parent pointer or reconstruct.
        # Let's modify the label to include the path or parent info.
        # Since n_c <= 8, storing the path as a tuple is feasible.
        
        # Redo DP with paths
        dp = [[[] for _ in range(n_c + 1)] for _ in range(1 << n_c)]
        dp[0][0] = [(0.0, battery_cap, 0.0, (0,))] # (time, battery, dist, path_tuple)
        
        for mask in range(1 << n_c):
            for i in range(n_c + 1):
                if not dp[mask][i]:
                    continue
                
                for (t, b, d, path) in dp[mask][i]:
                    for j in range(1, n_c + 1):
                        if mask & (1 << (j - 1)):
                            continue
                        
                        node_j = cluster_customers[j - 1]
                        global_i = 0 if i == 0 else cluster_customers[i - 1]
                        d_ij = dist[global_i, node_j]
                        
                        new_t = t + d_ij / speed
                        new_b = b - d_ij
                        if new_b < -_EPS:
                            continue
                        
                        start_service = max(new_t, ready[node_j])
                        if start_service > due[node_j] + _EPS:
                            continue
                        
                        end_service = start_service + service[node_j]
                        
                        new_mask = mask | (1 << (j - 1))
                        new_dist = d + d_ij
                        new_path = path + (node_j,)
                        
                        new_label = (end_service, new_b, new_dist, new_path)
                        
                        dominated = False
                        for existing in dp[new_mask][j]:
                            if existing[0] <= new_label[0] + _EPS and existing[1] >= new_label[1] - _EPS and existing[2] <= new_label[2] + _EPS:
                                dominated = True
                                break
                        
                        if not dominated:
                            dp[new_mask][j].append(new_label)
                            new_dp = []
                            for existing in dp[new_mask][j]:
                                if existing[0] >= new_label[0] - _EPS and existing[1] <= new_label[1] + _EPS and existing[2] >= new_label[2] - _EPS:
                                    continue
                                new_dp.append(existing)
                            dp[new_mask][j] = new_dp
        
        # Check return to depot from any node in full_mask
        for i in range(1, n_c + 1):
            for (t, b, d, path) in dp[full_mask][i]:
                node_i = cluster_customers[i - 1]
                d_ret = dist[node_i, 0]
                new_t = t + d_ret / speed
                new_b = b - d_ret
                if new_b < -_EPS:
                    continue
                if new_t > horizon + _EPS:
                    continue
                
                total_d = d + d_ret
                if total_d < best_dist:
                    best_dist = total_d
                    best_path = list(path)
        
        if best_path is None:
            # No valid route found without stations. 
            # Try with stations? For now, fall back to heuristic.
            return heuristic_route(cluster_customers)
        
        return best_path

    def heuristic_route(cluster_customers):
        """Simple nearest neighbor heuristic for a cluster."""
        route = []
        unvisited = list(cluster_customers)
        curr = 0
        while unvisited:
            d_to_unvisited = dist[curr, unvisited]
            min_idx = np.argmin(d_to_unvisited)
            next_node = unvisited[min_idx]
            route.append(next_node)
            unvisited.pop(min_idx)
            curr = next_node
        return route

    # Apply optimization to each cluster
    routes = []
    for cluster in clusters:
        if not cluster:
            continue
        optimized = optimize_cluster(cluster)
        if optimized:
            # Verify validity
            v, d, _ = check_route(optimized)
            if v:
                routes.append(optimized)
            else:
                # Fallback to heuristic
                heur = heuristic_route(cluster)
                v, d, _ = check_route(heur)
                if v:
                    routes.append(heur)
                else:
                    # Split into single customers as last resort
                    for c in cluster:
                        routes.append([c])
        else:
            # Fallback
            for c in cluster:
                routes.append([c])
    
    # Evaluate and refine with local search if time permits
    valid, total_dist = evaluate_routes(routes)
    if not valid:
        # Force valid fallback
        routes = [[c] for c in customers]
    
    # Local Search: 2-Opt and Or-Opt within routes
    # We can also try moving customers between routes if it improves distance
    # But for now, just polish each route.
    
    if time.time() < DEADLINE - 10:
        improved = True
        while improved and time.time() < DEADLINE - 10:
            improved = False
            for i in range(len(routes)):
                r = routes[i]
                if len(r) < 2:
                    continue
                # 2-Opt
                for j in range(len(r) - 1):
                    for k in range(j + 1, len(r)):
                        a = r[j]
                        b = r[j+1]
                        c = r[k]
                        d = r[k+1] if k+1 < len(r) else 0
                        
                        old_cost = dist[a, b] + dist[c, d]
                        new_cost = dist[a, c] + dist[b, d]
                        
                        if new_cost < old_cost - _EPS:
                            new_r = r[:j+1] + r[j+1:k+1][::-1] + r[k+1:]
                            v, d_new, _ = check_route(new_r)
                            if v:
                                routes[i] = new_r
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    
    return routes

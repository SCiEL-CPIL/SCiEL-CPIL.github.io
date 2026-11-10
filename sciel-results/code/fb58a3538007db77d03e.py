"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Model-and-optimize heuristic: solve a simplified CVRP via greedy route
    construction with local improvement, then derive edge attractiveness from
    the resulting route structure. Edges used in good routes get high
    attractiveness; unused edges get low values.
    
    Structural design:
    - Build routes using nearest-neighbor + 2-opt improvement under capacity
    - Compute edge frequency/quality from the improved routes
    - Heuristic = base inverse distance + route-edge bonus
    This replaces hand-tuned parametric formulas with a data-driven signal
    from an actual routing solution.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s kill
    
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    customers = np.arange(1, n)
    
    # --- Phase 1: Build routes via greedy + 2-opt ---
    def build_routes():
        """Construct feasible routes using nearest-neighbor with capacity, then 2-opt."""
        unvisited = set(customers.tolist())
        routes = []
        
        while unvisited:
            # Start new route from depot
            route = [0]
            load = 0.0
            current = 0
            
            while True:
                # Find best unvisited customer that fits
                best_j = -1
                best_score = -np.inf
                for j in unvisited:
                    if load + demands[j] > capacity + 1e-9:
                        continue
                    # Score: prefer close, and prefer completing capacity
                    remaining = capacity - load
                    fill_ratio = demands[j] / remaining if remaining > 0 else 0
                    dist = distance_matrix[current, j]
                    score = -dist + 0.3 * fill_ratio
                    if score > best_score:
                        best_score = score
                        best_j = j
                
                if best_j == -1:
                    break
                
                route.append(best_j)
                load += demands[best_j]
                current = best_j
                unvisited.discard(best_j)
            
            # Return to depot
            route.append(0)
            
            # 2-opt improvement on the route (excluding depot endpoints)
            if len(route) > 3:
                route = _two_opt(route, distance_matrix, demands, capacity)
            
            routes.append(route)
        
        return routes
    
    def _two_opt(route, dm, dem, cap):
        """Apply 2-opt to interior of route (between depot nodes)."""
        improved = True
        while improved:
            improved = False
            # Route is [0, c1, c2, ..., ck, 0]
            interior = route[1:-1]
            m = len(interior)
            for i in range(m - 1):
                for j in range(i + 1, m):
                    # Check capacity feasibility of reversed segment
                    # Segment interior[i..j] gets reversed
                    new_interior = interior[:i] + interior[i:j+1][::-1] + interior[j+1:]
                    # Verify capacity along the route
                    feasible = True
                    load = 0.0
                    for node in new_interior:
                        load += dem[node]
                        if load > cap + 1e-9:
                            feasible = False
                            break
                    if not feasible:
                        continue
                    
                    new_route = [0] + new_interior + [0]
                    old_cost = _route_cost(route, dm)
                    new_cost = _route_cost(new_route, dm)
                    if new_cost < old_cost - 1e-9:
                        route = new_route
                        interior = new_interior
                        improved = True
                        break
                if improved:
                    break
        return route
    
    def _route_cost(route, dm):
        return sum(dm[route[i], route[i+1]] for i in range(len(route)-1))
    
    # Build initial routes
    routes = build_routes()
    
    # --- Phase 2: Derive edge attractiveness from routes ---
    # Count how many times each edge is used and with what quality
    edge_count = np.zeros((n, n), dtype=np.float64)
    edge_quality = np.zeros((n, n), dtype=np.float64)
    
    for route in routes:
        route_len = len(route) - 1
        for i in range(route_len):
            u, v = route[i], route[i+1]
            edge_count[u, v] += 1.0
            # Quality: shorter edges are better
            edge_quality[u, v] += 1.0 / (distance_matrix[u, v] + 1e-9)
    
    # Base heuristic: inverse distance
    inv_dist = 1.0 / (distance_matrix + 1e-9)
    
    # Normalize edge usage signal
    max_count = edge_count.max() if edge_count.max() > 0 else 1.0
    edge_usage_signal = edge_count / max_count
    
    # Normalize edge quality signal  
    max_quality = edge_quality.max() if edge_quality.max() > 0 else 1.0
    edge_qual_signal = edge_quality / max_quality
    
    # Combine: base distance + route structure bonus
    heuristic = inv_dist.copy()
    
    # Add bonus for edges that appear in good routes
    # Scale so that route-edge bonus is comparable to base
    bonus_scale = 0.8
    heuristic += bonus_scale * edge_usage_signal * edge_qual_signal
    
    # Ensure symmetry-ish: also boost reverse edges slightly
    # (routes are directed but ACO uses directed heuristic)
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Time guard
    if time.time() > deadline:
        # Return a safe fallback
        return np.maximum(inv_dist, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

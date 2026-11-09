# EVOLVE-BLOCK-START
import numpy as np
import time

def _savings_2opt(distance_matrix, demands, capacity, n, max_time=120):
    """Clarke-Wright Savings + 2-Opt + Or-opt refinement for CVRP."""
    start_time = time.time()
    customers = list(range(1, n))
    
    # Initialize: each customer in its own route
    routes = [[0, c, 0] for c in customers if demands[c] <= capacity]
    route_of = {c: i for i, r in enumerate(routes) for c in r[1:-1]}
    
    # Compute savings
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                savings[i, j] = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
    
    # Sort by savings descending
    sorted_pairs = []
    for i in range(1, n):
        for j in range(i+1, n):
            sorted_pairs.append((savings[i, j], i, j))
    sorted_pairs.sort(reverse=True)
    
    def route_load(route):
        return sum(demands[c] for c in route[1:-1])
    
    def is_endpoint(route, node):
        return len(route) >= 3 and (route[1] == node or route[-2] == node)
    
    def merge_routes(ri, rj, i, j):
        route_i = routes[ri]
        route_j = routes[rj]
        
        # Determine orientation
        i_at_start = (route_i[1] == i)
        i_at_end = (route_i[-2] == i)
        j_at_start = (route_j[1] == j)
        j_at_end = (route_j[-2] == j)
        
        if not (i_at_start or i_at_end) or not (j_at_start or j_at_end):
            return False
        
        inner_i = route_i[1:-1]
        inner_j = route_j[1:-1]
        
        if i_at_start and j_at_end:
            new_inner = inner_i + inner_j
        elif i_at_end and j_at_start:
            new_inner = inner_j + inner_i
        elif i_at_start and j_at_start:
            new_inner = inner_i + inner_j[::-1]
        else:  # i_at_end and j_at_end
            new_inner = inner_j[::-1] + inner_i
        
        merged = [0] + new_inner + [0]
        routes[ri] = merged
        routes[rj] = None
        for node in new_inner:
            route_of[node] = ri
        return True
    
    for s_val, i, j in sorted_pairs:
        if time.time() - start_time > max_time:
            break
        if s_val <= 0:
            continue
        
        ri = route_of.get(i)
        rj = route_of.get(j)
        
        if ri is None or rj is None or ri == rj:
            continue
        
        # Check if i and j are endpoints
        if not is_endpoint(routes[ri], i) or not is_endpoint(routes[rj], j):
            continue
        
        # Check capacity
        if route_load(routes[ri]) + route_load(routes[rj]) > capacity:
            continue
        
        merge_routes(ri, rj, i, j)
    
    # Remove None routes
    routes = [r for r in routes if r is not None]
    route_of = {}
    for i, r in enumerate(routes):
        for c in r[1:-1]:
            route_of[c] = i
    
    # 2-Opt improvement
    improved = True
    passes = 0
    while improved and passes < 30:
        if time.time() - start_time > max_time:
            break
        improved = False
        passes += 1
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            n_route = len(route)
            if n_route < 4:
                continue
            for i in range(1, n_route - 2):
                for j in range(i + 1, n_route - 1):
                    u = route[i-1]
                    v = route[i]
                    w = route[j]
                    x = route[j+1]
                    old_dist = distance_matrix[u, v] + distance_matrix[w, x]
                    new_dist = distance_matrix[u, w] + distance_matrix[v, x]
                    if new_dist < old_dist - 1e-10:
                        routes[r_idx][i:j+1] = routes[r_idx][i:j+1][::-1]
                        improved = True
    
    # Or-opt improvement
    improved = True
    passes = 0
    while improved and passes < 15:
        if time.time() - start_time > max_time:
            break
        improved = False
        passes += 1
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            n_route = len(route)
            if n_route < 5:
                continue
            for seg_len in [1, 2, 3]:
                for i in range(1, n_route - seg_len - 1):
                    for j in range(i + seg_len, n_route - 1):
                        seg = route[i:i+seg_len]
                        u = route[i-1]
                        v = route[i+seg_len]
                        w = route[j-1]
                        x = route[j]
                        old_dist = distance_matrix[u, v] + distance_matrix[w, x]
                        new_dist = distance_matrix[u, seg[0]]
                        for k in range(seg_len - 1):
                            new_dist += distance_matrix[seg[k], seg[k+1]]
                        new_dist += distance_matrix[seg[-1], x]
                        if new_dist < old_dist - 1e-10:
                            # Move segment
                            remaining = route[:i] + route[i+seg_len:j] + route[j:]
                            new_route = remaining[:j] + seg + remaining[j:]
                            routes[r_idx] = [0] + new_route + [0]
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return routes

def _lagrangian_relaxation(distance_matrix, demands, capacity, max_dual_iters=80, max_time=80):
    """Lagrangian relaxation to find edge attractiveness."""
    start_time = time.time()
    n = distance_matrix.shape[0]
    edge_attractiveness = np.zeros((n, n), dtype=float)
    
    lambda_val = 0.0
    best_relaxed_cost = float('inf')
    
    for dual_iter in range(max_dual_iters):
        if time.time() - start_time > max_time:
            break
        
        # Modified cost matrix
        modified_dm = distance_matrix.copy()
        for j in range(1, n):
            penalty = lambda_val * demands[j]
            modified_dm[:, j] += penalty
            modified_dm[j, :] += penalty
        modified_dm = np.maximum(modified_dm, 1e-9)
        
        # Greedy construction with modified costs
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current_routes = []
        
        while not visited[1:].all():
            route = [0]
            current = 0
            remaining_cap = capacity
            
            while True:
                eligible = (~visited) & (np.arange(n) > 0)
                eligible &= (demands <= remaining_cap)
                eligible_indices = np.where(eligible)[0]
                
                if len(eligible_indices) == 0:
                    break
                
                costs = modified_dm[current, eligible_indices]
                next_node = eligible_indices[np.argmin(costs)]
                
                route.append(next_node)
                visited[next_node] = True
                remaining_cap -= demands[next_node]
                current = next_node
                
                if visited[1:].all():
                    break
            
            route.append(0)
            current_routes.append(route)
        
        # Calculate cost
        total_cost = 0.0
        for route in current_routes:
            for k in range(len(route) - 1):
                total_cost += distance_matrix[route[k], route[k+1]]
        
        # Subgradient
        subgradient = 0.0
        for route in current_routes:
            route_load = sum(demands[c] for c in route[1:-1])
            subgradient += (route_load - capacity)
        
        # Polyak step
        gap = total_cost - best_relaxed_cost
        if gap > 1e-6:
            step_size = 2.0 * gap / (subgradient**2 + 1e-9)
        else:
            step_size = 0.1
        
        lambda_val = max(0.0, lambda_val + step_size * subgradient)
        
        if total_cost < best_relaxed_cost:
            best_relaxed_cost = total_cost
        
        # Accumulate edge attractiveness
        solution_quality = max(0.0, 1.0 - (total_cost - best_relaxed_cost) / (best_relaxed_cost + 1e-9))
        
        for route in current_routes:
            for k in range(len(route) - 1):
                u, v = route[k], route[k+1]
                edge_attractiveness[u, v] += solution_quality
                edge_attractiveness[v, u] += solution_quality * 0.5
    
    max_attr = np.max(edge_attractiveness)
    if max_attr > 0:
        edge_attractiveness = edge_attractiveness / max_attr
    
    return edge_attractiveness

def _multi_start_savings(distance_matrix, demands, capacity, n, max_time=100):
    """Run savings heuristic multiple times with different orderings."""
    start_time = time.time()
    best_routes = None
    best_cost = float('inf')
    
    for trial in range(5):
        if time.time() - start_time > max_time:
            break
        
        customers = list(range(1, n))
        
        if trial > 0:
            # Shuffle for diversity
            rng = np.random.RandomState(trial * 100 + 42)
            rng.shuffle(customers)
        
        routes = [[0, c, 0] for c in customers if demands[c] <= capacity]
        route_of = {c: i for i, r in enumerate(routes) for c in r[1:-1]}
        
        savings = np.zeros((n, n))
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    savings[i, j] = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
        
        sorted_pairs = []
        for i in range(1, n):
            for j in range(i+1, n):
                sorted_pairs.append((savings[i, j], i, j))
        sorted_pairs.sort(reverse=True)
        
        def route_load(route):
            return sum(demands[c] for c in route[1:-1])
        
        def is_endpoint(route, node):
            return len(route) >= 3 and (route[1] == node or route[-2] == node)
        
        def merge_routes(ri, rj, i, j):
            route_i = routes[ri]
            route_j = routes[rj]
            
            i_at_start = (route_i[1] == i)
            i_at_end = (route_i[-2] == i)
            j_at_start = (route_j[1] == j)
            j_at_end = (route_j[-2] == j)
            
            if not (i_at_start or i_at_end) or not (j_at_start or j_at_end):
                return False
            
            inner_i = route_i[1:-1]
            inner_j = route_j[1:-1]
            
            if i_at_start and j_at_end:
                new_inner = inner_i + inner_j
            elif i_at_end and j_at_start:
                new_inner = inner_j + inner_i
            elif i_at_start and j_at_start:
                new_inner = inner_i + inner_j[::-1]
            else:
                new_inner = inner_j[::-1] + inner_i
            
            merged = [0] + new_inner + [0]
            routes[ri] = merged
            routes[rj] = None
            for node in new_inner:
                route_of[node] = ri
            return True
        
        for s_val, i, j in sorted_pairs:
            if s_val <= 0:
                continue
            ri = route_of.get(i)
            rj = route_of.get(j)
            if ri is None or rj is None or ri == rj:
                continue
            if not is_endpoint(routes[ri], i) or not is_endpoint(routes[rj], j):
                continue
            if route_load(routes[ri]) + route_load(routes[rj]) > capacity:
                continue
            merge_routes(ri, rj, i, j)
        
        routes = [r for r in routes if r is not None]
        
        # 2-Opt
        improved = True
        passes = 0
        while improved and passes < 20:
            improved = False
            passes += 1
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n_route = len(route)
                if n_route < 4:
                    continue
                for i in range(1, n_route - 2):
                    for j in range(i + 1, n_route - 1):
                        u = route[i-1]
                        v = route[i]
                        w = route[j]
                        x = route[j+1]
                        old_dist = distance_matrix[u, v] + distance_matrix[w, x]
                        new_dist = distance_matrix[u, w] + distance_matrix[v, x]
                        if new_dist < old_dist - 1e-10:
                            routes[r_idx][i:j+1] = routes[r_idx][i:j+1][::-1]
                            improved = True
        
        # Calculate cost
        total_cost = 0.0
        for route in routes:
            for k in range(len(route) - 1):
                total_cost += distance_matrix[route[k], route[k+1]]
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_routes = [r[:] for r in routes]
    
    return best_routes

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 280
    
    n = distance_matrix.shape[0]
    
    # 1. Lagrangian Relaxation-based edge weights
    try:
        lagrangian_weights = _lagrangian_relaxation(
            distance_matrix, demands, capacity,
            max_dual_iters=80, max_time=70
        )
    except Exception:
        lagrangian_weights = np.zeros((n, n), dtype=float)
    
    # 2. Savings + 2-Opt + Or-opt
    try:
        routes_sw = _savings_2opt(distance_matrix, demands, capacity, n, max_time=100)
        edge_matrix_sw = np.zeros((n, n), dtype=float)
        for route in routes_sw:
            for k in range(len(route) - 1):
                u = route[k]
                v = route[k+1]
                edge_matrix_sw[u, v] = 1.0
    except Exception:
        edge_matrix_sw = np.zeros((n, n), dtype=float)
    
    # 3. Multi-start savings
    try:
        routes_ms = _multi_start_savings(distance_matrix, demands, capacity, n, max_time=80)
        edge_matrix_ms = np.zeros((n, n), dtype=float)
        if routes_ms:
            for route in routes_ms:
                for k in range(len(route) - 1):
                    u = route[k]
                    v = route[k+1]
                    edge_matrix_ms[u, v] = 1.0
    except Exception:
        edge_matrix_ms = np.zeros((n, n), dtype=float)
    
    # 4. Base heuristic: 1/d
    dm_safe = distance_matrix + 1e-9
    heuristic_base = 1.0 / dm_safe
    max_base = np.max(heuristic_base)
    if max_base > 0:
        heuristic_base = heuristic_base / max_base
    
    # 5. Combine with tuned weights
    alpha_lagrangian = 8.0
    alpha_savings_sw = 4.0
    alpha_savings_ms = 3.0
    
    heuristic = heuristic_base + alpha_lagrangian * lagrangian_weights + alpha_savings_sw * edge_matrix_sw + alpha_savings_ms * edge_matrix_ms
    
    # Ensure positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Final check
    if not np.all(np.isfinite(heuristic)):
        heuristic = 1.0 / (distance_matrix + 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END

import numpy as np
import time

def _nearest_neighbor_tour(dist, n):
    """Build a nearest neighbor tour starting from node 0."""
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    tour = [0]
    current = 0
    for _ in range(n - 1):
        unvisited = np.where(~visited)[0]
        if len(unvisited) == 0:
            break
        # Find nearest unvisited
        dists = dist[current, unvisited]
        idx = np.argmin(dists)
        next_node = unvisited[idx]
        visited[next_node] = True
        tour.append(next_node)
        current = next_node
    tour.append(0)
    return np.array(tour)

def _two_opt_improve(dist, tour, n, max_iters=100):
    """Improve tour using 2-opt. tour is a closed tour (first==last)."""
    path = list(tour[:-1])
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n - 2):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = path[i], path[i + 1]
                c, d = path[j], path[(j + 1) % n]
                old_cost = dist[a, b] + dist[c, d]
                new_cost = dist[a, c] + dist[b, d]
                if new_cost < old_cost - 1e-10:
                    path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                    improved = True
    return np.array(path + [path[0]])

def _cheapest_insertion(dist, n, start_node=0):
    """
    Construct a tour using cheapest insertion heuristic.
    Start with a random edge, then insert remaining nodes at the position with minimum cost increase.
    """
    # Start with a small path
    # Pick a random second node close to start_node
    dists = dist[start_node, :].copy()
    dists[start_node] = np.inf
    second_node = np.argmin(dists)
    
    path = [start_node, second_node]
    visited = set([start_node, second_node])
    remaining = list(set(range(n)) - visited)
    
    while remaining:
        # Pick the node with the highest minimum distance to any visited node? 
        # Or just the first one? Standard cheapest insertion usually picks the node 
        # that is "furthest" from the current tour to force structure, 
        # or just iterates. Let's pick the one with max min-dist to visited to ensure spread.
        
        # Calculate min dist from each remaining node to any node in path
        # path is small, remaining is small.
        # Actually, standard cheapest insertion often just takes the first node in a list 
        # or the one with highest cost to insert. 
        # Let's use the "furthest" heuristic for diversity in construction.
        
        best_node_to_insert = -1
        max_min_dist = -1
        
        for node in remaining:
            min_d = np.min(dist[node, path])
            if min_d > max_min_dist:
                max_min_dist = min_d
                best_node_to_insert = node
        
        if best_node_to_insert == -1:
            best_node_to_insert = remaining[0]
            
        remaining.remove(best_node_to_insert)
        
        # Find best position to insert node in current path
        min_cost = float('inf')
        best_pos = -1
        
        m = len(path)
        for i in range(m):
            u = path[i]
            v = path[(i + 1) % m]
            # Cost increase if we insert node between u and v
            # New edges: u->node, node->v
            # Removed edge: u->v
            cost_inc = dist[u, best_node_to_insert] + dist[best_node_to_insert, v] - dist[u, v]
            if cost_inc < min_cost:
                min_cost = cost_inc
                best_pos = i
        
        # Insert node after path[best_pos]
        if best_pos == m - 1:
            path.append(best_node_to_insert)
        else:
            path.insert(best_pos + 1, best_node_to_insert)
            
    path.append(path[0])
    return np.array(path)

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    Strategy: "Skeleton Consensus"
    1. Generate multiple diverse tours (NN, Cheapest Insertion with different starts).
    2. Optimize each with 2-Opt.
    3. Identify edges that appear in the majority of these optimized tours (the "skeleton").
    4. Create a heuristic that strongly boosts skeleton edges and suppresses others.
    This guides the ACO ants to reconstruct the near-optimal structure found by local search,
    rather than just sampling from a distribution of mediocre tours.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    n = distance_matrix.shape[0]
    
    # Baseline: Inverse distance, but scaled down to be a weak prior
    # We want the "skeleton" to dominate.
    heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Edge frequency counter
    edge_counts = np.zeros((n, n), dtype=np.int32)
    
    num_tours = 0
    tours = []
    
    # 1. Standard NN from node 0
    if time.time() < deadline - 20:
        t = _nearest_neighbor_tour(distance_matrix, n)
        t = _two_opt_improve(distance_matrix, t, n, max_iters=100)
        tours.append(t)
        num_tours += 1
        
    # 2. Cheapest Insertion from various start nodes
    # We can't easily change the start node of NN in the simple loop without rewriting,
    # but Cheapest Insertion takes a start_node.
    start_nodes_to_try = [0, 1, 2, n//2, n-1]
    
    for start_node in start_nodes_to_try:
        if time.time() > deadline - 20:
            break
        if start_node == 0:
             # We already did NN from 0, but let's do Cheapest Insertion from 0 too for variety?
             # Or skip if we want diversity in construction method.
             # Let's add CI from 0.
             pass
             
        t = _cheapest_insertion(distance_matrix, n, start_node=start_node)
        t = _two_opt_improve(distance_matrix, t, n, max_iters=100)
        tours.append(t)
        num_tours += 1
        
    # 3. Randomized NN variants if time permits
    # We can simulate randomized NN by permuting the "nearest" selection slightly,
    # but for simplicity and speed, we rely on the diversity of CI starts and NN.
    # If we have time, add more CI starts.
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        # Random start node
        import random
        random.seed(int(time.time() * 1000) % 100000) # Pseudo-random seed based on time for variety
        sn = random.randint(0, n-1)
        t = _cheapest_insertion(distance_matrix, n, start_node=sn)
        t = _two_opt_improve(distance_matrix, t, n, max_iters=100)
        tours.append(t)
        num_tours += 1
        
    if num_tours == 0:
        # Fallback: just return inverse distance
        np.fill_diagonal(heuristic, 1e-9)
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        return heuristic

    # Count edges in all optimized tours
    for tour in tours:
        for k in range(n):
            u = tour[k]
            v = tour[k + 1]
            edge_counts[u, v] += 1
            edge_counts[v, u] += 1
            
    # Define the "Skeleton": Edges that appear in at least 60% of the tours
    threshold = 0.6 * num_tours
    skeleton_mask = edge_counts >= threshold
    
    # Construct the final heuristic
    # We want skeleton edges to be VERY attractive.
    # Non-skeleton edges to be weak.
    
    # Base heuristic is inverse distance.
    # For skeleton edges: multiply by a large factor (e.g., 100)
    # For non-skeleton edges: keep base, or multiply by a small factor (e.g., 0.1) to suppress noise?
    # Actually, if we suppress non-skeleton edges too much, we might break the tour if the skeleton
    # isn't a single connected cycle (which it might not be, it's a set of edges).
    # But 2-opt tours are cycles. The intersection of cycles might be a set of disjoint paths/cycles.
    # If the ACO ants are forced to use only skeleton edges, they might get stuck if the skeleton
    # doesn't form a single Hamiltonian cycle.
    
    # Safer approach: 
    # Heuristic = Base * (1 + Boost * SkeletonMask)
    # If SkeletonMask is 1, Heuristic is Base * (1+Boost).
    # If SkeletonMask is 0, Heuristic is Base.
    
    # Let's use a strong boost for skeleton edges.
    boost = 50.0
    
    # Also, we can penalize edges that are in NO tours? 
    # Edge counts 0 means never used in any optimized tour.
    # Let's suppress those slightly.
    
    # Create a multiplier matrix
    multiplier = np.ones((n, n))
    multiplier[skeleton_mask] = 1.0 + boost
    
    # Optional: Penalize edges that are very long and not in skeleton
    # But Base (inverse dist) already handles length.
    
    heuristic = heuristic * multiplier
    
    # Ensure diagonal is 0 (or very small)
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic

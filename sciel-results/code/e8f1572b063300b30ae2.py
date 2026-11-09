import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix for CVRP using a TSP-decomposition heuristic.
    1. Solves a relaxed TSP (ignoring capacity) using 2-Opt.
    2. Decomposes the TSP tour into segments that fit the capacity.
    3. Boosts the attractiveness of edges within these segments and their depot connections.
    """
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones((1, 1))
        
    # Base heuristic: Inverse distance
    # Avoid division by zero
    d = distance_matrix + 1e-9
    eta = 1.0 / d
    
    # 1. Solve a fast TSP to get a global structure
    # Start with Nearest Neighbor Tour
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    
    while unvisited:
        # Find nearest unvisited node
        next_node = min(unvisited, key=lambda x: distance_matrix[current, x])
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
        
    # 2. Improve with 2-Opt
    # 2-Opt on a list of nodes
    def two_opt_improve(tour, dist_matrix):
        n = len(tour)
        improved = True
        # Limit iterations to ensure speed, though n=51 is small
        max_iters = 100
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    if j == i + 1:
                        continue
                    # Calculate delta for 2-opt move
                    # Removing (tour[i-1], tour[i]) and (tour[j], tour[j+1])
                    # Adding (tour[i-1], tour[j]) and (tour[i], tour[j+1])
                    # Note: tour is a path, not a cycle in this representation? 
                    # Standard 2-opt is for cycles. 
                    # Let's treat it as a cycle for improvement, then fix depot at 0.
                    # Actually, simpler: just do 2-opt on the path.
                    # But TSP is a cycle.
                    # Let's stick to a simple 2-opt on the list, treating it as a path.
                    # This is less effective but faster and sufficient for heuristic generation.
                    
                    # Path 2-opt:
                    # Segment tour[i..j] is reversed.
                    # Cost change:
                    # Old: d(tour[i-1], tour[i]) + d(tour[j], tour[j+1]) (if j+1 exists)
                    # New: d(tour[i-1], tour[j]) + d(tour[i], tour[j+1]) (if j+1 exists)
                    
                    # Indices in tour list
                    # i is start of segment, j is end of segment.
                    # Predecessor of i is i-1.
                    # Successor of j is j+1.
                    
                    if i == 0:
                        # Reversing from start? No, usually we keep 0 fixed or handle carefully.
                        # For simplicity, let's assume 0 is at index 0.
                        # We only reverse segments not involving 0?
                        # Or just allow it, and the final tour will be sorted/rotated?
                        # Actually, the structure of the tour matters for decomposition.
                        # Let's just do a standard 2-opt on the cycle.
                        pass
                        
        return tour

    # Simpler 2-Opt for Path (Depot at 0 is fixed at start)
    # We will perform 2-opt moves that do not involve the depot (index 0) to keep it simple.
    # Or just do a few passes of Or-Opt?
    # Given n=51, a full 2-opt on the cycle is fine.
    
    def two_opt_cycle(tour, dist_matrix):
        n = len(tour)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if (j - i + 1 == n) or (j == i + 1):
                        continue
                    # Cycle 2-opt
                    # Nodes: a=tour[i-1], b=tour[i], c=tour[j], d=tour[(j+1)%n]
                    a = tour[(i - 1) % n]
                    b = tour[i]
                    c = tour[j]
                    d = tour[(j + 1) % n]
                    
                    old_cost = dist_matrix[a, b] + dist_matrix[c, d]
                    new_cost = dist_matrix[a, c] + dist_matrix[b, d]
                    
                    if new_cost < old_cost - 1e-9:
                        # Reverse segment i..j
                        tour[i:j+1] = tour[i:j+1][::-1]
                        improved = True
        return tour

    tour = two_opt_cycle(tour, distance_matrix)
    
    # Ensure depot is at start (for decomposition logic simplicity, though cycle is symmetric)
    # Find index of 0
    idx_0 = tour.index(0)
    tour = tour[idx_0:] + tour[:idx_0]
    
    # 3. Decompose tour into capacity-feasible segments
    # The tour is a cycle: t0(0), t1, ..., t_{n-1}, back to t0.
    # We traverse the cycle.
    # We accumulate demand.
    # When adding a node would exceed capacity, we "cut" the route.
    # The current segment becomes a route.
    # We start a new segment with the current node.
    # Note: The depot (0) has demand 0.
    # If we cut, the new route starts with the node that caused the overflow.
    # The depot is implicitly at the start/end of each route in CVRP.
    
    segments = []
    current_segment = [tour[0]] # Start with depot
    current_load = 0.0
    
    for i in range(1, n):
        node = tour[i]
        if current_load + demands[node] <= capacity:
            current_segment.append(node)
            current_load += demands[node]
        else:
            # Close current segment
            segments.append(current_segment)
            # Start new segment with the current node
            # The depot is not explicitly in the middle, but logically the route ends at depot and starts at depot.
            # For the purpose of identifying "good" edges, the edges within the segment are good.
            # The edge connecting the last node of the previous segment to the depot is good.
            # The edge connecting the depot to the first node of the new segment is good.
            # So we just record the segment.
            current_segment = [node]
            current_load = demands[node]
            
    # Don't forget the last segment
    if len(current_segment) > 0:
        segments.append(current_segment)
        
    # 4. Boost heuristic matrix
    boost = np.ones_like(eta)
    
    for seg in segments:
        # Internal edges
        for k in range(len(seg) - 1):
            u = seg[k]
            v = seg[k+1]
            boost[u, v] = 10.0
            boost[v, u] = 10.0
            
        # Depot connections
        # The route is 0 -> seg[0] -> ... -> seg[-1] -> 0
        # Note: seg[0] might be the depot itself if the segment was just [0] (empty route, unlikely)
        # Or if the first node in the tour was 0, seg[0] is 0.
        # In our loop, seg[0] is tour[0] which is 0.
        # So the first internal edge is (0, tour[1]).
        # The "depot connection" at the start is (0, seg[1]) if seg[1] exists?
        # Actually, the internal edges already cover (0, seg[1]) if seg[0]=0.
        # The missing edge is (seg[-1], 0).
        
        if len(seg) > 1:
            last_node = seg[-1]
            boost[last_node, 0] = 10.0
            boost[0, last_node] = 10.0
        elif len(seg) == 1:
            # Segment is just [0]? Or just [node]?
            # If segment is [node], it means it was the first node of a new route after a cut.
            # But in our logic, if we cut, we start new segment with [node].
            # So seg might be [node].
            # In that case, the route is 0 -> node -> 0.
            # Edges (0, node) and (node, 0) are good.
            node = seg[0]
            boost[0, node] = 10.0
            boost[node, 0] = 10.0

    eta = eta * boost
    
    # Ensure non-finite values are handled
    eta[~np.isfinite(eta)] = 1e-9
    
    # Ensure positive
    eta[eta <= 0] = 1e-9
    
    return eta

import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate candidate pool: Hexagon of radius 4 (61 points)
    R_pool = 4
    pool = []
    for a in range(-R_pool, R_pool + 1):
        for b in range(-R_pool, R_pool + 1):
            if max(abs(a), abs(b), abs(a + b)) <= R_pool:
                pool.append((a, b))
    
    n_pool = len(pool)
    if n_pool != 61:
        return [[a, b] for a in range(16) for b in range(2)]

    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(pool)}
    
    # Precompute adjacency matrix for pool
    adj = [[False] * n_pool for _ in range(n_pool)]
    degrees = [0] * n_pool
    
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da * da + db * db + da * db

    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            if dist_sq(pool[i], pool[j]) == 1:
                adj[i][j] = True
                adj[j][i] = True
                degrees[i] += 1
                degrees[j] += 1

    # Initial solution: Best subset of H3 (radius 3)
    # We can reuse the logic from the previous step or just start with a random subset of pool
    # Better: Start with the known good 75-edge solution from H3
    R_init = 3
    init_points = []
    for a in range(-R_init, R_init + 1):
        for b in range(-R_init, R_init + 1):
            if max(abs(a), abs(b), abs(a + b)) <= R_init:
                init_points.append((a, b))
                
    n_init = len(init_points)
    if n_init != 37:
         return [[a, b] for a in range(16) for b in range(2)]

    # Find best 32 from init_points (H3)
    # This is fast
    init_adj = [[False] * n_init for _ in range(n_init)]
    init_deg = [0] * n_init
    for i in range(n_init):
        for j in range(i + 1, n_init):
            if dist_sq(init_points[i], init_points[j]) == 1:
                init_adj[i][j] = True
                init_adj[j][i] = True
                init_deg[i] += 1
                init_deg[j] += 1
                
    total_init_edges = sum(init_deg) // 2
    best_score_init = -1
    best_removed_init = None
    
    for removed in itertools.combinations(range(n_init), 5):
        sum_deg = sum(init_deg[i] for i in removed)
        edges_within = 0
        r_list = list(removed)
        for i in range(5):
            for j in range(i + 1, 5):
                if init_adj[r_list[i]][r_list[j]]:
                    edges_within += 1
        lost = sum_deg - edges_within
        rem_edges = total_init_edges - lost
        if rem_edges > best_score_init:
            best_score_init = rem_edges
            best_removed_init = removed

    if best_removed_init is None:
        return [[a, b] for a in range(16) for b in range(2)]
        
    removed_set_init = set(best_removed_init)
    current_set = set()
    for i in range(n_init):
        if i not in removed_set_init:
            current_set.add(init_points[i])
            
    # Convert to indices in pool
    current_indices = [point_to_idx[p] for p in current_set]
    current_set_indices = set(current_indices)
    
    # Calculate current edge count
    def count_edges(idx_set):
        cnt = 0
        idx_list = list(idx_set)
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                if adj[idx_list[i]][idx_list[j]]:
                    cnt += 1
        return cnt
        
    current_edges = count_edges(current_set_indices)
    
    # Hill Climbing
    best_edges = current_edges
    best_config = set(current_set_indices)
    
    # Candidate pool indices not in current set
    # We will try swapping
    
    while time.time() < deadline - 5:
        improved = False
        # Get list of points in set and points in pool
        in_set = list(current_set_indices)
        not_in_set = [i for i in range(n_pool) if i not in current_set_indices]
        
        # Try all swaps
        best_swap = None
        best_gain = 0
        
        for p_in in in_set:
            for p_out in not_in_set:
                # Calculate gain
                # Remove p_in, Add p_out
                # Gain = deg(p_out) - deg(p_in) - 2 * (adj[p_in][p_out] ? 1 : 0) ... No
                # New edges involving p_out: count neighbors of p_out in (current_set - {p_in})
                # Lost edges involving p_in: count neighbors of p_in in (current_set - {p_in})
                
                # Optimization:
                # neighbors_in_set_p_in = sum(1 for x in current_set_indices if x != p_in and adj[p_in][x])
                # neighbors_in_set_p_out = sum(1 for x in current_set_indices if x != p_in and adj[p_out][x])
                # Note: if adj[p_in][p_out], then p_out is not in set, so it doesn't contribute to neighbors_in_set_p_out directly in the "lost" sense?
                # Let's be precise.
                # Old edges: E
                # New edges: E - deg_in(p_in) + deg_new(p_out)
                # deg_in(p_in) = number of neighbors of p_in in current_set_indices
                # deg_new(p_out) = number of neighbors of p_out in (current_set_indices - {p_in})
                
                deg_in_p_in = 0
                deg_new_p_out = 0
                for x in current_set_indices:
                    if x == p_in:
                        continue
                    if adj[p_in][x]:
                        deg_in_p_in += 1
                    if adj[p_out][x]:
                        deg_new_p_out += 1
                        
                gain = deg_new_p_out - deg_in_p_in
                
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (p_in, p_out)
                    
        if best_swap:
            p_in, p_out = best_swap
            current_set_indices.remove(p_in)
            current_set_indices.add(p_out)
            current_edges += best_gain
            improved = True
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_config = set(current_set_indices)
                
            if best_edges == 76:
                break
        else:
            # No improvement found, try a random restart or perturb
            # Simple perturbation: random swap
            if not_in_set and in_set:
                p_in = in_set[0] # or random
                p_out = not_in_set[0]
                # Check if this swap is valid (it is, distinct)
                # Just do it to escape local optimum?
                # Better: Random restart with new initial set?
                # For now, just break and try a different strategy if time allows
                break

    # If we reached 76, great.
    # If not, return the best found.
    
    result = []
    for idx in best_config:
        result.append([pool[idx][0], pool[idx][1]])
        
    if len(result) != 32:
        return [[a, b] for a in range(16) for b in range(2)]
        
    return result

# EVOLVE-BLOCK-END

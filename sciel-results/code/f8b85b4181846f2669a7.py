import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Generate Clebsch Graph
    # Vertices 0..15 are 5-bit even parity strings
    vectors = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vectors.append(i)

    adj = [0] * 16
    clebsch_edges = []
    for u_idx in range(16):
        for v_idx in range(u_idx + 1, 16):
            xor = vectors[u_idx] ^ vectors[v_idx]
            if bin(xor).count('1') == 4:
                clebsch_edges.append([u_idx, v_idx])
                adj[u_idx] |= (1 << v_idx)
                adj[v_idx] |= (1 << u_idx)

    best_edges = clebsch_edges
    best_count = len(clebsch_edges)  # 40

    # 2. Fast 3-colorability check using bitmasks and MRV
    def is_3_colorable(adj_mask):
        n = 16
        # Precompute neighbor bitmasks
        neighbors = [0] * n
        for i in range(n):
            neighbors[i] = adj_mask[i]
        
        # Unassigned vertices bitmask
        unassigned = (1 << n) - 1
        # Color classes bitmasks
        color_classes = [0, 0, 0]
        
        # MRV Heuristic: Pick vertex with fewest available colors
        # Available colors for v are those c where (neighbors[v] & color_classes[c]) == 0
        
        def backtrack():
            if unassigned == 0:
                return True
            
            # Select vertex with minimum available colors (MRV)
            # If a vertex has 0 available colors, return False (pruning)
            best_v = -1
            min_avail = 4
            best_avail_mask = 0 # Bitmask of available colors
            
            # Iterate through unassigned vertices
            # Optimization: Only check vertices in unassigned
            v = 0
            while v < n:
                if unassigned & (1 << v):
                    avail_mask = 0
                    if (neighbors[v] & color_classes[0]) == 0: avail_mask |= 1
                    if (neighbors[v] & color_classes[1]) == 0: avail_mask |= 2
                    if (neighbors[v] & color_classes[2]) == 0: avail_mask |= 4
                    avail_count = bin(avail_mask).count('1')
                    if avail_count == 0:
                        return False
                    if avail_count < min_avail:
                        min_avail = avail_count
                        best_v = v
                        best_avail_mask = avail_mask
                        if min_avail == 1:
                            break
                v += 1
            
            # Assign color to best_v
            # Try colors in best_avail_mask
            c = 0
            while c < 3:
                if best_avail_mask & (1 << c):
                    # Assign
                    color_classes[c] |= (1 << best_v)
                    unassigned &= ~(1 << best_v)
                    
                    if backtrack():
                        return True
                    
                    # Unassign
                    color_classes[c] &= ~(1 << best_v)
                    unassigned |= (1 << best_v)
                c += 1
            
            return False
        
        return backtrack()

    # 3. Local Search
    def can_add_edge(adj_mask, u, v):
        return (adj_mask[u] & adj_mask[v]) == 0

    def try_add_edges(adj_mask, current_edges, current_count):
        nonlocal best_edges, best_count
        n = 16
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Find all addable edges
            available = []
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj_mask[u] & (1 << v)):
                        if can_add_edge(adj_mask, u, v):
                            available.append((u, v))

            if not available:
                break

            # Try adding each available edge
            for u, v in available:
                if time.time() > deadline:
                    break
                # Add edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                new_edges = current_edges + [[u, v]]
                new_count = current_count + 1

                # Check if still 4-chromatic (not 3-colorable)
                if not is_3_colorable(adj_mask):
                    # Found better graph
                    if new_count > best_count:
                        best_edges = new_edges
                        best_count = new_count
                    # Continue trying to add more edges
                    new_edges, new_count = try_add_edges(adj_mask, new_edges, new_count)
                    current_edges = new_edges
                    current_count = new_count
                    changed = True
                else:
                    # Revert
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)

        return current_edges, current_count

    # Start local search from Clebsch graph
    try_add_edges(adj[:], best_edges, best_count)

    # 4. Random restarts
    all_pairs = [(u, v) for u in range(16) for v in range(u+1, 16)]
    clebsch_set = set((min(u,v), max(u,v)) for u,v in clebsch_edges)
    candidate_edges = [(u,v) for u,v in all_pairs if (u,v) not in clebsch_set]

    for iteration in range(2000):
        if time.time() > deadline:
            break

        random.shuffle(candidate_edges)
        adj_mask = [0] * 16
        edges = []

        # Greedy construction
        for u, v in candidate_edges:
            if time.time() > deadline:
                break
            if can_add_edge(adj_mask, u, v):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append([u, v])

        # Add Clebsch edges if possible
        for u, v in clebsch_edges:
            if not (adj_mask[u] & (1 << v)):
                if can_add_edge(adj_mask, u, v):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges.append([u, v])

        if len(edges) > best_count:
            if not is_3_colorable(adj_mask):
                best_edges = edges
                best_count = len(edges)
                # Try to add more
                new_edges, new_count = try_add_edges(adj_mask[:], edges, len(edges))

    return best_edges

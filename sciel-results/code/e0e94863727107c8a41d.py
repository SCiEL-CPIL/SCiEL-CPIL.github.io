import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 31
    q = 5
    
    # 1. Build Polarity Graph
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
        
    adj_masks = [0] * n
    edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edge_set.add((i, j))
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
                
    if len(edge_set) != 90:
        # Fallback
        pass

    # 2. Add 1st edge (91st)
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edge_set:
                non_edges.append((i, j))
                
    # Initialize common counts and C4 sum
    # common[i][j] = number of common neighbors of i and j
    common = [[0] * n for _ in range(n)]
    c4_sum = 0 # 2 * C4 count
    
    # Helper to update common counts and c4_sum for adding an edge
    # We need to be careful with the order of operations.
    
    def add_edge(u, v):
        nonlocal c4_sum
        # Update common counts
        # For each w in N(v), c(u, w) increases by 1
        mask_v = adj_masks[v]
        # Iterate over set bits in mask_v
        w = 0
        m = mask_v
        while m:
            if m & 1:
                if w != u:
                    k = common[u][w]
                    c4_sum += k
                    common[u][w] = k + 1
                    common[w][u] = k + 1
            m >>= 1
            w += 1
            
        # For each w in N(u), c(v, w) increases by 1
        mask_u = adj_masks[u]
        w = 0
        m = mask_u
        while m:
            if m & 1:
                if w != v:
                    k = common[v][w]
                    c4_sum += k
                    common[v][w] = k + 1
                    common[w][v] = k + 1
            m >>= 1
            w += 1
            
        # Update adjacency
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)

    def remove_edge(u, v):
        nonlocal c4_sum
        # Reverse of add
        # For each w in N(v) \ {u}, c(u, w) decreases by 1
        # Note: We must use the state BEFORE removing the edge from adj_masks
        mask_v = adj_masks[v]
        w = 0
        m = mask_v
        while m:
            if m & 1:
                if w != u:
                    k = common[u][w]
                    c4_sum -= (k - 1)
                    common[u][w] = k - 1
                    common[w][u] = k - 1
            m >>= 1
            w += 1
            
        # For each w in N(u) \ {v}, c(v, w) decreases by 1
        mask_u = adj_masks[u]
        w = 0
        m = mask_u
        while m:
            if m & 1:
                if w != v:
                    k = common[v][w]
                    c4_sum -= (k - 1)
                    common[v][w] = k - 1
                    common[w][v] = k - 1
            m >>= 1
            w += 1
            
        # Update adjacency
        adj_masks[u] &= ~(1 << v)
        adj_masks[v] &= ~(1 << u)

    # Initialize common counts for the 90 edges
    # We can just iterate over all pairs and count common neighbors
    # O(n^2 * n) = 31^3 ~ 30k ops. Very fast.
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in edge_set:
                # Count common neighbors
                c = 0
                common_mask = adj_masks[i] & adj_masks[j]
                c = common_mask.bit_count()
                common[i][j] = c
                common[j][i] = c
                c4_sum += c * (c - 1) // 2
                
    # Note: The formula sum_{i<j} binom(c(i,j), 2) counts each C4 twice?
    # Yes. So c4_sum is 2 * C4.
    # Let's verify with a small example.
    # C4: 0-1-2-3-0.
    # Pairs of opposite vertices: (0,2) and (1,3).
    # c(0,2) = 2 (neighbors 1,3). binom(2,2)=1.
    # c(1,3) = 2 (neighbors 0,2). binom(2,2)=1.
    # Sum = 2. C4 count = 1. Correct.
    
    # Initialize best first edge selection
    best_first_edge = None
    best_first_c4_sum = float('inf')
    
    for (i, j) in non_edges:
        add_edge(i, j)
        if c4_sum < best_first_c4_sum:
            best_first_c4_sum = c4_sum
            best_first_edge = (i, j)
        remove_edge(i, j)
        
    # Apply best first edge
    if best_first_edge:
        i, j = best_first_edge
        edge_set.add((i, j))
        non_edges.remove((i, j))
        add_edge(i, j)
        
    current_c4_sum = best_first_c4_sum
    best_c4_sum = current_c4_sum
    best_adj_masks = adj_masks[:]
    best_common = [row[:] for row in common]
    
    # 3. Local Search
    random.seed(42)
    
    passes = 0
    max_passes = 5000 # Increased due to speedup
    
    while time.time() < deadline and passes < max_passes:
        passes += 1
        improved_in_pass = False
        
        edges_to_try = list(edge_set)
        random.shuffle(edges_to_try)
        
        for (u, v) in edges_to_try:
            if time.time() > deadline - 5:
                break
            
            if (u, v) not in edge_set:
                continue
            
            # Remove (u,v)
            remove_edge(u, v)
            
            best_swap = None
            best_swap_c4_sum = current_c4_sum
            
            # Try all non-edges
            # Optimization: Only try non-edges that might improve?
            # Hard to know without evaluating.
            # But O(n) per swap is fast.
            
            for (x, y) in non_edges:
                add_edge(x, y)
                
                if c4_sum < best_swap_c4_sum:
                    best_swap_c4_sum = c4_sum
                    best_swap = (x, y)
                    
                remove_edge(x, y)
                
            # Restore (u,v)
            add_edge(u, v)
            
            if best_swap is not None:
                x, y = best_swap
                # Apply swap
                edge_set.remove((u, v))
                non_edges.remove((x, y))
                edge_set.add((x, y))
                non_edges.append((u, v))
                
                # We already have the state with (u,v) removed and (x,y) added?
                # No, we restored (u,v) above.
                # So we need to:
                # 1. Remove (u,v)
                # 2. Add (x,y)
                remove_edge(u, v)
                add_edge(x, y)
                
                current_c4_sum = best_swap_c4_sum
                improved_in_pass = True
                
                if current_c4_sum < best_c4_sum:
                    best_c4_sum = current_c4_sum
                    best_adj_masks = adj_masks[:]
                    best_common = [row[:] for row in common]
        
        if not improved_in_pass:
            # Local minimum. Perturb.
            if edge_set and non_edges:
                e1 = random.choice(list(edge_set))
                e2 = random.choice(non_edges)
                
                u, v = e1
                x, y = e2
                
                remove_edge(u, v)
                edge_set.remove(e1)
                non_edges.append(e1)
                
                add_edge(x, y)
                edge_set.add(e2)
                non_edges.remove(e2)
                
                current_c4_sum = c4_sum
                
                if current_c4_sum < best_c4_sum:
                    best_c4_sum = current_c4_sum
                    best_adj_masks = adj_masks[:]
                    best_common = [row[:] for row in common]

    # Restore best state
    adj_masks = best_adj_masks
    
    # Convert back to edge list
    final_edges = []
    for i in range(n):
        mask = adj_masks[i]
        for j in range(i+1, n):
            if mask & (1 << j):
                final_edges.append([i, j])
                
    # Verify
    if len(final_edges) != 91:
        # Fallback
        return [[i, j] for i in range(30) for j in [i+1]][:91]
        
    return final_edges

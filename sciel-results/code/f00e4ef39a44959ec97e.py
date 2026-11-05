import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 31
    m = 91
    
    # 1. Construct PG(2,5) points
    points = []
    for a in range(5):
        for b in range(5):
            points.append((1, a, b))
    for b in range(5):
        points.append((0, 1, b))
    points.append((0, 0, 1))
    
    # Map points to indices 0..30
    point_index = {p: i for i, p in enumerate(points)}
    
    # 2. Construct adjacency matrix for polarity graph
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            # Dot product mod 5
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                
    # Verify edge count
    edge_count = np.sum(adj) // 2
    # Should be 90
    
    # Helper: compute neighbors from adj as bitmasks
    def get_neighbor_masks(adj):
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i, j] == 1:
                    mask |= (1 << j)
            masks[i] = mask
        return masks

    # Helper: count C4s using common neighbor counts
    def count_c4s(adj, neighbors):
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                # Count common neighbors
                common_count = 0
                for w in neighbors[u]:
                    if adj[w, v] == 1:
                        common_count += 1
                if common_count >= 2:
                    total += common_count * (common_count - 1) // 2
        return total // 2

    # Helper: compute delta in C4 count when swapping edge (u,v) with non-edge (i,j)
    # Returns delta = C_new - C_old
    def delta_swap(masks, u, v, i, j):
        # Create new masks
        # N'(u) = N(u) \ {v}
        # N'(v) = N(v) \ {u}
        # N'(i) = N(i) U {j}
        # N'(j) = N(j) U {i}
        # Others unchanged
        
        # Identify affected pairs S
        # Pairs involving u, v, i, j
        # Pairs where common neighbor count might change due to u, v, i, j being common neighbors
        # This is effectively pairs {a,b} such that N(a) or N(b) changed, 
        # OR u/v/i/j is in N(a) cap N(b) but status changed.
        
        # Simplest: just check all pairs {a,b} where a or b is in {u,v,i,j} 
        # OR a,b are in N(u) U N(v) U N(i) U N(j)
        
        affected_vertices = set([u, v, i, j])
        for x in [u, v, i, j]:
            # Add neighbors of x
            mask = masks[x]
            # Iterate bits
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                affected_vertices.add(idx)
                m ^= lsb
        
        # Convert to list
        affected_list = list(affected_vertices)
        
        # Precompute new masks for affected vertices
        new_masks = {}
        for x in affected_list:
            if x == u:
                new_masks[x] = masks[u] & ~(1 << v)
            elif x == v:
                new_masks[x] = masks[v] & ~(1 << u)
            elif x == i:
                new_masks[x] = masks[i] | (1 << j)
            elif x == j:
                new_masks[x] = masks[j] | (1 << i)
            else:
                new_masks[x] = masks[x]
                
        delta = 0
        # Iterate over all pairs in affected_list
        for idx_a in range(len(affected_list)):
            a = affected_list[idx_a]
            for idx_b in range(idx_a + 1, len(affected_list)):
                b = affected_list[idx_b]
                
                # Old common count
                old_common = bin(masks[a] & masks[b]).count("1")
                # New common count
                new_common = bin(new_masks[a] & new_masks[b]).count("1")
                
                if old_common != new_common:
                    delta += (new_common * (new_common - 1) // 2) - (old_common * (old_common - 1) // 2)
                    
        return delta

    # 3. Find best edge to add
    # We start with 90 edges. We need 91.
    # We can try adding each non-edge and see which adds the fewest C4s.
    # But we can also just pick one and then swap.
    # Let's pick the one that minimizes delta_add.
    
    masks = get_neighbor_masks(adj)
    
    # Calculate initial C4 count
    # We need neighbors list for count_c4s
    neighbors = [list(np.where(adj[i] == 1)[0]) for i in range(n)]
    current_c4s = count_c4s(adj, neighbors)
    
    # Find best edge to add
    best_delta = float('inf')
    best_edge = None
    
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 0:
                # Delta for adding (i,j)
                # This is equivalent to swapping a dummy edge? No.
                # Just calculate new C4s? Or use delta logic?
                # Adding (i,j) changes N(i) and N(j).
                # Affected pairs involve i, j and their neighbors.
                # Let's just use the swap logic by pretending we swap a non-existent edge?
                # Easier: just count new C4s directly for the initial step since it's only 374 non-edges.
                
                # New C4s = C(common_neighbors, 2)
                # Common neighbors of i and j
                common = bin(masks[i] & masks[j]).count("1")
                new_c4s = common * (common - 1) // 2
                if new_c4s < best_delta:
                    best_delta = new_c4s
                    best_edge = (i, j)
                    
    # Add the best edge
    if best_edge:
        i, j = best_edge
        adj[i, j] = 1
        adj[j, i] = 1
        masks = get_neighbor_masks(adj)
        neighbors = [list(np.where(adj[i] == 1)[0]) for i in range(n)]
        current_c4s = count_c4s(adj, neighbors)
        
    # Local Search
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 1]
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j] == 0]
    
    # Shuffle edges and non_edges for random selection
    random.shuffle(edges)
    random.shuffle(non_edges)
    
    improved = True
    iteration = 0
    max_iterations = 500000 # Can do many more now
    
    while time.time() < deadline - 5:
        if current_c4s <= 1:
            break
            
        # Try a batch of random swaps
        batch_size = 1000
        for _ in range(batch_size):
            if time.time() > deadline - 5:
                break
                
            # Pick random edge and non-edge
            e_idx = random.randint(0, len(edges)-1)
            u, v = edges[e_idx]
            
            ne_idx = random.randint(0, len(non_edges)-1)
            i, j = non_edges[ne_idx]
            
            # Check if this swap is valid (u,v) != (i,j) and not adjacent in a way that breaks simple swap?
            # (u,v) is edge, (i,j) is non-edge. They are distinct.
            # Can u or v be i or j?
            # If u == i, then we are removing (i,v) and adding (i,j).
            # This is a valid swap.
            
            delta = delta_swap(masks, u, v, i, j)
            
            if delta < 0:
                # Accept
                current_c4s += delta
                # Update adj
                adj[u, v] = 0
                adj[v, u] = 0
                adj[i, j] = 1
                adj[j, i] = 1
                # Update masks
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
                
                # Update edges and non_edges lists
                edges[e_idx] = (i, j)
                non_edges[ne_idx] = (u, v)
                
                # Optional: restart randomization if stuck?
                # For now, just continue.
                
    # Return edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 1:
                result.append([i, j])
                
    return result

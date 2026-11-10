import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n = 31
    target_edges = 91
    
    # Build a Cayley graph on Z_5 x Z_5
    # Vertices are (i, j) for i, j in 0..4, plus one extra vertex
    # Actually, let's use 31 vertices indexed 0..30
    # Map vertex k to (k % 5, k // 5) for k = 0..24, and handle 25..30 separately
    
    # Better approach: use the polarity graph of PG(2,5) as base, then optimize
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph base edges
    base_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                base_edges.add((i, j))
    
    # The polarity graph of PG(2,5) should have around 90 edges
    # Let's count
    num_base = len(base_edges)
    
    # All possible edges
    all_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.add((i, j))
    
    # We need exactly 91 edges
    # Start with base edges, add/remove to get to 91
    
    if num_base <= target_edges:
        current_edges = set(base_edges)
        # Add edges greedily to minimize C4
        candidates = list(all_pairs - base_edges)
        random.shuffle(candidates)
        
        # Use efficient C4 counting with adjacency matrix
        adj = np.zeros((n, n), dtype=np.int32)
        for u, v in current_edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        def count_c4(adj):
            """Count C4s using matrix multiplication: C4 count = (1/8) * sum_{i,j} (A^2)_{ij} * (A^2)_{ji} - ... 
            Actually: number of C4 = (1/8) * (sum of all entries of A^2 squared - 4*sum of entries of A^3 - ...)
            Simpler: for each pair (i,j), count common neighbors, then C4 = sum over pairs of C(common, 2) / 2? No.
            
            Correct formula: A C4 is a set of 4 vertices {a,b,c,d} with edges ab, bc, cd, da.
            For each pair of non-adjacent vertices (a,c), the number of C4s through them is C(common_neighbors(a,c), 2).
            But this counts each C4 twice (once for each diagonal pair).
            So total C4 = (1/2) * sum over all pairs (a,c) of C(|N(a) ∩ N(c)|, 2)
            where the sum is over all pairs, and we need a,c to be non-adjacent for the C4 to be proper.
            
            Actually, if a and c are adjacent, then a-b-c-d-a would have a and c both in the C4 and adjacent, 
            which means it's not a simple C4. So we should only count pairs (a,c) that are NOT adjacent.
            
            Let's use: C4 = (1/2) * sum_{a<c, a not adj c} C(|N(a) ∩ N(c)|, 2)
            """
            count = 0
            A2 = adj @ adj  # (A^2)_{ij} = number of common neighbors of i and j
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j] == 0:  # i and j not adjacent
                        common = A2[i, j]
                        if common >= 2:
                            count += common * (common - 1) // 2
            return count // 2  # Each C4 counted twice (two diagonals)
        
        def c4_delta_add_edge(u, v, adj, A2):
            """Compute change in C4 count when adding edge (u,v)."""
            # When we add edge (u,v), we need to:
            # 1. Remove C4s that used (u,v) as a diagonal (u,v non-adjacent before)
            # 2. Add C4s that use (u,v) as an edge
            # This is complex. Let's just recompute for simplicity if graph is small.
            # For n=31, A2 computation is O(n^3) = ~30K ops, very fast.
            pass
        
        # Given time constraints, let's use a simpler approach:
        # 1. Start with polarity graph
        # 2. If we have too few edges, add random edges
        # 3. If we have too many, remove random edges
        # 4. Do local search: swap edges to reduce C4
        
        # First, get to exactly 91 edges
        while len(current_edges) < target_edges:
            if time.time() > deadline:
                break
            added = False
            for edge in candidates:
                if edge not in current_edges:
                    current_edges.add(edge)
                    adj[edge[0], edge[1]] = 1
                    adj[edge[1], edge[0]] = 1
                    added = True
                    break
            if not added:
                break
        
        while len(current_edges) > target_edges:
            if time.time() > deadline:
                break
            # Remove a random edge
            edge_list = list(current_edges)
            edge_to_remove = random.choice(edge_list)
            current_edges.remove(edge_to_remove)
            adj[edge_to_remove[0], edge_to_remove[1]] = 0
            adj[edge_to_remove[1], edge_to_remove[0]] = 0
        
        # Now do local search: try to swap edges to reduce C4
        # Precompute A2 for fast C4 counting
        def compute_A2(adj):
            return adj @ adj
        
        A2 = compute_A2(adj)
        current_c4 = count_c4(adj)
        best_c4 = current_c4
        best_adj = adj.copy()
        
        # Local search: swap one edge with another
        all_edges_list = list(all_pairs)
        random.shuffle(all_edges_list)
        
        temperature = 20.0
        cooling = 0.999
        min_temp = 0.1
        
        for iteration in range(100000):
            if time.time() > deadline - 5:
                break
            if temperature < min_temp:
                break
            
            # Pick a random edge in current_edges to remove
            if not current_edges:
                break
            remove_edge = random.choice(list(current_edges))
            
            # Pick a random edge not in current_edges to add
            available = [e for e in all_edges_list if e not in current_edges]
            if not available:
                break
            add_edge = random.choice(available)
            
            # Compute delta C4
            # Temporarily modify adj
            old_u, old_v = remove_edge
            new_u, new_v = add_edge
            
            adj[old_u, old_v] = 0
            adj[old_v, old_u] = 0
            adj[new_u, new_v] = 1
            adj[new_v, new_u] = 1
            
            new_A2 = compute_A2(adj)
            new_c4 = count_c4(adj)
            delta = new_c4 - current_c4
            
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                # Accept
                current_edges.remove(remove_edge)
                current_edges.add(add_edge)
                current_c4 = new_c4
                A2 = new_A2
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj = adj.copy()
            else:
                # Revert
                adj[old_u, old_v] = 1
                adj[old_v, old_u] = 1
                adj[new_u, new_v] = 0
                adj[new_v, new_u] = 0
            
            temperature *= cooling
        
        # Extract edges from best_adj
        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if best_adj[i, j] == 1:
                    edges.add((i, j))
        
    else:
        # Too many base edges, need to remove some
        current_edges = set(base_edges)
        adj = np.zeros((n, n), dtype=np.int32)
        for u, v in current_edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        def count_c4(adj):
            count = 0
            A2 = adj @ adj
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j] == 0:
                        common = A2[i, j]
                        if common >= 2:
                            count += common * (common - 1) // 2
            return count // 2
        
        while len(current_edges) > target_edges:
            if time.time() > deadline:
                break
            edge_to_remove = random.choice(list(current_edges))
            current_edges.remove(edge_to_remove)
            adj[edge_to_remove[0], edge_to_remove[1]] = 0
            adj[edge_to_remove[1], edge_to_remove[0]] = 0
        
        # Local search
        A2 = compute_A2(adj) if 'compute_A2' in dir() else adj @ adj
        current_c4 = count_c4(adj)
        best_c4 = current_c4
        best_adj = adj.copy()
        
        all_edges_list = list(all_pairs)
        random.shuffle(all_edges_list)
        
        temperature = 20.0
        cooling = 0.999
        min_temp = 0.1
        
        for iteration in range(100000):
            if time.time() > deadline - 5:
                break
            if temperature < min_temp:
                break
            
            if not current_edges:
                break
            remove_edge = random.choice(list(current_edges))
            available = [e for e in all_edges_list if e not in current_edges]
            if not available:
                break
            add_edge = random.choice(available)
            
            old_u, old_v = remove_edge
            new_u, new_v = add_edge
            
            adj[old_u, old_v] = 0
            adj[old_v, old_u] = 0
            adj[new_u, new_v] = 1
            adj[new_v, new_u] = 1
            
            new_A2 = adj @ adj
            new_c4 = count_c4(adj)
            delta = new_c4 - current_c4
            
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_edges.remove(remove_edge)
                current_edges.add(add_edge)
                current_c4 = new_c4
                A2 = new_A2
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj = adj.copy()
            else:
                adj[old_u, old_v] = 1
                adj[old_v, old_u] = 1
                adj[new_u, new_v] = 0
                adj[new_v, new_u] = 0
            
            temperature *= cooling
        
        edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                if best_adj[i, j] == 1:
                    edges.add((i, j))
    
    # Ensure exactly 91 edges
    if len(edges) < target_edges:
        for edge in all_pairs:
            if len(edges) >= target_edges:
                break
            if edge not in edges:
                edges.add(edge)
    elif len(edges) > target_edges:
        edge_list = list(edges)
        random.shuffle(edge_list)
        edges = set(edge_list[:target_edges])
    
    result = [[u, v] for u, v in edges]
    return result

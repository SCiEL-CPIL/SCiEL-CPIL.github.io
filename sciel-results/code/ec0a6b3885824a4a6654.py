import time
import itertools
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    n = 16
    total_possible_edges = n * (n - 1) // 2  # 120
    
    # Seed: K_{3,13}
    def seed_edges():
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    best_edges = seed_edges()
    best_m = len(best_edges)
    
    # Q3 structure: 3-cube
    # Vertices 0..7, edges where hamming distance is 1
    Q3_EDGES = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # Precompute Q3 adjacency
    q3_adj = [set() for _ in range(8)]
    for u, v in Q3_EDGES:
        q3_adj[u].add(v)
        q3_adj[v].add(u)
        
    # Precompute Q3 neighbors as lists for faster access
    q3_neighbors = [list(q3_adj[i]) for i in range(8)]

    def find_q3(adj_mask):
        """
        Find a Q3 in the graph defined by adj_mask.
        Returns a list of 12 edges (as tuples) if found, else None.
        Uses bitmask-based backtracking.
        """
        # Iterate over all 8-subsets
        # Optimization: Only consider subsets with min degree >= 3
        for S in itertools.combinations(range(n), 8):
            S_list = list(S)
            S_set = set(S)
            
            # Quick filter: min degree in induced subgraph must be >= 3
            min_deg = 8
            for v in S_list:
                deg = 0
                mask = adj_mask[v]
                # Count neighbors in S
                # Using bit operations is faster
                # But S is small, simple loop is okay
                for u in S_list:
                    if u != v and (mask >> u) & 1:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
            if min_deg < 3:
                continue
                
            # Backtracking search for Q3 isomorphism
            mapping = [-1] * 8
            # We assign Q3 vertices 0..7 to graph vertices in S
            # Order of assignment: 0, 1, 2...
            # To speed up, we can fix vertex 0 of Q3 to the first vertex in S?
            # Q3 is vertex transitive, so we can fix mapping[0] = S_list[0]
            # This reduces branching by factor 8.
            
            # Let's try fixing mapping[0] = S_list[0]
            mapping[0] = S_list[0]
            
            # Available vertices for Q3 vertices 1..7
            # We need to assign Q3 vertices 1..7 to S_list[1..7]
            
            # Precompute which Q3 vertices are neighbors of which
            # We will use a recursive search
            
            def dfs(idx):
                if idx == 8:
                    return True
                
                # Try assigning Q3 vertex 'idx' to each available vertex in S_list
                # We must ensure it's not already used
                # We can maintain a used mask for S_list vertices
                
                # Optimization: Check constraints with already assigned neighbors
                assigned_neighbors = [nb for nb in q3_neighbors[idx] if nb < idx]
                
                # Iterate over S_list vertices that are not used
                # To avoid checking 'used' status repeatedly, we can pass a mask
                # But for simplicity and speed in Python, a list of available indices is good
                
                # We'll just iterate S_list and check if used
                for v in S_list:
                    # Check if v is already used
                    if v in mapping: # This is O(8) check, slow
                        continue
                    # Better: maintain a boolean array or set
                    # Let's use a set of used vertices
                    # But creating a set in each call is slow.
                    # Let's pass a 'used_mask' of S_list vertices?
                    # S_list indices 0..7 correspond to vertices S_list[0]..S_list[7]
                    
                    # Let's stick to a simple 'used' list
                    # Actually, we can just check if v is in the current mapping values
                    # mapping values are graph vertex IDs.
                    # So:
                    used_verts = set(mapping[:idx])
                    if v in used_verts:
                        continue
                    
                    # Check adjacency constraints
                    valid = True
                    for nb in assigned_neighbors:
                        mapped_nb = mapping[nb]
                        if not (adj_mask[mapped_nb] >> v) & 1:
                            valid = False
                            break
                    
                    if valid:
                        mapping[idx] = v
                        if dfs(idx + 1):
                            return True
                        mapping[idx] = -1
                    
                return False
            
            if dfs(1):
                # Found Q3
                edges_in_q3 = []
                for u, v in Q3_EDGES:
                    edges_in_q3.append((mapping[u], mapping[v]))
                return edges_in_q3
        
        return None

    def get_edge_list(adj_mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Strategy: Greedy removal from K_{4,12}
    # This is denser than K_{3,13} but less dense than K16.
    
    if time.time() < deadline - 5.0:
        # Initialize K_{4,12}
        adj_mask = [0] * n
        for u in range(4):
            for v in range(4, 16):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        current_edges = 48
        
        # Greedy loop
        while time.time() < deadline - 2.0 and current_edges > best_m:
            found = find_q3(adj_mask)
            
            if found is None:
                if current_edges > best_m:
                    best_m = current_edges
                    best_edges = get_edge_list(adj_mask)
                break
            
            # Remove a random edge from the found Q3
            random.shuffle(found)
            u, v = found[0]
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            current_edges -= 1

    # Strategy 2: Try adding edges to the best found so far?
    # If we have time, try to add edges to best_edges
    if time.time() < deadline - 5.0:
        # Reconstruct adj_mask from best_edges
        adj_mask_best = [0] * n
        for u, v in best_edges:
            adj_mask_best[u] |= (1 << v)
            adj_mask_best[v] |= (1 << u)
            
        current_edges = best_m
        
        # List of missing edges
        missing_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask_best[u] >> v) & 1:
                    missing_edges.append((u, v))
        
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() > deadline - 2.0:
                break
            if current_edges <= best_m:
                break
                
            # Try adding edge (u, v)
            adj_mask_best[u] |= (1 << v)
            adj_mask_best[v] |= (1 << u)
            
            # Check if Q3-free
            # This is expensive. We can do a quick check:
            # Does adding (u,v) create a Q3?
            # We can just call find_q3.
            # But find_q3 is slow.
            # We can optimize: Only check if (u,v) is part of a Q3.
            # For now, just use find_q3.
            
            if find_q3(adj_mask_best) is None:
                current_edges += 1
                if current_edges > best_m:
                    best_m = current_edges
                    best_edges = get_edge_list(adj_mask_best)
            else:
                # Revert
                adj_mask_best[u] &= ~(1 << v)
                adj_mask_best[v] &= ~(1 << u)

    return best_edges

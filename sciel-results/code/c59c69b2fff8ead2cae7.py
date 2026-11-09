"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # 1. Generate the Grötzsch graph (Mycielski of C5) as the initial 4-chromatic core.
    # Vertices 0..10 form the Grötzsch graph. Vertices 11..15 are isolated.
    def get_groetzsch_edges():
        edges = []
        # C5 cycle: 0-1-2-3-4-0
        for i in range(5):
            edges.append([i, (i + 1) % 5])
        # Mycielski construction:
        # New vertices 5..9 connect to neighbors of 0..4 respectively
        for i in range(5):
            neighbors = [(i + 1) % 5, (i - 1) % 5]
            for n in neighbors:
                edges.append([5 + i, n])
        # New vertex 10 connects to 5..9
        for i in range(5):
            edges.append([10, 5 + i])
        return edges

    best_edges = get_groetzsch_edges()
    best_count = len(best_edges)
    
    # Helper to check if adding edge [u, v] maintains triangle-freeness
    def is_triangle_free(adj, u, v):
        # Check if u and v share a common neighbor
        # adj is a list of sets
        return len(adj[u] & adj[v]) == 0

    # Helper to check if graph is 4-chromatic (not 3-colorable) using bitset-based backtracking
    def is_not_3_colorable(adj):
        n = 16
        # Convert adjacency to bitmasks for speed
        adj_masks = [0] * n
        for u in range(n):
            mask = 0
            for v in adj[u]:
                mask |= (1 << v)
            adj_masks[u] = mask
        
        # Order vertices by degree (descending) for better pruning
        order = list(range(n))
        order.sort(key=lambda x: -len(adj[x]))
        
        # Map original vertex to position in order
        pos = [0] * n
        for i, v in enumerate(order):
            pos[v] = i
        
        # Rebuild adjacency masks in order
        ordered_adj = [0] * n
        for i, v in enumerate(order):
            mask = 0
            for u in adj[v]:
                mask |= (1 << pos[u])
            ordered_adj[i] = mask
        
        colors = [-1] * n
        # available_colors[v] = bitmask of available colors (bits 0,1,2)
        available = [(1 << 3) - 1] * n  # all 3 colors available initially
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Get available colors for this vertex
            avail = available[idx]
            if avail == 0:
                return False
            
            # Try each available color
            while avail:
                # Pick the lowest set bit
                c_bit = avail & (-avail)
                c = c_bit.bit_length() - 1
                avail ^= c_bit
                
                colors[v] = c
                # Update available colors for uncolored neighbors
                updated = []
                conflict = False
                for j in range(idx + 1, n):
                    u = order[j]
                    if ordered_adj[j] & (1 << idx):
                        # u is neighbor of v
                        new_avail = available[j] & ~(1 << c)
                        if new_avail == 0:
                            conflict = True
                            break
                        if new_avail != available[j]:
                            updated.append((j, available[j]))
                            available[j] = new_avail
                
                if not conflict:
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                colors[v] = -1
                for j, old_avail in updated:
                    available[j] = old_avail
            
            return False
        
        return not backtrack(0)

    # Helper to add edge if valid
    def try_add_edge(adj, edge_set, u, v):
        if u == v or u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return False
        if not is_triangle_free(adj, u, v):
            return False
        
        # Tentatively add
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))
        
        if is_not_3_colorable(adj):
            return True
        else:
            # Revert
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.remove((u, v))
            return False

    # Main search loop
    while time.time() < deadline:
        # Start from best known or random restart from Grötzsch
        if random.random() < 0.3:
            # Restart from Grötzsch
            base_edges = get_groetzsch_edges()
        else:
            # Restart from current best
            base_edges = list(best_edges)
        
        # Build adjacency and edge set
        adj = [set() for _ in range(16)]
        edge_set = set()
        for e in base_edges:
            u, v = e
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
        
        current_edges = list(base_edges)
        
        # Greedy addition phase
        # Generate all possible edges not currently present
        possible_edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                if (i, j) not in edge_set:
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add one edge
            # To make it more effective, we might try adding multiple edges if possible,
            # but simple greedy one-by-one is safer for validity.
            for u, v in possible_edges:
                if (u, v) in edge_set:
                    continue
                if try_add_edge(adj, edge_set, u, v):
                    current_edges.append([u, v])
                    improved = True
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = list(current_edges)
                    break # Restart scanning possible edges after an addition
        
        # If we found a better solution, update best
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
        # If we've hit a high number, we can stop early? No, keep searching until deadline.
        # But if we reach a known upper bound or very high number, we might just return.
        # Max edges in triangle-free graph on 16 vertices is floor(16^2/4) = 64.
        # But 4-chromatic constraint is tighter.
        
    return best_edges
# EVOLVE-BLOCK-END

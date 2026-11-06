import time
import random
import sys

def construct():
    n = 16
    deadline = time.time() + 110
    
    # Known 4-chromatic triangle-free graphs as seeds
    # We'll use a few known constructions or just start from a base and add edges.
    
    # A simple 4-chromatic triangle-free graph is the Grötzsch graph (11 vertices).
    # We can embed it in 16 vertices and add more edges.
    # Or we can use the Mycielski construction of C5 (Grötzsch) and add 5 more vertices.
    
    # Let's generate a random 4-chromatic triangle-free graph by starting with Grötzsch
    # and adding 5 isolated vertices, then adding edges greedily.
    
    # Grötzsch graph edges (0-indexed, 11 vertices)
    # Vertices 0-4 form C5
    # Vertices 5-9 are the "shadow" of 0-4
    # Vertex 10 is connected to 5-9
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # C5
        (0, 5), (1, 5), (1, 6), (2, 6), (2, 7),  # Shadow connections
        (3, 7), (3, 8), (4, 8), (4, 9), (0, 9),  # Shadow connections
        (5, 10), (6, 10), (7, 10), (8, 10), (9, 10)  # Top vertex
    ]
    
    def make_grotzsch_adj():
        adj = [0] * n
        for u, v in groetzsch_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def is_triangle_free_adj(adj, u, v):
        return (adj[u] & adj[v]) == 0
    
    def is_3_colorable(adj):
        # Fast backtracking with bitmasks
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            u = idx
            # Try each color 0, 1, 2
            for c in range(3):
                valid = True
                # Check if any neighbor has color c
                m = adj[u]
                while m:
                    v_bit = m & (-m)
                    v = v_bit.bit_length() - 1
                    m ^= v_bit
                    if color[v] == c:
                        valid = False
                        break
                
                if valid:
                    color[u] = c
                    if backtrack(idx + 1):
                        return True
                    color[u] = -1
            
            return False
            
        return backtrack(0)
    
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges
    
    def greedy_add_edges(adj, time_limit):
        added = 0
        possible = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] & (1 << v))]
        random.shuffle(possible)
        
        for u, v in possible:
            if time.time() > time_limit:
                break
            if is_triangle_free_adj(adj, u, v):
                # Adding edges preserves non-3-colorability
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added
    
    def local_search(adj, time_limit):
        # Try to add edges one by one, and if we can't, try to remove an edge
        # and add two different edges? This is complex.
        # For now, just greedy add.
        # A better local search: try all possible edge additions, pick the one
        # that allows the most future additions? Too slow.
        # Just greedy is fine for now.
        return greedy_add_edges(adj, time_limit)
    
    best_edges = []
    best_count = 0
    
    # Start with Grötzsch graph
    adj = make_grotzsch_adj()
    
    # Verify it's not 3-colorable
    if not is_3_colorable(adj):
        greedy_add_edges(adj, deadline - 10)
        edges = get_edges_from_adj(adj)
        best_count = len(edges)
        best_edges = edges
    
    # Now try random perturbations and restarts
    while time.time() < deadline - 5:
        # Start from a random 4-chromatic triangle-free graph
        # We can start from Grötzsch and randomly add/remove edges,
        # or start from a random graph and check if it's 4-chromatic.
        
        # Strategy: Start from current best, remove a random edge,
        # then greedily add edges.
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Remove 1-3 random edges
        num_remove = random.randint(1, 3)
        edges_list = list(best_edges)
        random.shuffle(edges_list)
        for i in range(min(num_remove, len(edges_list))):
            u, v = edges_list[i]
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        # Verify it's still not 3-colorable
        if not is_3_colorable(adj):
            greedy_add_edges(adj, deadline - 5)
            edges = get_edges_from_adj(adj)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
    
    # Final validation
    # Check triangle-free
    adj_final = [0] * n
    for u, v in best_edges:
        adj_final[u] |= (1 << v)
        adj_final[v] |= (1 << u)
    
    for u in range(n):
        for v in range(u + 1, n):
            if adj_final[u] & (1 << v):
                if adj_final[u] & adj_final[v]:
                    # Triangle found, fall back to Grötzsch
                    adj_final = make_grotzsch_adj()
                    best_edges = get_edges_from_adj(adj_final)
                    break
    
    return best_edges

import time
import random

def construct():
    n = 16
    deadline = time.time() + 110
    
    best_edges = []
    best_count = 0
    
    # Grötzsch graph structure
    # Vertices 0-10
    grotzsch_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (0,6), (1,5), (1,7), (2,6), (2,8),
        (3,7), (3,9), (4,5), (4,8), (0,9),
        (10,5), (10,6), (10,7), (10,8), (10,9)
    ]
    
    def get_grotzsch_adj():
        adj = [0] * 11
        for u, v in grotzsch_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    g_adj = get_grotzsch_adj()
    
    def count_edges_for_weights(weights):
        # weights is a list of 11 integers
        # Map Grotzsch vertices to sets of actual vertices
        # This is just for counting, not building
        count = 0
        for u, v in grotzsch_edges:
            count += weights[u] * weights[v]
        return count

    def optimize_weights():
        # Start with all 1s
        weights = [1] * 11
        # We have 16 - 11 = 5 units to distribute
        for _ in range(5):
            best_w = -1
            best_i = -1
            for i in range(11):
                # Calculate marginal gain
                gain = 0
                mask = g_adj[i]
                m = mask
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    gain += weights[j]
                    m ^= lsb
                if gain > best_w:
                    best_w = gain
                    best_i = i
            weights[best_i] += 1
        return weights

    weights = optimize_weights()
    
    def build_blowup(weights):
        # Assign vertices 0..15 to parts
        # Part i gets vertices offset[i] ... offset[i]+weights[i]-1
        offsets = []
        current = 0
        for w in weights:
            offsets.append(current)
            current += w
        
        adj = [0] * 16
        edges = []
        for u, v in grotzsch_edges:
            start_u = offsets[u]
            end_u = start_u + weights[u]
            start_v = offsets[v]
            end_v = start_v + weights[v]
            for i in range(start_u, end_u):
                for j in range(start_v, end_v):
                    if i == j: continue # Should not happen as u != v
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.append([i, j])
        return adj, edges

    adj, edges = build_blowup(weights)
    best_count = len(edges)
    best_edges = edges
    
    # Try to add more edges?
    # Check if we can add edges between non-adjacent parts
    # This is complex, so let's just do a greedy add on the current graph
    
    def is_triangle_free_add(adj, u, v):
        return (adj[u] & adj[v]) == 0

    def greedy_expand(adj):
        possible = []
        for u in range(16):
            for v in range(u + 1, 16):
                if not (adj[u] & (1 << v)):
                    possible.append((u, v))
        
        random.shuffle(possible)
        
        for u, v in possible:
            if is_triangle_free_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Reconstruct edges
        final_edges = []
        for u in range(16):
            mask = adj[u]
            mask_higher = mask & ~((1 << (u + 1)) - 1)
            m = mask_higher
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                final_edges.append([u, v])
                m ^= lsb
        return final_edges

    expanded_edges = greedy_expand(adj)
    if len(expanded_edges) > best_count:
        best_count = len(expanded_edges)
        best_edges = expanded_edges

    return best_edges
